"""Show the hybrid working on text instead of reporting a pass rate.

The comparison said pseudo-inverse 58.3% against eigenvector 44.4%. That is a
statistic about 36 directions and it is not something anyone can picture. This
runs the same four methods on the same direction and prints what the model
writes, so the difference can be read rather than trusted.

Same setup as the pass-rate run: the target semantics are named by the
EIGENVECTOR's readout, every method is asked to deliver them, and a random
direction of equal norm is the control.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = ["The engineer opened the toolbox because",
           "The nurse looked at the chart and then",
           "After the operation the surgeon told the family that"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--eig", type=int, default=0)
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("out/hybrid_text.json"))
    a = ap.parse_args()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, norm = _find_blocks_and_norm(m)
    W_U = m.get_output_embeddings().weight.detach()
    J = blob["J"][a.layer].float()
    U, S, Vh = torch.linalg.svd(J)
    w, V = torch.linalg.eig(J)
    j = int(torch.argsort(w.abs(), descending=True)[a.eig])
    ve = V[:, j].real; ve = ve / ve.norm()
    c = U.T @ ve
    i_close = int(c.abs().argmax())
    k = min(a.topk, S.numel())
    pinv = Vh[:k].T @ (c[:k] / S[:k]); pinv = pinv / pinv.norm()
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(J.shape[0], generator=g)
    cands = {"eigenvector": ve,
             "closest u (rank 1)": torch.sign(c[i_close]) * Vh[i_close],
             "pseudo-inverse": pinv,
             "random control": rnd}

    ids = tok(PROMPTS[0], return_tensors="pt")["input_ids"]
    with _MultiCapture(m, [a.layer], a.target) as cap:
        with torch.no_grad():
            m(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        hn = float(cap.h[a.layer][0].norm(dim=-1).mean())
    with torch.no_grad():
        t = (J @ ve).to(W_U.dtype)
        pos = int(torch.softmax(norm(t) @ W_U.T, -1).argmax())
        neg = int(torch.softmax(norm(-t) @ W_U.T, -1).argmax())
    print(f"layer {a.layer}, eigenvector {a.eig}   |lambda| = {w[j].abs():.2f}")
    print(f"its readout names the pair {tok.decode(pos)!r} / {tok.decode(neg)!r}")
    print(f"cos(best single u, eigenvector) = {c.abs().max():.3f}\n")

    def gen(p, n=15):
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            gg = m.generate(**e, max_new_tokens=n, do_sample=False,
                            pad_token_id=tok.eos_token_id)
        o = gg[0][e["input_ids"].shape[1]:]
        return (tok.decode(o).replace("\n", " ").strip(),
                1 - len(set(o.tolist())) / len(o))

    def frac(p):
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            pr = torch.softmax(m(**e).logits[0, -1].float(), -1)
        return pr[pos].item() / (pr[pos].item() + pr[neg].item())

    rec = {}
    for al in [0.01, 0.02]:
        print(f"\n{'#'*78}\n#  alpha = {al}\n{'#'*78}")
        rec[al] = {}
        for p in PROMPTS:
            b, _ = gen(p)
            print(f"\n  {p}…")
            print(f"    {'unmodified':<20} [{frac(p):.2f}]  {b[:60]}")
            rec[al][p] = {"base": [round(frac(p), 2), b]}
            for nm, d in cands.items():
                with AddDir(m, a.layer, d, al * hn):
                    txt, rep = gen(p); fr = frac(p)
                flag = "   ← degenerate" if rep > 0.30 else ""
                print(f"    {nm:<20} [{fr:.2f}]  {txt[:60]}{flag}")
                rec[al][p][nm] = [round(fr, 2), txt, round(rep, 2)]
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
