"""Generalized eigenvectors: max gain PER UNIT occupancy + steer test.

Solve J^T J v = lam C v (C = activation covariance + ridge).
Top solutions = directions the transport amplifies most among ones the model
actually uses. Compare vs pullback/raw/random on France->Rome (efficacy) +
unrelated (TV drift). Also scores each gen-eigvec's own Rome-Paris gap.

Run: PYTHONPATH=. .venv/bin/python -m jlens.geneig_rome --layer 12 --out out/geneig_rome.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--ntext", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/geneig_rome.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    J = torch.load(a.jall, map_location="cpu", weights_only=False)["J"][a.layer].float()
    g = torch.Generator().manual_seed(a.seed)

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]
    Hs = []
    with torch.no_grad():
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=64)
            store = {}

            def hook(m, i, o, d=store):
                d["h"] = (o if torch.is_tensor(o) else o[0]).detach()
                return None

            h = blocks[a.layer].register_forward_hook(hook)
            model(**e, use_cache=False)
            h.remove()
            Hs.append(store["h"][0].float())
    H = torch.cat(Hs, 0)
    Hc = H - H.mean(0, keepdim=True)
    C = (Hc.T @ Hc) / H.shape[0]
    ev, EV = torch.linalg.eigh(C)
    ev = ev.clamp(min=0)
    print(f"occupancy concentration: top PC share = {float(ev.max()/ev.sum()):.3f} "
          f"(>0.5 = massive-direction regime)")
    Ch = EV @ torch.diag((ev + a.ridge * ev.max()).sqrt()) @ EV.T   # C^{1/2}
    Chi = EV @ torch.diag(1.0 / (ev + a.ridge * ev.max()).sqrt()) @ EV.T  # C^{-1/2}

    # gen-eigvecs: v = C^{-1/2} u, u = right-singular of J C^{1/2}
    Uw, Sw, Vwh = torch.linalg.svd((J @ Ch).float())
    V = (Chi @ Vwh.T)  # cols = gen-eigvecs (unnormalized)
    V = V / V.norm(dim=0, keepdim=True)
    print("top generalized gains Sw:", [round(float(x), 2) for x in Sw[:5]])

    tid = tok.encode(" Paris", add_special_tokens=False)[0]
    fid = tok.encode(" Rome", add_special_tokens=False)[0]
    with torch.no_grad():
        T = J @ V  # transport each gen-eigvec
        logits = norm(T.T).to(W_U.dtype) @ W_U.T
        lp = torch.log_softmax(logits.float(), -1)
        gaps = (lp[:, fid] - lp[:, tid])
    order = torch.argsort(gaps, descending=True)
    print("top gen-eigvec gaps(Rome-Paris):",
          [(int(i), round(float(gaps[i]), 2)) for i in order[:6].tolist()])
    print("top-gain gen-eigvec gaps:", [round(float(gaps[i]), 2) for i in range(4)])

    # occupancy of each: x random baseline
    totE = H.pow(2).sum()
    d_model = H.shape[1]
    for i in list(order[:3].tolist()) + [0]:
        d = V[:, i] / V[:, i].norm()
        occ = float(((H @ d).pow(2).sum() / totE) * d_model)
        print(f"  genvec {i}: gap={float(gaps[i]):+.2f} occupancy={occ:.2f}x random")

    # steer test: best-gap gen-eigvec vs pullback vs raw vs random
    d_raw = (W_U[fid] - W_U[tid]); d_raw /= d_raw.norm()
    pb = (J.T @ d_raw); pb /= pb.norm()
    rnd = torch.randn(d_raw.shape[0], generator=g); rnd /= rnd.norm()
    gd = V[:, int(order[0])]; gd /= gd.norm()

    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    spos = len(ids) - 2  # last subject-ish: 'France' token (prompt ends 'is')
    with torch.no_grad():
        base = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
    base_f = float(base[fid])
    e0 = tok("The report was finished on Tuesday and", return_tensors="pt")["input_ids"]
    store = {}

    def hook2(m, i, o):
        store["h"] = (o if torch.is_tensor(o) else o[0]).detach()
        return None

    h = blocks[a.layer].register_forward_hook(hook2)
    with torch.no_grad():
        model(input_ids=e0, attention_mask=torch.ones_like(e0), use_cache=False)
    h.remove()
    hn = float(store["h"][0].norm(dim=-1).mean())

    print(f"\nsteer on {prompt!r} (subj pos {spos}, alpha={a.alpha}):")
    res = {}
    for name, dvec in [("geneig-best", gd), ("pullback", pb), ("raw", d_raw), ("random", rnd)]:
        def hook(m, i, o, d=dvec):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            t2[0, spos] += (a.alpha * hn * d).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
        hh = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        hh.remove()
        eff = float(lp[fid]) - base_f
        res[name] = round(eff, 3)
        print(f"  {name:<12} dlogP(Rome)={eff:+.3f} P(Rome)={math.exp(float(lp[fid])):.5f}")
    # unrelated drift for geneig-best vs pullback
    for name, dvec in [("geneig-best", gd), ("pullback", pb)]:
        e = tok("Two plus two equals", return_tensors="pt")
        with torch.no_grad():
            b = torch.softmax(model(**e).logits[0, -1].float(), -1)
            def hook(m, i, o, d=dvec):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t.clone()
                t2[0, -1] += (a.alpha * hn * d).to(t2.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
            hh = blocks[a.layer].register_forward_hook(hook)
            with torch.no_grad():
                q = torch.softmax(model(**e).logits[0, -1].float(), -1)
            hh.remove()
        print(f"  unrelated TV [{name}] = {float((q-b).abs().sum()/2):.3f}")

    a.out.write_text(json.dumps({"gains": [float(x) for x in Sw[:16]],
                                 "best_gaps": [(int(i), float(gaps[i])) for i in order[:8].tolist()],
                                 "steer": res}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
