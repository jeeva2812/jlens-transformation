"""Generic J labelling for Qwen/Llama (same protocol as SmolLM2 J).

Usage: PYTHONPATH=. .venv/bin/python -m jlens.label_other --model qwen|llama
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.label_axes import build_extended, held_out_filter

CFGS = {
    "qwen": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
             "jpath": "out/em05_emprompts/J_base.pt",
             "tag": "qwen-base", "target": 22, "topk": 20, "nnull": 300},
    "llama": {"model_id": "unsloth/Llama-3.2-1B-Instruct",
              "jpath": "out/llama_em/J_base.pt",
              "tag": "llama-base", "target": 14, "topk": 20, "nnull": 150},
}
TOP_TOKENS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "llama"])
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    a = ap.parse_args()
    C = CFGS[a.model]
    a.outdir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(C["model_id"], trust_remote_code=True)
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(C["model_id"], dtype=torch.float32,
                                                 trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm_mod = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    norm_mod = norm_mod.eval().cpu()
    AX = build_extended(tok)
    print(f"{len(AX)} axes", flush=True)
    blob = torch.load(C["jpath"], map_location="cpu", weights_only=False)
    layers = sorted(blob["J"].keys())
    print("layers", layers, flush=True)

    def logits_of(t):
        with torch.no_grad():
            return (norm_mod(t.float().unsqueeze(0)).squeeze(0) @ W_U.T)

    for L in layers:
        J = blob["J"][L].float()
        print(f"=== layer {L} ===", flush=True)
        g = torch.Generator().manual_seed(3000 + int(L))
        R = torch.empty(C["nnull"], W_U.shape[0])
        for i in range(C["nnull"]):
            R[i] = logits_of(J @ torch.randn(J.shape[0], generator=g))
            if (i + 1) % 50 == 0:
                print(f"  null {i+1}/{C['nnull']}", flush=True)
        U, S, Vh = torch.linalg.svd(J)
        print("  svd done", flush=True)
        try:
            w, V = torch.linalg.eig(J)
            lam = w.abs()
            order = torch.argsort(lam, descending=True).tolist()
            kept, ku = [], []
            for j in order:
                v = V[:, j].real.clone()
                n = float(v.norm())
                if n < 1e-9:
                    continue
                v = v / n
                if any(abs(float(v @ u)) > 0.99 for u in ku):
                    continue
                kept.append(j); ku.append(v)
                if len(kept) >= C["topk"]:
                    break
        except Exception as e:
            print(f"  eig failed {e}", flush=True)
            w, V, kept, ku, lam = None, None, [], [], None
        fams = [("svd", [(i, Vh[i].clone(), float(S[i])) for i in range(C["topk"])])]
        if w is not None:
            fams.append(("eigen", [(j, ku[k], float(lam[j])) for k, j in enumerate(kept)]))
        for fam, dirs in fams:
            recs = []
            for idx, d, sig in dirs:
                d = d / d.norm()
                z = logits_of(J @ d)
                toks_pos = [tok.decode([i]) for i in z.topk(TOP_TOKENS).indices.tolist()]
                toks_neg = [tok.decode([i]) for i in logits_of(-(J @ d)).topk(TOP_TOKENS).indices.tolist()]
                rall = toks_pos + toks_neg
                best = None
                for name, (Aa0, Bb0, kp0) in AX.items():
                    kp = held_out_filter(name, kp0, rall)
                    if len(kp) < 4:
                        continue
                    lut = {p: (int(x), int(y)) for p, x, y in zip(kp0, Aa0.tolist(), Bb0.tolist())}
                    ids = [lut[p] for p in kp]
                    Aa = torch.tensor([x[0] for x in ids]); Bb = torch.tensor([x[1] for x in ids])
                    s = float(z[Bb].mean() - z[Aa].mean())
                    ns = R[:, Bb].mean(1) - R[:, Aa].mean(1)
                    na = ns.abs()
                    # 99.9th pct; for llama nnull=150 the 0.999 quantile interpolates near max
                    thr = float(na.quantile(0.999))
                    pct = float((na >= abs(s)).float().mean().item() * 100)
                    r_ = abs(s) / max(thr, 1e-9)
                    if best is None or r_ > best[0]:
                        best = (r_, name, s, thr, pct, kp)
                if best is None or best[0] <= 1.0:
                    recs.append({"model": C["tag"], "matrix": "J", "layer": int(L),
                                 "family": fam, "index": int(idx), "sigma_or_lambda": sig,
                                 "tokens_pos": toks_pos, "tokens_neg": toks_neg,
                                 "hypothesis": None, "pairs": [],
                                 "score": best[2] if best else 0.0,
                                 "null_threshold": best[3] if best else 0.0,
                                 "null_pct_beating": best[4] if best else 100.0,
                                 "verdict": "no-hypothesis", "steer": None,
                                 **({"best_attempt": {"axis": best[1], "ratio": best[0]}} if best else {})})
                else:
                    recs.append({"model": C["tag"], "matrix": "J", "layer": int(L),
                                 "family": fam, "index": int(idx), "sigma_or_lambda": sig,
                                 "tokens_pos": toks_pos, "tokens_neg": toks_neg,
                                 "hypothesis": best[1], "pairs": [list(p) for p in best[5]],
                                 "score": best[2], "null_threshold": best[3],
                                 "null_pct_beating": best[4], "verdict": "validated", "steer": None})
            outp = a.outdir / f"{C['tag']}_J_L{L}_{fam}.json"
            outp.write_text(json.dumps(recs, indent=1))
            nv = sum(1 for r in recs if r["verdict"] == "validated")
            print(f"  {fam}: {nv}/{len(recs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
