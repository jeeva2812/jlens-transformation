"""Phase-1 labelling for SmolLM2 dJ batch (no steering).

dJ = J_step600 - J_step0 per layer (14 layers, excl target 28).
SVD: inject Vh[i]; input readout J_base@Vh[i]; output readout U[:,i].
Eigen: v=V[:,j].real; input J_base@v; output v.
Nulls per layer: null_in (300 random via J_base), null_out (300 random direct).
Threshold 99.9th pct per axis per side. Hypothesis "<axis> (in|out)".
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.label_axes import build_extended, held_out_filter

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
TOPK = 20
NNULL = 300
TOP_TOKENS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm_mod = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    norm_mod = norm_mod.eval()

    AX = build_extended(tok)
    print(f"{len(AX)} axes", flush=True)
    J0 = torch.load("out/ft/J_step0.pt", map_location="cpu", weights_only=False)["J"]
    J6 = torch.load("out/ft/J_step600.pt", map_location="cpu", weights_only=False)["J"]

    def logits_of(t):
        with torch.no_grad():
            return (norm_mod(t.float().unsqueeze(0)).squeeze(0) @ W_U.T)

    def top_tokens(z, k=TOP_TOKENS):
        ids = z.topk(k).indices.tolist()
        return [tok.decode([i]) for i in ids]

    for L in a.layers:
        Jb = J0[L].float()
        dJ = (J6[L].float() - Jb)
        print(f"\n=== layer {L} |dJ|/|J|={float(dJ.norm()/Jb.norm()):.3f} ===", flush=True)
        g = torch.Generator().manual_seed(2000 + L)
        Rin = torch.empty(NNULL, W_U.shape[0])
        Rout = torch.empty(NNULL, W_U.shape[0])
        for i in range(NNULL):
            r = torch.randn(Jb.shape[0], generator=g)
            Rin[i] = logits_of(Jb @ r)
            r2 = torch.randn(Jb.shape[0], generator=g)
            Rout[i] = logits_of(r2)
        print("  nulls ready", flush=True)

        U, S, Vh = torch.linalg.svd(dJ)
        try:
            w, V = torch.linalg.eig(dJ)
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
                if len(kept) >= TOPK:
                    break
        except Exception as e:
            print(f"  eig failed {e}", flush=True)
            w, V, kept, ku, lam = None, None, [], [], None

        fams = [("svd", [(i, Vh[i].clone(), U[:, i].clone(), float(S[i])) for i in range(TOPK)])]
        if w is not None:
            fams.append(("eigen", [(j, ku[k], ku[k].clone(), float(lam[j])) for k, j in enumerate(kept)]))

        for fam, dirs in fams:
            records = []
            for idx, vin, vout, sig in dirs:
                vin = vin / vin.norm()
                zin = logits_of(Jb @ vin)
                zout = logits_of(vout / vout.norm())
                tin_pos = top_tokens(zin); tin_neg = top_tokens(logits_of(-(Jb @ vin)))
                tout_pos = top_tokens(zout); tout_neg = top_tokens(logits_of(-vout))
                best = None
                # NOTE: held-out filtering uses the side's own readout tokens
                for side, z, R, toks in (("in", zin, Rin, tin_pos + tin_neg),
                                         ("out", zout, Rout, tout_pos + tout_neg)):
                    for ax_name, (A_all, B_all, kept_all) in AX.items():
                        keptp = held_out_filter(ax_name, kept_all, toks)
                        if len(keptp) < 4:
                            continue
                        lut = {p: (int(x), int(y)) for p, x, y in
                               zip(kept_all, A_all.tolist(), B_all.tolist())}
                        ids = [lut[p] for p in keptp]
                        Aa = torch.tensor([x[0] for x in ids])
                        Bb = torch.tensor([x[1] for x in ids])
                        score = float(z[Bb].mean() - z[Aa].mean())
                        ns = R[:, Bb].mean(1) - R[:, Aa].mean(1)
                        na = ns.abs()
                        thr = float(na.quantile(0.999))
                        pct = float((na >= abs(score)).float().mean().item() * 100)
                        ratio = abs(score) / max(thr, 1e-9)
                        if best is None or ratio > best[0]:
                            best = (ratio, f"{ax_name} ({side})", score, thr, pct, keptp, side)
                if best is None or best[0] <= 1.0:
                    rec = {"model": "smollm2-ft-step600-minus-step0", "matrix": "dJ",
                           "layer": L, "family": fam, "index": int(idx),
                           "sigma_or_lambda": sig,
                           "tokens_pos": tout_pos, "tokens_neg": tout_neg,
                           "tokens_in_pos": tin_pos, "tokens_in_neg": tin_neg,
                           "hypothesis": None, "pairs": [],
                           "score": best[2] if best else 0.0,
                           "null_threshold": best[3] if best else 0.0,
                           "null_pct_beating": best[4] if best else 100.0,
                           "verdict": "no-hypothesis", "steer": None}
                    if best:
                        rec["best_attempt"] = {"axis": best[1], "ratio": best[0]}
                else:
                    _, name, score, thr, pct, keptp, side = best
                    rec = {"model": "smollm2-delta", "matrix": "dJ",
                           "layer": L, "family": fam, "index": int(idx),
                           "sigma_or_lambda": sig,
                           "tokens_pos": tout_pos, "tokens_neg": tout_neg,
                           "tokens_in_pos": tin_pos, "tokens_in_neg": tin_neg,
                           "hypothesis": name, "pairs": [list(p) for p in keptp],
                           "score": score, "null_threshold": thr,
                           "null_pct_beating": pct, "verdict": "validated",
                           "steer": None}
                records.append(rec)
            outp = a.outdir / f"smollm2-delta_dJ_L{L}_{fam}.json"
            outp.write_text(json.dumps(records, indent=1))
            nv = sum(1 for r in records if r["verdict"] == "validated")
            print(f"  {fam}: {nv}/{len(records)} validated", flush=True)
            for r in records:
                if r["verdict"] == "validated":
                    print(f"    {fam}[{r['index']}] {r['hypothesis']} s={r['score']:.1f} thr={r['null_threshold']:.1f} "
                          f"out={r['tokens_pos'][:5]} in={r['tokens_in_pos'][:5]}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
