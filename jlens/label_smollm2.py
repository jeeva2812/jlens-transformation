"""Phase-1 labelling for SmolLM2 J batch (no steering).

Units: (smollm2-step0, J, layers 4,8,12,16,20,24, families svd/eigen), top20 each.
Writes shards out/labels/smollm2-step0_J_L<layer>_<family>.json

Readout: z = norm(J @ d) @ W_U.T  (logits). score = mean(z[B]-z[A]).
Null: 300 random r~N(0,I) per layer, same formula, threshold = 99.9th pct of |null|.
Held-out: pairs where either side appears in top15 pos/neg readout are dropped
per direction; axis needs >=4 pairs remaining to be testable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.label_axes import build_extended, held_out_filter

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
J_PATH = Path("out/ft/J_step0.pt")
LAYERS = [4, 8, 12, 16, 20, 24]
TOPK = 20
NNULL = 300
TOP_TOKENS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    print("loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print("loading model (for W_U + norm)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm_mod = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    # norm module may be on meta? ensure cpu float
    norm_mod = norm_mod.eval()

    AX = build_extended(tok)
    print(f"{len(AX)} axes usable: {sorted(AX)}", flush=True)
    # pre-extract pair lists
    axis_pairs = {k: v[2] for k, v in AX.items()}

    blob = torch.load(J_PATH, map_location="cpu", weights_only=False)
    Jdict = blob["J"]

    def logits_of(t):
        with torch.no_grad():
            n = norm_mod(t.float().unsqueeze(0)).squeeze(0)
            return (n @ W_U.T)

    def top_tokens(z, k=TOP_TOKENS):
        ids = z.topk(k).indices.tolist()
        return [tok.decode([i]) for i in ids], ids

    for layer in a.layers:
        J = Jdict[layer].float()
        print(f"\n=== layer {layer} ===", flush=True)
        # --- null: 300 random transported logits ---
        g = torch.Generator().manual_seed(1000 + layer)
        R = torch.empty(NNULL, W_U.shape[0])
        for i in range(NNULL):
            r = torch.randn(J.shape[0], generator=g)
            R[i] = logits_of(J @ r)
        print(f"  null ready {tuple(R.shape)}", flush=True)

        # --- decompositions ---
        U, S, Vh = torch.linalg.svd(J)
        try:
            w, V = torch.linalg.eig(J)
            lam = w.abs()
            order = torch.argsort(lam, descending=True).tolist()
            # dedup complex-conjugate pairs: V[:,j].real is identical for
            # conjugates, so greedily keep only directions with |cos|<0.99
            # against already-kept ones, then take the top 20 unique.
            kept_idx, kept_vecs = [], []
            for j in order:
                v = V[:, j].real.clone()
                n = float(v.norm())
                if n < 1e-9:
                    continue
                v = v / n
                if any(abs(float(v @ u)) > 0.99 for u in kept_vecs):
                    continue
                kept_idx.append(j); kept_vecs.append(v)
                if len(kept_idx) >= TOPK:
                    break
            eo = kept_idx
        except Exception as e:
            print(f"  eig failed: {e}", flush=True)
            w, V, eo = None, None, []

        fams = [("svd", [(i, Vh[i].clone(), float(S[i])) for i in range(TOPK)])]
        if w is not None:
            fams.append(("eigen", [(j, V[:, j].real.clone(), float(lam[j])) for j in eo]))

        for fam, dirs in fams:
            records = []
            for idx, d, sig in dirs:
                d = d / d.norm()
                t = J @ d
                z = logits_of(t)
                zneg = logits_of(-t)
                toks_pos, _ = top_tokens(z)
                toks_neg, _ = top_tokens(zneg)
                readout_all = toks_pos + toks_neg

                best = None  # (ratio, name, score, thr, pct, kept_pairs)
                for ax_name, (A_all, B_all, kept_all) in AX.items():
                    kept = held_out_filter(ax_name, kept_all, readout_all)
                    if len(kept) < 4:
                        continue
                    # map kept pair strings back to ids
                    # build lookup from kept_all
                    lut = {p: (int(aa), int(bb)) for p, aa, bb in
                           zip(kept_all, A_all.tolist(), B_all.tolist())}
                    try:
                        ids = [lut[p] for p in kept]
                    except KeyError:
                        continue
                    Aa = torch.tensor([x[0] for x in ids])
                    Bb = torch.tensor([x[1] for x in ids])
                    score = float(z[Bb].mean() - z[Aa].mean())
                    null_scores = R[:, Bb].mean(dim=1) - R[:, Aa].mean(dim=1)
                    null_abs = null_scores.abs()
                    thr = float(null_abs.quantile(0.999).item())
                    pct = float((null_abs >= abs(score)).float().mean().item() * 100)
                    ratio = abs(score) / max(thr, 1e-9)
                    if best is None or ratio > best[0]:
                        best = (ratio, ax_name, score, thr, pct, kept)
                if best is None:
                    rec = {"model": "smollm2-ft-step0", "matrix": "J",
                           "layer": layer, "family": fam, "index": int(idx),
                           "sigma_or_lambda": sig,
                           "tokens_pos": toks_pos, "tokens_neg": toks_neg,
                           "hypothesis": None, "pairs": [],
                           "score": 0.0, "null_threshold": 0.0,
                           "null_pct_beating": 100.0, "verdict": "no-hypothesis",
                           "steer": None}
                else:
                    ratio, ax_name, score, thr, pct, kept = best
                    if ratio > 1.0:
                        rec = {"model": "smollm2-ft-step0", "matrix": "J",
                               "layer": layer, "family": fam, "index": int(idx),
                               "sigma_or_lambda": sig,
                               "tokens_pos": toks_pos, "tokens_neg": toks_neg,
                               "hypothesis": ax_name,
                               "pairs": [list(p) for p in kept],
                               "score": score, "null_threshold": thr,
                               "null_pct_beating": pct, "verdict": "validated",
                               "steer": None}
                    else:
                        rec = {"model": "smollm2-ft-step0", "matrix": "J",
                               "layer": layer, "family": fam, "index": int(idx),
                               "sigma_or_lambda": sig,
                               "tokens_pos": toks_pos, "tokens_neg": toks_neg,
                               "hypothesis": None, "pairs": [],
                               "score": score, "null_threshold": thr,
                               "null_pct_beating": pct, "verdict": "no-hypothesis",
                               "best_attempt": {"axis": ax_name, "ratio": ratio},
                               "steer": None}
                records.append(rec)
            outp = a.outdir / f"smollm2-step0_J_L{layer}_{fam}.json"
            outp.write_text(json.dumps(records, indent=1))
            nv = sum(1 for r in records if r["verdict"] == "validated")
            print(f"  {fam}: {nv}/{len(records)} validated -> {outp.name}", flush=True)
            for r in records:
                if r["verdict"] == "validated":
                    print(f"    {fam}[{r['index']}] sig={r['sigma_or_lambda']:.2f} "
                          f"{r['hypothesis']} score={r['score']:.1f} thr={r['null_threshold']:.1f} "
                          f"pct={r['null_pct_beating']:.2f} pos={r['tokens_pos'][:6]}", flush=True)
    print("\ndone")


if __name__ == "__main__":
    main()
