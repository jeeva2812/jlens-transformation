"""Re-decide an earlier result of ours, using a free operation.

Earlier finding (docs/GAUGE_RESULT.md, MoE row): OLMoE's router rows read ~2x
better than an orthonormal basis of their span, and the decomposition said the
CONE SHAPE carried it, not the identity of the rows.

But the router has an exact gauge freedom: softmax and top-k both ignore a
constant added to every expert's score, so W_gate -> W_gate + 1 v^T changes no
routing decision. 38.3% of OLMoE's router norm sits in that free direction, and
a shared component across all rows is exactly what produces a narrow cone.

So: centre the router (subtract the mean over experts -- free), re-run the same
readout comparison, and see whether the effect survives.
"""
from __future__ import annotations
import json
import torch
from pathlib import Path
from jlens.privilege import Weights, make_bases
from jlens.corpus_npmi import NPMI, build as build_npmi
from jlens.tokspace import EmbedCoherence

MODEL = "allenai/OLMoE-1B-7B-0924"


def main():
    w = Weights(MODEL)
    npmi = NPMI(build_npmi(MODEL, n_windows=24576), device="cpu")
    kept = torch.nonzero(npmi.remap >= 0).squeeze(1)
    emb = EmbedCoherence(MODEL, kept, npmi.df)
    W_U = w.get("lm_head.weight")[kept]
    W_U = (W_U - W_U.mean(0, keepdim=True)) * w.get("model.norm.weight").unsqueeze(0)

    gen = torch.Generator().manual_seed(0)
    out = {}
    for mode in ("raw", "centred"):
        rows = []
        for li in range(16):
            R = w.get(f"model.layers.{li}.mlp.gate.weight")        # (64, d)
            if mode == "centred":
                R = R - R.mean(0, keepdim=True)                    # the free operation
            A = R.T.contiguous()
            A = A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)
            rec = {}
            for name, M in make_bases(A, 16, gen).items():
                ids = kept[(M.T @ W_U.T).topk(10, 1).indices]
                _, dc = emb.score(ids)
                per = dc.reshape(-1, A.shape[1])
                rec[name] = {"cos": float(dc.mean()),
                             "q90": float(torch.quantile(per, 0.9, dim=1).mean())}
            rec["_layer"] = li
            rows.append(rec)
        bases = [k for k in rows[0] if k != "_layer"]
        agg = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in ("cos", "q90")}
               for k in bases}
        out[mode] = agg
        out[mode + "_perlayer"] = [{k: r[k]["cos"] for k in r if k != "_layer"} for r in rows]
        print(f"\n=== router matrix: {mode.upper()}  (16 layers, 64 rows each)")
        print(f"{'basis':14s} {'mean coh':>9s} {'q90':>9s}")
        for k in ("raw", "reparam_null", "gram_null", "svd", "orth_null"):
            print(f"{k:14s} {agg[k]['cos']:+9.4f} {agg[k]['q90']:+9.4f}")
        print(f"  raw - orthonormal   {agg['raw']['cos']-agg['orth_null']['cos']:+.4f}   "
              f"<- the effect we reported")
        print(f"  orbit - orthonormal {agg['reparam_null']['cos']-agg['orth_null']['cos']:+.4f}   "
              f"<- the 'cone' component")
    a, b = out["raw"], out["centred"]
    ea = a["raw"]["cos"] - a["orth_null"]["cos"]
    eb = b["raw"]["cos"] - b["orth_null"]["cos"]
    ca = a["reparam_null"]["cos"] - a["orth_null"]["cos"]
    cb = b["reparam_null"]["cos"] - b["orth_null"]["cos"]
    print(f"\n{'':22s} {'raw router':>12s} {'centred':>12s} {'change':>10s}")
    print(f"{'raw - orthonormal':22s} {ea:+12.4f} {eb:+12.4f} {100*(eb-ea)/abs(ea):+9.0f}%")
    print(f"{'cone (orbit - orth)':22s} {ca:+12.4f} {cb:+12.4f} {100*(cb-ca)/abs(ca):+9.0f}%")
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path("out/gauge/router_centre.json").write_text(json.dumps(out, indent=1))
    # per-layer bootstrap (n=16 layers)
    import numpy as np
    def boot(v, B=20000, seed=0):
        v = np.asarray(v); g = np.random.default_rng(seed)
        s = np.array([v[g.integers(0, len(v), len(v))].mean() for _ in range(B)])
        return v.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)
    print(f"\n{'contrast':30s} {'raw router':>26s} {'centred router':>26s}")
    for lab, f in [("identity  (raw - orbit)", lambda r: r["raw"] - r["reparam_null"]),
                   ("cone      (orbit - orth)", lambda r: r["reparam_null"] - r["orth_null"]),
                   ("total     (raw - orth)", lambda r: r["raw"] - r["orth_null"])]:
        a = boot([f(r) for r in out["raw_perlayer"]])
        b = boot([f(r) for r in out["centred_perlayer"]])
        print(f"{lab:30s} {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}] "
              f"{b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]")
    print("\n  95% bootstrap CIs over the 16 layers.")
    print("\nwrote out/gauge/router_centre.json")


if __name__ == "__main__":
    main()
