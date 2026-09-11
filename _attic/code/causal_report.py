"""Experiment C tables: does the model's own basis ACT better, at every dose?"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from jlens.privilege_agg import boot

ORDER = ["pair_headcols", "attn_head", "pair_ovsvd", "attn_cross",
         "residual", "mlp_write", "router"]
LABEL = {"pair_headcols": "attn head cols*", "attn_head": "attn head cols (sub)",
         "pair_ovsvd": "OV circuit SVD", "attn_cross": "attn across heads",
         "residual": "residual basis", "mlp_write": "MLP neurons",
         "router": "MoE router rows"}


def load(pat="out/priv/causal_*.json"):
    rows = []
    for f in sorted(glob.glob(pat)):
        if any(k in f for k in ("pilot", "diag", "lowpow")):
            continue
        o = json.loads(Path(f).read_text())
        rows += o["rows"]
    return rows


def zblock(rows, title):
    """Mean within-group z (raw vs its own 8 Haar draws) with a bootstrap CI.

    Higher power than bootstrapping raw-minus-orbit across groups, because each
    cell is normalised by the null spread measured inside that same group.
    """
    alphas = sorted({r["alpha"] for r in rows})
    print(f"\n{title}")
    print(f"{'arm':22s} {'n':>3s}" + "".join(f"{'a=' + str(a):>22s}" for a in alphas))
    for arm in ORDER:
        cells, n = [], 0
        for a in alphas:
            g = [r for r in rows if r["arm"] == arm and r["alpha"] == a]
            if len(g) < 3:
                cells.append(f"{'-':>22s}"); continue
            n = len(g)
            z = [r["zq90_vs_reparam_null"] for r in g]
            m, lo, hi = boot(z)
            s_ = "*" if lo > 0 else ("-" if hi < 0 else " ")
            cells.append(f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]{s_}".rjust(22))
        if n:
            print(f"{LABEL[arm]:22s} {n:3d}" + "".join(cells))


def block(rows, key, title):
    alphas = sorted({r["alpha"] for r in rows})
    print(f"\n{title}")
    hdr = "".join(f"{'a=' + str(a):>26s}" for a in alphas)
    print(f"{'arm':22s} {'n':>3s}" + hdr)
    for arm in ORDER:
        cells = []
        n = 0
        for a in alphas:
            g = [r for r in rows if r["arm"] == arm and r["alpha"] == a]
            if len(g) < 3:
                cells.append(f"{'-':>26s}"); continue
            n = len(g)
            d = [r["raw"][key] - r["reparam_null"][key] for r in g]
            m, lo, hi = boot(d)
            s = "*" if lo > 0 else ("-" if hi < 0 else " ")
            cells.append(f"{m:+.4f}[{lo:+.4f},{hi:+.4f}]{s}".rjust(26))
        if n:
            print(f"{LABEL[arm]:22s} {n:3d}" + "".join(cells))


def main():
    rows = load()
    if not rows:
        print("no causal runs found"); return
    models = sorted({r["model"].split("/")[-1] for r in rows})
    print("=" * 104)
    print("C.  DO THE MODEL'S OWN DIRECTIONS *ACT* BETTER, OR ONLY *READ* BETTER?")
    print("=" * 104)
    print(f"   {len(rows)} group x dose cells, {len(models)} models: {', '.join(models)}")
    print("   score = coherence of the top-10 tokens the INTERVENTION promotes")
    print("   (Experiment B scored the top-10 the READOUT names; same metric, same arms)")
    zblock(rows, "-- PRIMARY: within-group z of raw vs its own null draws (q90) --")
    block(rows, "q90", "-- 90th percentile of per-direction coherence: raw - orbit --")
    block(rows, "cos", "-- mean over the 64 directions: raw - orbit --")
    print("\n  * / - = 95% bootstrap CI over groups excludes zero.")
    print("  orbit = A@H: for attn head cols this is EXACTLY the same function, so that")
    print("  row is a measured zero at every dose, not an assumption.")

    print("\n" + "-" * 104)
    print("   K1 (pre-registered): MLP arm vs the provable-null arm, per model and dose")
    print("-" * 104)
    print(f"{'model':24s} {'alpha':>6s} {'null arm':>10s} {'MLP arm':>10s} {'ratio':>8s} {'K1':>6s}")
    for m in models:
        for a in sorted({r["alpha"] for r in rows}):
            sel = [r for r in rows if r["model"].endswith(m) and r["alpha"] == a]
            nu = [r["raw"]["q90"] - r["reparam_null"]["q90"] for r in sel if r["arm"] == "pair_headcols"]
            ml = [r["raw"]["q90"] - r["reparam_null"]["q90"] for r in sel if r["arm"] == "mlp_write"]
            if not nu or not ml:
                continue
            rt = np.mean(ml) / abs(np.mean(nu)) if np.mean(nu) else float("inf")
            print(f"{m:24s} {a:6g} {np.mean(nu):+10.4f} {np.mean(ml):+10.4f} "
                  f"{rt:7.1f}x {'pass' if rt > 2 else 'FAIL':>6s}")

    print("\n" + "-" * 104)
    print("   CONCENTRATION: share of the induced positive logit mass in the top 10 tokens")
    print("-" * 104)
    print(f"{'arm':22s} {'raw':>9s} {'orbit':>9s} {'orth':>9s} {'raw/orbit':>10s}")
    for arm in ORDER:
        g = [r for r in rows if r["arm"] == arm]
        if len(g) < 3:
            continue
        rw = np.mean([r["raw"]["conc"] for r in g])
        ob = np.mean([r["reparam_null"]["conc"] for r in g])
        ot = np.mean([r["orth_null"]["conc"] for r in g])
        print(f"{LABEL[arm]:22s} {rw:9.4f} {ob:9.4f} {ot:9.4f} {rw/ob:9.2f}x")


if __name__ == "__main__":
    main()
