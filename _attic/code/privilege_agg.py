"""Aggregate privilege runs: per-arm effect sizes with bootstrap CIs over groups."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

ORDER = ["attn_head", "pair_headcols", "pair_ovsvd", "attn_cross", "residual",
         "mlp_read", "mlp_write", "router", "embed", "unembed"]


def load(paths):
    rows = []
    for p in paths:
        o = json.loads(Path(p).read_text())
        for r in o["rows"]:
            r["model"] = o["model"]; r["seed"] = o.get("seed", 0)
            rows.append(r)
    return rows


def boot(x, f=np.mean, B=10000, seed=0):
    g = np.random.default_rng(seed)
    x = np.asarray(x)
    s = np.array([f(x[g.integers(0, len(x), len(x))]) for _ in range(B)])
    return float(f(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarise(rows, label=""):
    print(f"\n=== {label}  ({len({r['model'] for r in rows})} model(s), "
          f"{len(rows)} groups)")
    print(f"{'arm':11s} {'n':>3s} {'raw-gram':>18s} {'ratio':>16s} "
          f"{'z':>7s} {'SVD keeps':>11s} {'dlogdf':>7s}")
    out = {}
    for arm in ORDER:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        d = [r["raw"]["cos"] - r["reparam_null"]["cos"] for r in rs]
        ratio = [r["raw"]["cos"] / r["reparam_null"]["cos"] for r in rs
                 if r["reparam_null"]["cos"] > 1e-4]
        z = [r["z_vs_reparam_null"] for r in rs]
        keep = [(r["svd"]["cos"] - r["reparam_null"]["cos"]) /
                (r["raw"]["cos"] - r["reparam_null"]["cos"]) for r in rs
                if r["raw"]["cos"] - r["reparam_null"]["cos"] > 2e-3]
        dl = [r["raw"]["logdf"] - r["reparam_null"]["logdf"] for r in rs]
        m, lo, hi = boot(d)
        rm, rlo, rhi = boot(ratio) if ratio else (float("nan"),) * 3
        km = np.median(keep) if keep else float("nan")
        print(f"{arm:11s} {len(rs):3d} {m:+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{rm:5.2f} [{rlo:4.2f},{rhi:4.2f}] {np.mean(z):+7.2f} "
              f"{km*100:9.0f}% {np.mean(dl):+7.2f}")
        out[arm] = {"delta": m, "ci": [lo, hi], "ratio": rm, "ratio_ci": [rlo, rhi],
                    "mean_z": float(np.mean(z)), "svd_keeps": float(km),
                    "n_groups": len(rs), "dlogdf": float(np.mean(dl))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--by-model", action="store_true")
    a = ap.parse_args()
    paths = [p for g in a.globs for p in sorted(glob.glob(g))]
    rows = load(paths)
    if a.by_model:
        for m in sorted({r["model"] for r in rows}):
            summarise([r for r in rows if r["model"] == m], m)
    summarise(rows, "ALL")
