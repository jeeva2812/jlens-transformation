"""Does a direction's readout predict what it does? By depth, and by dose."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
from collections import defaultdict

ORDER = ["pair_headcols", "attn_head", "pair_ovsvd", "attn_cross",
         "residual", "mlp_write", "router"]
LABEL = {"pair_headcols": "attn head cols", "attn_head": "attn head cols (sub)",
         "pair_ovsvd": "OV circuit SVD", "attn_cross": "attn across heads",
         "residual": "residual basis", "mlp_write": "MLP neurons",
         "router": "MoE router rows"}


def depth(pat="out/priv/agree_*.json"):
    print("=" * 96)
    print("D.  DOES THE LOGIT LENS PREDICT WHAT A DIRECTION ACTUALLY DOES?")
    print("=" * 96)
    print("    readout = top-10 of W_U(gamma*d)      action = top-10 of logits(h+ad) - logits(h)")
    for f in sorted(glob.glob(pat)):
        o = json.loads(Path(f).read_text()); rs = o["rows"]
        if not rs:
            continue
        nl = max(r["layer"] for r in rs) + 2
        print(f"\n{o['model']}  ({len(rs)} cells, alpha={sorted({r['alpha'] for r in rs})})")
        print(f"  {'depth':>12s} {'overlap':>9s} {'cos(pred,act)':>14s} {'common-mode':>12s} {'n':>3s}")
        bins = defaultdict(list)
        for r in rs:
            bins[min(3, int(4 * r["layer"] / nl))].append(r)
        for b in sorted(bins):
            g = bins[b]
            lo, hi = min(r["layer"] for r in g), max(r["layer"] for r in g)
            print(f"  L{lo:02d}-L{hi:02d}{'':>4s} "
                  f"{100*np.mean([r['overlap'] for r in g]):8.1f}% "
                  f"{np.mean([r['cos_pred_act'] for r in g]):14.3f} "
                  f"{np.mean([r['common_mode'] for r in g]):12.3f} {len(g):3d}")
        print(f"  {'-'*54}")
        print(f"  {'by component (last quarter of layers only)':<54s}")
        late = [r for r in rs if r["layer"] >= 0.75 * nl]
        for arm in ORDER:
            g = [r for r in late if r["arm"] == arm]
            if not g:
                continue
            print(f"  {LABEL[arm]:>20s} {100*np.mean([r['overlap'] for r in g]):8.1f}% "
                  f"{np.mean([r['cos_pred_act'] for r in g]):14.3f}")


def dose(f="out/priv/dose_smollm2.json"):
    if not Path(f).exists():
        return
    rs = json.loads(Path(f).read_text())["rows"]
    print("\n" + "=" * 96)
    print("E.  DOSE CALIBRATION: where is an activation edit still direction-specific?")
    print("=" * 96)
    als = sorted({r["alpha"] for r in rs})
    print(f"  {'alpha':>7s} {'|delta| top-10':>15s} {'common-mode':>13s} {'cos(pred,act)':>14s}  verdict")
    for a in als:
        g = [r for r in rs if r["alpha"] == a]
        cm = np.mean([r["common_mode"] for r in g])
        v = "linear, direction-specific" if cm < 0.05 else (
            "drifting" if cm < 0.2 else "SATURATED - all directions do the same thing")
        print(f"  {a:7g} {np.mean([r['mag_nats'] for r in g]):14.2f}n "
              f"{cm:13.3f} {np.mean([r['cos_pred_act'] for r in g]):14.3f}  {v}")
    print("\n  common-mode = mean pairwise cosine of the induced logit change across the 64")
    print("  directions. At high dose every direction pushes the model the same way, so no")
    print("  comparison between bases can discriminate. Experiment C is run at alpha=0.05.")


if __name__ == "__main__":
    depth(); dose()
