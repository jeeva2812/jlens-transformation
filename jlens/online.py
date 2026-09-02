"""Metrics you could compute DURING training, without knowing the endpoint.

Every curve so far has been "similarity to the final lens", which needs the
finished model. That is fine for a post-hoc study and useless as a training-time
signal, and it also builds the answer into the question: anything is on its way
to the final lens if you measure distance to the final lens.

These three need no reference:

  identity alignment   cos(J^T v, v)  -- how much the transport still just
      passes the residual through. Starts at 1.0 for an untrained model, by
      construction, and falls as the layer learns to do something.
  deviation size       |(J - I)^T v|  -- how big the learned part is.
  concentration        participation ratio of the deviation's singular values,
      as a fraction of the probe count: 1.0 means the change is spread evenly
      over all probed directions, near 0 means it lives in a few.

Plus velocity: cos between consecutive checkpoints, which is also reference-free
but is confounded by unequal step gaps, so it is reported per checkpoint pair
with its gap rather than plotted as a rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .analyze import STAGE_OFFSET, cos, deviation, load_dir, stage_of
from .lens import random_seeds

COLORS = {1: "#0B6E78", 2: "#C77B29", 3: "#8B3A62"}


def participation_ratio(M: torch.Tensor) -> float:
    """(sum s^2)^2 / sum s^4, normalised by the number of probes.

    Cheap proxy for "how many directions is this change using". Bounded above by
    the probe count (32 here), so a true effective rank would need the full
    Jacobian -- this only sees the change within the probed subspace.
    """
    s = torch.linalg.svdvals(M.float())
    pr = (s.pow(2).sum() ** 2) / s.pow(4).sum().clamp(min=1e-12)
    return float(pr / M.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/online.png"))
    args = ap.parse_args()

    items = [(s, r, L) for s, r, L in load_dir(args.dir) if r != "main"]
    V = random_seeds(items[0][2].shape[1], 32, seed=0)

    rows = []
    for i, (step, rev, L) in enumerate(items):
        dev = deviation(L, V)
        rows.append({
            "step": step, "rev": rev, "stage": stage_of(rev),
            "identity": cos(L, V).mean().item(),
            "devsize": dev.norm(dim=-1).mean().item(),
            "conc": participation_ratio(dev),
            "vel": cos(L, items[i - 1][2]).mean().item() if i else float("nan"),
            "gap": step - items[i - 1][0] if i else 0,
        })

    print(f"{'revision':>20} {'identity':>9} {'|dev|':>8} {'concen':>8} "
          f"{'vel':>7} {'gap':>10}")
    for r in rows:
        print(f"{r['rev']:>20} {r['identity']:9.4f} {r['devsize']:8.4f} "
              f"{r['conc']:8.4f} {r['vel']:7.4f} {r['gap']:>10,}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    specs = [
        ("identity", "cos$(J^Tv,\\ v)$", "Transport stops being a pass-through"),
        ("devsize", "$|(J-I)^Tv|$", "Size of the learned part"),
        ("conc", "participation ratio / 32", "Spread across directions"),
    ]
    for ax, (key, ylab, title) in zip(axes, specs):
        for st in (1, 2, 3):
            seg = [(r["step"], r[key]) for r in rows if r["stage"] == st]
            ax.plot([max(s, 1) for s, _ in seg], [v for _, v in seg], "o-",
                    c=COLORS[st], ms=4.5, lw=1.7)
        ax.set_xscale("log")
        for b in (STAGE_OFFSET[2], STAGE_OFFSET[3]):
            ax.axvline(b, c="0.85", lw=1, zorder=0)
        ax.set_xlabel("cumulative step (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].plot([], [], "o-", c=COLORS[1], label="stage 1")
    axes[0].plot([], [], "o-", c=COLORS[2], label="stage 2")
    axes[0].plot([], [], "o-", c=COLORS[3], label="stage 3")
    axes[0].legend(frameon=False, fontsize=8.5)

    fig.suptitle("Reference-free: computable during training, no final model needed",
                 fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
