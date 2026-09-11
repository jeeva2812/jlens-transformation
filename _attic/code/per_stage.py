"""One panel per stage, each on its own linear axis.

A shared log axis is the right view for "when does this happen" and the wrong
one for "what shape does it have within a stage" -- stages 2 and 3 are 3.2% and
0.8% of training, so they collapse into the right-hand margin. Here each stage
gets its own linear x, scaled to its own length.

Every metric plotted is reference-free: nothing uses the final checkpoint, so
these are all quantities you could watch during a live training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .analyze import STAGE_OFFSET, cos, deviation, load_dir, stage_of
from .lens import random_seeds

COLORS = {1: "#0B6E78", 2: "#C77B29", 3: "#8B3A62"}
LENGTH = {1: 1_413_814, 2: 47_684, 3: 11_921}
TITLE = {1: "stage 1 · pretraining", 2: "stage 2 · midtraining",
         3: "stage 3 · long context"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/per_stage.png"))
    ap.add_argument("--floor", type=float, default=0.027,
                    help="noise floor, as 1-cosine")
    args = ap.parse_args()

    items = [(s, r, L) for s, r, L in load_dir(args.dir) if r != "main"]
    V = random_seeds(items[0][2].shape[1], 32, seed=0)

    per = {1: [], 2: [], 3: []}
    for step, rev, L in items:
        st = stage_of(rev)
        local = step - STAGE_OFFSET[st]
        per[st].append((local, cos(L, V).mean().item(),
                        deviation(L, V).norm(dim=-1).mean().item()))
    for st in per:
        per[st].sort()

    for st in (1, 2, 3):
        print(f"\n{TITLE[st]}  ({len(per[st])} checkpoints, {LENGTH[st]:,} steps)")
        print(f"  {'step':>10} {'identity':>9} {'|dev|':>8}")
        for s, i, d in per[st]:
            print(f"  {s:>10,} {i:9.4f} {d:8.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), sharey="row")
    for col, st in enumerate((1, 2, 3)):
        xs = [p[0] for p in per[st]]
        for row, idx, lab in [(0, 1, "cos$(J^Tv,v)$"), (1, 2, "$|(J-I)^Tv|$")]:
            ax = axes[row][col]
            ax.plot(xs, [p[idx] for p in per[st]], "o-", c=COLORS[st], ms=5, lw=1.8)
            ax.set_xlim(-LENGTH[st] * 0.04, LENGTH[st] * 1.04)
            ax.grid(alpha=0.2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(side == "none")
            if col == 0:
                ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(f"{TITLE[st]}\n{LENGTH[st]:,} steps", fontsize=10)
            else:
                ax.set_xlabel("step within stage")
        # noise band on the identity row, so scatter is readable as scatter
        ax0 = axes[0][col]
        lo = min(p[1] for p in per[st])
        ax0.axhspan(lo - args.floor, lo + args.floor, color="crimson",
                    alpha=0.07, zorder=0)

    fig.suptitle("Per-stage, linear axes — reference-free metrics "
                 "(red band = noise floor width)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
