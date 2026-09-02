"""A clearer view of the trajectory: where the transport actually forms.

The single log-x plot compresses stages 2 and 3 into the right-hand margin,
which hides the whole point -- that those stages are short but do a lot. Two
panels instead: the trajectory, and the rate per stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .analyze import STAGE_OFFSET, TOTAL_STEPS, cos, deviation, load_dir, stage_of
from .lens import random_seeds

COLORS = {1: "#0B6E78", 2: "#C77B29", 3: "#8B3A62"}
NAMES = {1: "stage 1 · pretraining", 2: "stage 2 · midtraining",
         3: "stage 3 · long context"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/stages.png"))
    ap.add_argument("--floor", type=float, default=0.9735)
    args = ap.parse_args()

    items = [(s, r, L) for s, r, L in load_dir(args.dir) if r != "main"]
    V = random_seeds(items[-1][2].shape[1], 32, seed=0)
    dev_final = deviation(items[-1][2], V)

    pts = [(s, r, stage_of(r), cos(deviation(L, V), dev_final).mean().item())
           for s, r, L in items]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                 gridspec_kw={"width_ratios": [2.1, 1]})

    for st in (1, 2, 3):
        seg = [(s, c) for s, _, g, c in pts if g == st]
        ax.plot([max(s, 1) for s, _ in seg], [c for _, c in seg], "o-",
                c=COLORS[st], ms=5, lw=1.8, label=NAMES[st])
    ax.set_xscale("log")
    for b in (STAGE_OFFSET[2], STAGE_OFFSET[3]):
        ax.axvline(b, c="0.8", lw=1, zorder=0)
    ax.axhline(args.floor, ls=":", c="crimson", lw=1,
               label=f"noise floor ({args.floor:.3f})")
    ax.set_xlabel("cumulative training step (log)")
    ax.set_ylabel("similarity to final lens  $\\cos((J-I)^Tv)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("The transport forms late")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.2)

    # Rate per stage: this is the finding the log axis hides.
    ends = {1: 1_413_814, 2: 47_684, 3: 11_921}
    marks = {0: 0.00759, 1: 0.63777, 2: 0.85215, 3: 1.0}
    rates, gains = [], []
    for st in (1, 2, 3):
        gain = marks[st] - marks[st - 1]
        gains.append(gain)
        rates.append(gain / (ends[st] / 1e6))

    bars = bx.bar([NAMES[s].split(" · ")[1] for s in (1, 2, 3)], rates,
                  color=[COLORS[s] for s in (1, 2, 3)])
    for b, r, g, st in zip(bars, rates, gains, (1, 2, 3)):
        bx.text(b.get_x() + b.get_width() / 2, r + 0.35,
                f"{r:.1f}\n({100*ends[st]/TOTAL_STEPS:.1f}% of steps)",
                ha="center", fontsize=8, c="0.25")
    bx.set_ylabel("similarity gained per million steps")
    bx.set_title("Rate of change per stage")
    bx.set_ylim(0, max(rates) * 1.35)
    bx.grid(alpha=0.2, axis="y")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for s, r, st, c in pts:
        print(f"  {r:>20} stage{st}  step {s:>9,}  dev cos {c:.4f}")


if __name__ == "__main__":
    main()
