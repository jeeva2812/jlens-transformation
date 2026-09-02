"""Layer x position readout grid -- what the lens says, everywhere, at once.

Rows are layers (bottom = early), columns are token positions. Each cell shows
the top-1 token the lens reads out there, shaded by confidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/grid.png"))
    ap.add_argument("--positions", type=int, nargs="+",
                    default=[3, 4, 5, 9, 11, 13, 14, 16, 19, 20, 26, 30, 31])
    args = ap.parse_args()

    d = torch.load(args.file, map_location="cpu", weights_only=False)
    layers, toks, grid = d["layers"], d["tokens"], d["grid"]
    sel = [p for p in args.positions if p < len(toks)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("t", ["#FFFFFF", "#0B6E78"])
    conf = [[grid[l][p][0][1] for p in sel] for l in layers]

    fig, ax = plt.subplots(figsize=(1.15 * len(sel) + 2.2, 0.42 * len(layers) + 2.0))
    ax.imshow(conf, cmap=cmap, aspect="auto", vmin=0, vmax=1, origin="lower")

    for i, l in enumerate(layers):
        for j, p in enumerate(sel):
            w, pr = grid[l][p][0]
            lab = w.strip() or repr(w).strip("'")
            ax.text(j, i, lab[:11], ha="center", va="center", fontsize=7.2,
                    c="white" if pr > 0.5 else "#20303A")

    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([repr(toks[p]).strip("'") for p in sel],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_ylabel("layer  (target = 30)")
    ax.set_xlabel("token position in the prompt")
    ax.set_title("J-Lens readout by layer and position — Olmo 3 7B (main)\n"
                 "shading = top-1 confidence", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
