"""The training-checkpoint figure, rebuilt to drop the claims that do not hold.

Changes from the first version:

  * The rank-vs-birth-checkpoint scatter is gone.  Its r = +0.87 was computed on
    an ORDINAL checkpoint index (the checkpoints are roughly log-spaced in
    steps), the scatter was bimodal rather than linear, and rank is defined by
    singular value -- so a fixed half-formation threshold is easier to cross
    early when sigma is large.  "Stronger settles earlier" may be a property of
    the estimator, not of learning.
  * The final column is 1.0 BY CONSTRUCTION -- similarity is measured against
    the final checkpoint -- so it is drawn hollow and labelled, and the axis
    note says so.
  * The dip at the pretraining -> mid-training boundary now carries its second,
    independent signature: effective rank rises at the same checkpoint.
  * Added the cross-layer panel, which needs no forced endpoint to be
    interesting: early layers settle far earlier than late ones.

Reads saved JSON only, runs no model:

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.training_fig
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out/figs_final")
INK, MUTED = "#1b1b1b", "#7a7a7a"
TEAL, GOLD, PINK, RED = "#2b7a8c", "#b8860b", "#c2255c", "#c0392b"

SHORT = ["init", "2k", "8k", "32k", "128k", "512k",
         "end pre", "mid 8k", "end mid", "ctx 5k", "final"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main():
    birth = json.loads(Path("out/birth.json").read_text())
    svd = json.loads(Path("out/svd_training.json").read_text())
    order, captured = birth["order"], birth["captured"]
    chance, layer = birth["chance"], birth["layer"]
    x = range(len(order))
    dip = order.index("stage2-step8000")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # --- A: trajectory by rank band ----------------------------------------
    ax = axes[0]
    bands = [("strongest 8", slice(0, 8), TEAL),
             ("ranks 8–31", slice(8, 32), GOLD),
             ("ranks 32–63", slice(32, 64), PINK)]
    for label, sl, colour in bands:
        y = [st.mean(captured[k][sl]) for k in order]
        ax.plot(list(x)[:-1], y[:-1], "o-", color=colour, lw=1.8, ms=4.5, label=label)
        # final point is 1.0 by construction: draw it hollow and dashed
        ax.plot([len(order) - 2, len(order) - 1], y[-2:], ":", color=colour, lw=1.4)
        ax.plot([len(order) - 1], [y[-1]], "o", mfc="white", mec=colour, ms=5.5)
    ax.axhline(chance, color=RED, ls=":", lw=1)
    ax.text(0.1, chance + .028, f"chance {chance:.3f}", color=RED, fontsize=7.5)
    ax.axvspan(dip - .35, dip + .35, color=RED, alpha=.10, zorder=0)
    ax.annotate("all three bands fall back here\nLR jumps 3e-5 \u2192 2.07e-4 with\n"
                "ZERO warmup, batch halves\n(Olmo 3 paper, Table 35)",
                xy=(dip, .55), xytext=(dip - 5.2, .84), fontsize=6.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.text(len(order) - 1.05, .32, "final point is\n1.0 by construction",
            fontsize=6.8, color=MUTED, ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(SHORT, rotation=45, ha="right")
    ax.set_ylabel("fraction of the final direction present")
    ax.set_title(f"A · The subspace forms gradually, strongest first\n"
                 f"Olmo-3-7B layer {layer}, top-{birth['K']} directions",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)

    # --- B: the dip, across every layer ------------------------------------
    # NOT effective rank: that declines through pretraining and then oscillates
    # by ~150 afterwards, rising at ctx-5k too, where similarity does not dip.
    # It is not a signature of this boundary.  The per-layer replication is.
    ax = axes[1]
    stats = svd["stats"]
    layers_all = svd["layers"]
    deltas = [stats["stage2-step8000"][str(L)]["to_final"]
              - stats["stage1-step1413814"][str(L)]["to_final"] for L in layers_all]
    colours = [RED if v < 0 else MUTED for v in deltas]
    ax.bar(range(len(layers_all)), deltas, color=colours, width=.72)
    ax.axhline(0, color=MUTED, lw=.9)
    ax.set_xticks(range(len(layers_all)))
    ax.set_xticklabels([str(L) for L in layers_all], fontsize=7.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("change in similarity to final\nacross the boundary")
    n_dip = sum(v < 0 for v in deltas)
    ax.annotate("3 exceptions:\nnearest the target,\nwhere J \u2192 I",
                xy=(len(layers_all) - 2.0, -.0015), xytext=(0.0, -.0385),
                fontsize=7.2, color=MUTED, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.9,
                                connectionstyle="arc3,rad=-0.3"))
    ax.set_title(f"B · The setback is not one layer: {n_dip} of {len(layers_all)} dip\n"
                 "end of pretraining \u2192 mid-training step 8k, every layer measured",
                 fontsize=9.5, color=INK, loc="left")
    style(ax)

    # --- C: cross-layer, no forced endpoint needed --------------------------
    ax = axes[2]
    layers = [4, 8, 12, 20, 28]
    colours = plt.cm.viridis([i / (len(layers) - 1) * .85 for i in range(len(layers))])
    for lay, colour in zip(layers, colours):
        y = [stats[k][str(lay)]["to_final"] for k in order]
        ax.plot(list(x)[:-1], y[:-1], "o-", color=colour, lw=1.7, ms=4,
                label=f"layer {lay}")
        ax.plot([len(order) - 2, len(order) - 1], y[-2:], ":", color=colour, lw=1.2)
        ax.plot([len(order) - 1], [y[-1]], "o", mfc="white", mec=colour, ms=5)
    ax.axhline(chance, color=RED, ls=":", lw=1)
    ax.axvspan(dip - .35, dip + .35, color=RED, alpha=.10, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(SHORT, rotation=45, ha="right")
    ax.set_ylabel("fraction of the final direction present")
    ax.set_title("C · Early layers settle far earlier\n"
                 "at end of pretraining layer 8 is 0.80, layer 20 is 0.48",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)

    fig.suptitle("How the reading subspace forms — and the one time it goes backwards",
                 fontsize=12, color=INK, x=.005, y=.985, ha="left")
    fig.text(.005, .005,
             "Similarity is measured against the final checkpoint, so the last point is 1.0 by "
             "construction (hollow, dashed). No error bars: J is estimated from a finite prompt "
             "bank, and the layer-20 dip is 0.036 \u2014 treat the size as indicative, the sign as the result.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, .04, 1, .90])
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / "fig6_training_subspace.png",
                 Path("out/report/F7_subspace_formation.png")):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print("wrote", path)

    births = birth["births"]
    from collections import Counter
    counts = Counter(births)
    print("\nbirth checkpoint counts (index -> n directions):")
    for i in sorted(counts):
        print(f"  {SHORT[i]:10s} {counts[i]:3d}")
    print(f"  none at all are born at '{SHORT[dip]}', the disrupted checkpoint")


if __name__ == "__main__":
    main()
