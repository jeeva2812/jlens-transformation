"""What training does to J, measured WITHOUT reference to the final model.

F7 asks "how much of the final direction is present yet?", which needs the
finished model as a yardstick and is 1.0 at the end by construction.  This
figure uses only quantities computable at a single checkpoint, so every point
stands on its own and nothing is forced.

The quantity is the eigenvalue spectrum of J.  Because J^T is the gradient
propagator, |lambda| > 1 marks a direction along which a perturbation -- or a
gradient -- is amplified on its way to the output rather than damped.

Reads out/eigen_training.json and out/svd_training.json.  Runs no model.

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.training_spectrum_fig
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out/figs_final")
INK, MUTED, RED = "#1b1b1b", "#7a7a7a", "#c0392b"
COLOURS = {"8": "#2b7a8c", "16": "#b8860b", "24": "#c2255c"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main():
    eig = json.loads(Path("out/eigen_training.json").read_text())
    svd = json.loads(Path("out/svd_training.json").read_text())
    order = list(eig.keys())
    labels = [eig[k]["label"] for k in order]
    x = range(len(order))
    dip = order.index("stage2-step8000")
    n = eig[order[0]]["8"]["n"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # A: share of directions that amplify
    ax = axes[0]
    for L, colour in COLOURS.items():
        y = [100 * eig[k][L]["gt1"] / n for k in order]
        ax.plot(list(x), y, "o-", color=colour, lw=1.9, ms=4.5, label=f"layer {L}")
    ax.axhline(50, color=MUTED, ls=":", lw=1)
    ax.text(.1, 52, "half of all directions", fontsize=7.2, color=MUTED)
    ax.annotate("layer 8 reaches\nZERO by step 8k",
                xy=(2, 0.4), xytext=(4.2, 14), fontsize=7.5, color=COLOURS["8"],
                arrowprops=dict(arrowstyle="->", color=COLOURS["8"], lw=1))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("% of directions with |λ| > 1")
    ax.set_ylim(-3, 62)
    ax.set_title("A · At init half of J amplifies; training kills it\n"
                 "further from the output \u2192 more completely killed",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax)

    # B: mean spectral radius
    ax = axes[1]
    for L, colour in COLOURS.items():
        y = [eig[k][L]["mean_abs"] for k in order]
        ax.plot(list(x), y, "o-", color=colour, lw=1.9, ms=4.5, label=f"layer {L}")
    ax.axhline(1.0, color=RED, ls="--", lw=1.2)
    ax.text(3.0, 1.03, "|λ| = 1 : neither damped nor amplified",
            fontsize=7.2, color=RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean |λ| over all 4096 directions")
    ax.set_title("B · The network becomes a contraction\n"
                 "an untrained residual stack is neutral; a trained one damps",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    # C: the same setback, reference-free
    ax = axes[2]
    steps = list(range(1, len(order)))
    width = .38
    for i, (L, colour) in enumerate([("8", COLOURS["8"]), ("16", COLOURS["16"])]):
        y = [eig[order[j]][L]["mean_abs"] - eig[order[j - 1]][L]["mean_abs"]
             for j in steps]
        ax.bar([j + (i - .5) * width for j in steps], y, width,
               color=colour, label=f"layer {L}")
    ax.axhline(0, color=MUTED, lw=.9)
    ax.axvspan(dip - .5, dip + .5, color=RED, alpha=.10, zorder=0)
    ax.annotate("largest rebound after pretraining,\non the same boundary F7 flags\n"
                "(ctx 5k also rises, ~2.3× smaller)",
                xy=(dip - .2, .045), xytext=(2.6, -.27), fontsize=7.2, color=RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xticks(steps)
    ax.set_xticklabels([labels[j] for j in steps], rotation=45, ha="right")
    ax.set_ylabel("change in mean |λ| from the\nprevious checkpoint")
    ax.set_title("C · The same setback, no final model used\n"
                 "the network briefly becomes less contractive",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)

    fig.suptitle("What training does to J — measured without the finished model as a yardstick",
                 fontsize=12, color=INK, x=.005, y=.985, ha="left")
    fig.text(.005, .005,
             "Every quantity here is computable from a single checkpoint: no comparison to the "
             "final model, nothing pinned to 1.0 by construction. J^T is the gradient propagator, "
             "so |λ| > 1 is an amplifying — and exploding-gradient — channel.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, .04, 1, .90])
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / "fig7_training_spectrum.png",
                 Path("out/report/F7b_spectrum_reference_free.png")):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print("wrote", path)

    print("\n% of directions with |lambda| > 1")
    print(f"{'ckpt':10s}" + "".join(f"{'L'+L:>9s}" for L in COLOURS))
    for k in order:
        print(f"{eig[k]['label']:10s}"
              + "".join(f"{100 * eig[k][L]['gt1'] / n:8.1f}%" for L in COLOURS))


if __name__ == "__main__":
    main()
