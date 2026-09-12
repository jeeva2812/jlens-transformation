"""Cross-layer subspace overlap across all 11 Olmo-3-7B checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RED = "#1b1b1b", "#7a7a7a", "#c0392b"
TEAL, GOLD = "#2b7a8c", "#b8860b"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    d = json.loads(Path("out/rare/layer_subspaces_traj.json").read_text())
    ck = d["checkpoints"]
    labels = list(ck.keys())
    x = range(len(labels))
    chance = d["chance"]["mean"]
    near = [ck[k]["adjacent_mean"] for k in labels]
    far = [ck[k]["far_mean"] for k in labels]
    flat = [ck[k]["sigma64_over_sigma1_mean"] for k in labels]
    dip = labels.index("mid 8k")
    spike = labels.index("2k")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    ax.plot(list(x), near, "o-", color=TEAL, lw=2, ms=5, label="2 layers apart")
    ax.axvspan(spike - .4, spike + .4, color=GOLD, alpha=.13, zorder=0)
    ax.axvspan(dip - .4, dip + .4, color=RED, alpha=.13, zorder=0)
    ax.annotate("training starts:\nlayers collapse\ntoward each other",
                xy=(spike + .35, near[spike] - .01), xytext=(spike + 1.6, .655),
                fontsize=7.2, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1))
    ax.annotate("data mix changes:\nit happens again",
                xy=(dip, near[dip] + .008), xytext=(dip - 3.6, .585),
                fontsize=7.2, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("overlap of top-64 subspaces")
    ax.set_title("A · Layers 2 apart: every regime change\nmakes them briefly more alike",
                 fontsize=9.5, color=INK, loc="left")
    style(ax)

    ax = axes[1]
    ax.plot(list(x), far, "o-", color=TEAL, lw=2, ms=5)
    ax.axhline(chance, color=RED, ls="--", lw=1.2)
    ax.text(.2, chance + .004, f"measured chance {chance:.4f}", fontsize=7.2, color=RED)
    ax.axvspan(spike - .4, spike + .4, color=GOLD, alpha=.13, zorder=0)
    ax.annotate("post-pretraining stages\nsteadily re-share",
                xy=(len(labels) - 2, far[-2]), xytext=(4.0, .098),
                fontsize=7.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.9))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("overlap of top-64 subspaces")
    ax.set_title("B · Layers ≥10 apart stay near-disjoint\nnever more than ~7× a 1.56% floor",
                 fontsize=9.5, color=INK, loc="left")
    style(ax)

    ax = axes[2]
    ax.plot(list(x), flat, "s-", color=GOLD, lw=2, ms=4.5)
    ax.axvspan(spike - .4, spike + .4, color=GOLD, alpha=.13, zorder=0)
    ax.annotate("sharpens once, early, then flat —\nso it does not explain\nthe mid-training bump",
                xy=(dip, flat[dip]), xytext=(2.2, .78), fontsize=7.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.9))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("σ₆₄ / σ₁   (1.0 = flat spectrum)")
    ax.set_title("C · The confound control\nhow well-determined 'top-64' even is",
                 fontsize=9.5, color=INK, loc="left")
    style(ax)

    fig.suptitle("Do layers share a subspace? All 11 Olmo-3-7B checkpoints",
                 fontsize=12, color=INK, x=.005, y=.985, ha="left")
    fig.text(.005, .005,
             "Overlap = mean squared canonical cosine between top-64 right singular subspaces of J; chance "
             "measured from 30 random orthonormal pairs.\nExact SVD throughout — randomised SVD agrees at "
             "0.9997 on the trained model but only 0.77 at init, where the spectrum is nearly flat.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, .06, 1, .90])
    for p in (Path("out/figs_final/fig9_layer_subspaces_traj.png"),
              Path("out/report/F8b_layer_subspaces_training.png")):
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180)
        print("wrote", p)

    print(f"\n{'checkpoint':10s} {'2-apart':>9s} {'>=10-apart':>11s} {'x chance':>9s} {'sigma64/1':>10s}")
    for i, k in enumerate(labels):
        print(f"{k:10s} {near[i]:9.3f} {far[i]:11.3f} {far[i]/chance:8.1f}x {flat[i]:10.3f}")


if __name__ == "__main__":
    main()
