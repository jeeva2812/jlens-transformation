"""Subspace formation, shown per direction rather than as a band average.

The band-average table says ranks 0-7 converge before ranks 32-63. That is a
summary of 64 individual trajectories, and it hides the two things worth seeing:
WHICH rank is born WHEN, and that the ordering is monotone rather than noisy.

Panel A is the full 64 x 11 matrix, one row per direction. Panel B is when each
direction crosses half-formed, against its rank. Panel C is the aggregate with
the mid-training dip marked, which is the one feature a reader would otherwise
dismiss as noise.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL, ROSE, GREY, AMBER = "#0B6E78", "#A83A63", "#9AA8AD", "#8A6410"
LAB = ["init", "2k", "8k", "32k", "128k", "512k", "end pre",
       "mid 8k", "end mid", "ctx 5k", "final"]


def main():
    b = json.load(open("out/birth.json"))
    order, K, chance = b["order"], b["K"], b["chance"]
    M = np.array([b["captured"][r] for r in order])          # [ckpt, rank]
    births = np.array(b["births"])

    fig = plt.figure(figsize=(15.4, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.15], wspace=0.28)

    # ---- A: every direction, every checkpoint ----
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M.T, aspect="auto", cmap="viridis", vmin=chance, vmax=1.0,
                   origin="lower", interpolation="nearest")
    ax.plot(births, np.arange(K), c="white", lw=1.6, alpha=.9)
    ax.plot(births, np.arange(K), c=ROSE, lw=1.0)
    ax.axvline(6.5, c="white", lw=1.4, ls="--", alpha=.85)
    ax.text(6.65, K*0.04, "post-training", color="white", fontsize=8.5, rotation=90)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(LAB, fontsize=7.5, rotation=55, ha="right")
    ax.set_ylabel("direction rank  (0 = strongest)")
    ax.set_title("A · Every direction, every checkpoint\n"
                 "red line = when each direction reaches half its final form",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, label="fraction of the final direction present",
                 shrink=.85, pad=.02)

    # ---- B: birth step vs rank ----
    ax = fig.add_subplot(gs[1])
    j = (np.random.default_rng(0).random(K) - .5) * 0.5   # jitter, else 34 dots stack
    ax.scatter(births + j, np.arange(K), s=22, c=TEAL, alpha=.55,
               edgecolors="none")
    z = np.polyfit(np.arange(K), births, 1)
    ax.plot(np.polyval(z, np.arange(K)), np.arange(K), c=TEAL, lw=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(LAB, fontsize=7.5, rotation=55, ha="right")
    ax.set_ylabel("direction rank"); ax.set_xlim(4.4, len(order)-1.4)
    ax.grid(alpha=.25)
    r = np.corrcoef(np.arange(K), births)[0, 1]
    ax.set_title(f"B · Rank order is preserved\n"
                 f"stronger directions settle earlier (r = {r:+.2f})", fontsize=10.5)

    # ---- C: the aggregate, with the dip ----
    ax = fig.add_subplot(gs[2])
    bands = [(0, 8, "ranks 0–7", TEAL), (8, 32, "ranks 8–31", AMBER),
             (32, 64, "ranks 32–63", ROSE)]
    x = np.arange(len(order))
    for a, bnd, lab, c in bands:
        ax.plot(x, M[:, a:bnd].mean(1), "o-", c=c, ms=4.5, lw=2, label=lab)
    ax.axhline(chance, ls=":", c="crimson", lw=1.2)
    ax.text(0.15, chance+0.015, f"chance {chance:.3f}", fontsize=8, c="crimson")
    ax.axvline(6.5, c="0.7", lw=1.2, ls="--")
    ax.annotate("post-training disrupts\nbefore it overshoots",
                xy=(7, M[7, :8].mean()), xytext=(0.4, 0.74),
                fontsize=8.5, color=ROSE,
                arrowprops=dict(arrowstyle="->", color=ROSE, lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7.5, rotation=55, ha="right")
    ax.set_ylabel("mean fraction present"); ax.set_ylim(0, 1.05)
    ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    n_post = int(((births >= 7) & (births <= 8)).sum())
    ax.set_title(f"C · {n_post} of {K} directions are born in mid-training\n"
                 "— 3.2% of total training steps", fontsize=10.5)

    fig.suptitle("How the reading subspace forms  ·  Olmo 3 7B, layer 20, top-64 "
                 "directions", fontsize=12.5, y=1.02)
    fig.savefig("out/report/F7_subspace_formation.png", dpi=150,
                bbox_inches="tight")
    print("wrote out/report/F7_subspace_formation.png")
    print(f"  rank/birth correlation r = {r:+.3f}")
    print(f"  born in mid-training: {n_post}/{K}")
    for i in [0, 1, 7, 20, 40, 63]:
        print(f"  rank {i:>2} born at {LAB[births[i]].replace(chr(10),' ')}"
              f"   readout {b['readouts'][str(i)][:4]}")


if __name__ == "__main__":
    main()
