"""Pile vs misalignment-eliciting prompts. The prompt distribution IS the result.

J is a prompt-average. Averaged over Pile web text -- on which these models
barely differ -- dJ carries nothing that distinguishes misalignment. Averaged
over the probes that actually elicit the behaviour, it does. Same models, same
metric, same code; only the expectation changed.

Panel 3 is the honest qualifier: the effect is not in the leading directions.
It appears only from k=8 and strengthens through k=64, which is why reading out
the top singular directions shows formatting and code junk rather than anything
about misalignment.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .em_delta import top_subspace, overlap, measured_null

LAY = [4, 8, 12, 16, 20]
PAIRS = [("medical", "financial"), ("medical", "sports"), ("medical", "control"),
         ("financial", "sports"), ("financial", "control"), ("sports", "control")]
MF = ("medical", "financial")


def load(d):
    return {n: torch.load(f"{d}/J_{n}.pt", map_location="cpu", weights_only=False)
            for n in ["base", "medical", "financial", "sports", "control"]}


def pair_overlaps(J, l, k=8):
    Jb = J["base"]["J"][l].float()
    S = {n: top_subspace(J[n]["J"][l].float() - Jb, k, "left")[0]
         for n in ["medical", "financial", "sports", "control"]}
    return {p: overlap(S[p[0]], S[p[1]]) for p in PAIRS}


def main(out=Path("out/em05/em05_prompts.png")):
    P, E = load("out/em05"), load("out/em05_emprompts")
    null8, _ = measured_null(896, 8)
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))

    # panels 1 and 2: same plot, two prompt distributions
    for ax, J, name in [(axes[0], P, "Pile web text"),
                        (axes[1], E, "misalignment probes (chat-formatted)")]:
        series = {p: [] for p in PAIRS}
        for l in LAY:
            o = pair_overlaps(J, l)
            for p in PAIRS:
                series[p].append(o[p])
        for p, v in series.items():
            is_mf = p == MF
            ax.plot(LAY, v, "o-" if is_mf else "o--",
                    c="#A83A63" if is_mf else "0.62",
                    lw=2.6 if is_mf else 1.2, ms=6 if is_mf else 4,
                    zorder=3 if is_mf else 2,
                    label="misaligned pair (med-fin)" if is_mf else
                          ("every other pair" if p == PAIRS[1] else None))
        ax.axhline(null8, ls=":", c="crimson", lw=1.2)
        ax.set_xlabel("layer"); ax.set_ylabel("top-8 subspace overlap of $\\Delta J$")
        ax.set_ylim(0, 0.70); ax.set_xticks(LAY); ax.grid(alpha=.25)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax.set_title(f"J averaged over {name}", fontsize=10.5)
    axes[0].text(0.5, 0.06, "no separation", transform=axes[0].transAxes,
                 fontsize=11, c="0.4", ha="center", style="italic")
    axes[1].text(0.5, 0.06, "misaligned pair on top at every layer",
                 transform=axes[1].transAxes, fontsize=11, c="#A83A63",
                 ha="center", style="italic")

    # panel 3: rank dependence on the EM prompts
    ax = axes[2]
    ks = [1, 2, 4, 8, 16, 32, 64]
    mf, mx = [], []
    for k in ks:
        acc = {p: [] for p in PAIRS}
        for l in LAY:
            o = pair_overlaps(E, l, k)
            for p in PAIRS:
                acc[p].append(o[p])
        m = {p: sum(v) / len(v) for p, v in acc.items()}
        mf.append(m[MF])
        mx.append(max(v for p, v in m.items() if p != MF))
    ax.semilogx(ks, mf, "o-", c="#A83A63", lw=2.6, ms=6, label="misaligned pair")
    ax.semilogx(ks, mx, "o--", c="0.5", lw=1.6, ms=5, label="best other pair")
    ax.fill_between(ks, mf, mx, where=[a > b for a, b in zip(mf, mx)],
                    color="#A83A63", alpha=.13, interpolate=True)
    ax.axvline(8, c="0.75", lw=1, ls="-")
    ax.text(8.4, 0.70, "effect appears\nfrom k=8", fontsize=8.5, c="0.4")
    ax.set_xlabel("k (subspace rank)"); ax.set_ylabel("overlap, mean over layers")
    ax.set_xticks(ks); ax.set_xticklabels(ks); ax.grid(alpha=.25)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title("Not in the leading directions\n"
                 "separation only from k=8, growing to k=64", fontsize=10.5)

    fig.suptitle("The prompt distribution is the result  ·  same models, same "
                 "metric, different expectation in $J = E_{prompt}[\\partial h / "
                 "\\partial h_l]$", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
