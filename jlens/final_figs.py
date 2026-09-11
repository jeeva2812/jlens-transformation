"""The figures for the write-up.  Reads saved JSON only -- runs no model.

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.final_figs

Writes into out/figs_final/:

    fig1_jacobian_vs_logitlens.png   is the Jacobian worth anything?
    fig2_slot_not_belief.png         the money figure: the push ignores what the model believed
    fig3_spectrum.png                no single singular direction is the concept
    fig4_contrast_layer_sweep.png    does it hold off Rome/Paris, and off one layer?
    fig5_headroom.png                is the semantic spread just headroom?
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out/figs_final")
MODELS = [
    ("SmolLM2-135M", "out/rare/smollm_rome_paris.json",
     "out/rare/smollm_rome_paris_facts.json", "out/rare/sweep_smollm2.json",
     "out/rare/headroom_smollm2.json"),
    ("Qwen3.5-4B", "out/rare/qwen35_4b_rome_paris.json",
     "out/rare/qwen35_4b_rome_paris_facts.json", "out/rare/sweep_qwen35_4b.json",
     "out/rare/headroom_qwen35_4b.json"),
    ("OLMo-3-7B", "out/rare/olmo3_7b_rome_paris.json",
     "out/rare/olmo3_7b_rome_paris_facts.json", "out/rare/sweep_olmo3_7b.json",
     "out/rare/headroom_olmo3_7b.json"),
]
INK, MUTED = "#1b1b1b", "#7a7a7a"
BLUE, ORANGE, GREY = "#2b6cb0", "#c05621", "#a0aec0"


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def fig1():
    """Grouped bars: logit-lens baseline vs the J-Lens pullback, matched norm."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names, width = [], .26
    for i, (name, main_path, _, _, _) in enumerate(MODELS):
        d = load(main_path)
        if d is None:
            continue
        m = d["methods"]
        names.append(name)
        direct = m["direct_w"]["mean_target_logit_contrast_change"]
        pull = m["pullback"]["mean_target_logit_contrast_change"]
        reverse = m["reverse_pullback"]["mean_target_logit_contrast_change"]
        null = d["random_null"]["distribution"]["target"]
        ax.bar(i - width, direct, width, color=GREY,
               label="w  (logit lens, no Jacobian)" if i == 0 else None)
        ax.bar(i, pull, width, color=BLUE,
               label="J$^T$w  (J-Lens pullback)" if i == 0 else None)
        ax.bar(i + width, reverse, width, color=ORANGE,
               label="$-$J$^T$w  (sign control)" if i == 0 else None)
        lo = null["mean"] - 2 * null["sd"]
        ax.add_patch(plt.Rectangle((i - .45, lo), .9, 4 * null["sd"],
                                   color="#d9534f", alpha=.18, zorder=0))
        ax.text(i, pull + .25, f"{pull / direct:.2f}×", ha="center",
                fontsize=9, color=BLUE, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Δ (logit “ Rome” − logit “ Paris”), nats")
    ax.set_title("The Jacobian buys 1.3–1.7×, not an order of magnitude\n"
                 "all directions injected at identical L2 norm; red band = ±2 sd of 30 random draws",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_jacobian_vs_logitlens.png", dpi=180)
    plt.close(fig)


def fig2():
    """Induced shift against what the model already believed.  Flatness is the finding."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, (name, _, facts_path, _, _) in zip(axes, MODELS):
        d = load(facts_path)
        if d is None:
            continue
        xs, ys = [], []
        for row in d["rows"]:
            clean = row["methods"]["clean"]["rome_minus_paris_answer_logit"]
            pull = row["methods"]["pullback"]["rome_minus_paris_answer_logit"]
            xs.append(clean)
            ys.append(pull - clean)
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        r = sxy / (sxx * syy) ** .5 if sxx and syy else 0.
        slope = sxy / sxx if sxx else 0.
        ax.scatter([x for x in xs if x < 0], [y for x, y in zip(xs, ys) if x < 0],
                   s=52, color=ORANGE, zorder=3, label="model said Paris")
        ax.scatter([x for x in xs if x >= 0], [y for x, y in zip(xs, ys) if x >= 0],
                   s=52, color=BLUE, zorder=3, label="model said Rome")
        lo, hi = min(xs) - 1, max(xs) + 1
        ax.plot([lo, hi], [my + slope * (lo - mx), my + slope * (hi - mx)],
                color=MUTED, lw=1.4, ls="--", zorder=2)
        ax.axvline(0, color=GREY, lw=.8, zorder=1)
        ax.set_title(f"{name}\nr = {r:+.2f}", fontsize=10, color=INK)
        ax.set_xlabel("clean logit gap (Rome − Paris)")
        style(ax)
    axes[0].set_ylabel("shift induced by J$^T$w, nats")
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("The push ignores what the model believed: a token-slot bias, not a belief edit",
                 fontsize=11, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "fig2_slot_not_belief.png", dpi=180)
    plt.close(fig)


def fig3():
    """How many singular components before the pullback's effect is recovered."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (name, main_path, _, _, _), colour in zip(MODELS, (GREY, BLUE, ORANGE)):
        d = load(main_path)
        if d is None:
            continue
        m = d["methods"]
        full = m["pullback"]["mean_target_logit_contrast_change"]
        ks = sorted(int(k.split("_")[1]) for k in m if k.startswith("top_"))
        ax.plot(ks, [100 * m[f"top_{k}"]["mean_target_logit_contrast_change"] / full
                     for k in ks], "o-", color=colour, label=name, lw=1.8, ms=5)
    ax.axhline(100, color=MUTED, ls=":", lw=1)
    ax.text(1.1, 102, "full pullback", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("singular components of J kept (k)")
    ax.set_ylabel("% of the full pullback's effect recovered")
    ax.set_title("No single singular direction is the concept\n"
                 "k = 1 gives 15% / 2% / −2%; the objective is spread over hundreds of components",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_spectrum.png", dpi=180)
    plt.close(fig)


def fig4():
    """Does pullback > direct hold for other contrasts, and at other layers?"""
    available = [(n, load(p)) for n, _, _, p, _ in MODELS]
    available = [(n, d) for n, d in available if d]
    if not available:
        print("  fig4 skipped: no sweep files yet")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(3.7 * len(available), 3.8),
                             sharey=True, squeeze=False)
    for ax, (name, d) in zip(axes[0], available):
        by_contrast = {}
        for cell in d["cells"]:
            by_contrast.setdefault(cell["contrast"], []).append(cell)
        for contrast, cells in by_contrast.items():
            cells.sort(key=lambda c: c["layer"])
            geographic = cells[0]["kind"] == "geography"
            ax.plot([c["layer"] for c in cells],
                    [c["ratio_pullback_over_direct"] for c in cells],
                    "o-", ms=3.5, lw=1.4,
                    color=BLUE if geographic else ORANGE,
                    alpha=.85, label=contrast.replace("_", "–"))
        ax.axhline(1, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("layer")
        ax.set_title(name, fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        style(ax)
    axes[0][0].set_ylabel("effect of J$^T$w ÷ effect of w")
    summary = [d["summary"] for _, d in available]
    wins = sum(s["pullback_beats_direct"] for s in summary)
    total = sum(s["n_cells"] for s in summary)
    fig.suptitle(f"Holds off Rome/Paris and off one layer: pullback beats the baseline in "
                 f"{wins}/{total} cells\nblue = geographic contrasts, orange = "
                 f"occupation / season / colour; the advantage shrinks toward the target layer",
                 fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .88])
    fig.savefig(OUT / "fig4_contrast_layer_sweep.png", dpi=180)
    plt.close(fig)


def fig5():
    """Do the held-out semantic groups beat what headroom alone predicts?"""
    available = [(n, load(p)) for n, _, _, _, p in MODELS]
    available = [(n, d) for n, d in available if d]
    if not available:
        print("  fig5 skipped: no headroom files yet")
        return
    labels = [("italy_heldout", "Italy\n(held out)", BLUE),
              ("france_heldout", "France\n(held out)", ORANGE),
              ("control_japan", "Japan\n(control)", GREY),
              ("control_china", "China\n(control)", GREY),
              ("control_spain", "Spain\n(control)", GREY),
              ("control_germany", "Germany\n(control)", GREY)]
    fig, axes = plt.subplots(1, len(available), figsize=(3.9 * len(available), 3.8),
                             sharey=True, squeeze=False)
    for ax, (name, d) in zip(axes[0], available):
        groups = d["groups"]
        xs = [i for i, (key, _, _) in enumerate(labels) if key in groups]
        ax.bar(xs, [groups[key]["residual"] for key, _, _ in labels if key in groups],
               color=[c for key, _, c in labels if key in groups], width=.62)
        ax.axhline(0, color=MUTED, lw=.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for key, lab, _ in labels if key in groups], fontsize=7.5)
        ax.set_title(f"{name}\nheadroom fit r = {d['headroom_fit']['r']:+.2f}",
                     fontsize=10, color=INK)
        style(ax)
    axes[0][0].set_ylabel("shift beyond what headroom predicts, nats")
    fig.suptitle("Headroom control: shift minus the vocabulary-wide fit on clean log-probability\n"
                 "positive = moved more than its rarity alone explains",
                 fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, .89])
    fig.savefig(OUT / "fig5_headroom.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, function in [("fig1", fig1), ("fig2", fig2), ("fig3", fig3),
                           ("fig4", fig4), ("fig5", fig5)]:
        function()
        print(f"  {name} done")
    print(f"\nwritten to {OUT}/")
