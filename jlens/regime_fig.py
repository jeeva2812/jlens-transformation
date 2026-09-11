"""Figure 6 for the write-up: J tracks regime, not topic, and conditioning is free.
Reads out/rare/regime_*.json only -- runs no model.

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.regime_fig
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out/figs_final")
BLUE, ORANGE, GREY, INK = "#2b6cb0", "#c05621", "#a0aec0", "#1b1b1b"
MODELS = [("SmolLM2-135M", "out/rare/regime_smollm2.json"),
          ("Qwen3.5-4B", "out/rare/regime_qwen35_4b.json")]

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    # left: centred cosine between food_en and each other group
    cmp = [("food_en", "same regime,\nsame topic\n(within group)"),
           ("abstract_en", "different\ntopic"),
           ("food_fr", "different\nlanguage"),
           ("code", "different\nregime (code)")]
    w = 0.38
    for i, (name, path) in enumerate(MODELS):
        g = json.load(open(path))["geometry"]["centred_group_means"]["food_en"]
        vals = [g[k] for k, _ in cmp]
        x = np.arange(len(cmp)) + (i - 0.5) * w
        a.bar(x, vals, w, color=[GREY, BLUE][i], label=name)
        for xi, v in zip(x, vals):
            a.text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center", fontsize=8, color=INK)
    a.axhline(0, color=INK, lw=0.8); a.margins(y=0.15)
    a.set_xticks(range(len(cmp))); a.set_xticklabels([t for _, t in cmp], fontsize=9)
    a.set_ylabel("centred cosine to the food_en prompts' J")
    a.set_title("Changing the topic barely moves J;\nchanging the language or regime does", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=9); style(a)
    # right: causal
    conds = [("own", "own prompt's J\n(oracle)"), ("own_group_loo", "same-regime\nlens (LOO)"),
             ("global_loo", "global lens\n(LOO)"), ("other_group", "wrong-regime\nlens"),
             ("direct_w", "w alone\n(no J)"), ("random", "random")]
    for i, (name, path) in enumerate(MODELS):
        d = json.load(open(path))
        c = d["causal"]; s = d["summary"]
        vals = [c[k]["mean"] for k, _ in conds]
        x = np.arange(len(conds)) + (i - 0.5) * w
        cols = [ORANGE if k == "own_group_loo" else ([GREY, BLUE][i]) for k, _ in conds]
        b.bar(x, vals, w, color=cols, alpha=1 if i else 0.85,
              label=f"{name}: conditional beats global {s['own_group_beats_global']}/{s['n_prompts']}")
        for xi, v in zip(x, vals):
            b.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    b.set_xticks(range(len(conds))); b.set_xticklabels([t for _, t in conds], fontsize=9)
    b.set_ylabel("Δ(logit target − contrast), nats, matched norm")
    b.set_title("Grouping the per-prompt pullbacks by regime before averaging\nbeats the global lens for free (orange)", fontsize=10, loc="left")
    b.legend(frameon=False, fontsize=8, loc="upper right"); style(b)
    fig.suptitle("J is a property of the machinery that is running, not of what the text is about", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_regime_conditioning.png", dpi=180)
    print("written", OUT / "fig6_regime_conditioning.png")

if __name__ == "__main__":
    main()
