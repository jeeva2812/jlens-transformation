"""Application figures for the Rome-minus-Paris pullback experiment."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("out/rare")
FILES = [
    ("SmolLM2-135M", ROOT / "smollm_rome_paris.json", "#3977b8"),
    ("Qwen3.5-4B", ROOT / "qwen35_4b_rome_paris.json", "#e68632"),
    ("OLMo-3-7B", ROOT / "olmo3_7b_rome_paris.json", "#4b9b69"),
]
OUT = Path("out/figs_core")
OUT.mkdir(parents=True, exist_ok=True)

data = [(name, json.loads(path.read_text()), color) for name, path, color in FILES]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
x = np.arange(3)
width = .23
for j, (name, d, color) in enumerate(data):
    exact = d["methods"]["pullback"]["mean_target_logit_contrast_change"]
    related = d["methods"]["pullback"]["mean_heldout_related_logit_contrast_change"]
    direct = d["methods"]["direct_w"]["mean_target_logit_contrast_change"]
    ax.bar(x[j] - .75 * width, direct, width=width, color="#aaaaaa", alpha=.8)
    ax.bar(x[j], exact, width=width, color=color)
    ax.bar(x[j] + .75 * width, related, width=width, color=color, alpha=.48)
ax.set_xticks(x, [x[0] for x in FILES])
ax.set_ylabel("Mean change in logit contrast")
ax.set_title("A two-token objective generalizes to held-out associations")
ax.legend(handles=[plt.Rectangle((0,0),1,1,color="#aaaaaa",label="Direct w: Rome−Paris"),
                   plt.Rectangle((0,0),1,1,color="#555555",label="Pullback: Rome−Paris"),
                   plt.Rectangle((0,0),1,1,color="#555555",alpha=.48,label="Pullback: Italy−France (held out)")],
          frameon=False, fontsize=8)
ax.axhline(0, color="black", lw=.7)

ax = axes[1]
for name, d, color in data:
    ks = [1, 8, 32, 64, 128, 256]
    ys = [d["methods"][f"top_{k}"]["mean_target_logit_contrast_change"] for k in ks]
    full = d["methods"]["pullback"]["mean_target_logit_contrast_change"]
    ax.plot(ks + [512], ys + [full], marker="o", color=color, label=name)
ax.set_xscale("log", base=2)
ax.set_xticks([1,8,32,64,128,256,512], ["1","8","32","64","128","256","full"])
ax.set_xlabel("Leading singular components retained")
ax.set_ylabel("Rome−Paris logit contrast change")
ax.set_title("The effect is distributed, not a top-component trick")
ax.legend(frameon=False)
ax.axhline(0, color="black", lw=.7)
fig.tight_layout()
fig.savefig(OUT / "fig15_rome_paris_contrast.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for name, d, color in data:
    rows = d["prompt_rows"]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.axis("off")
    table_rows = []
    for row in rows:
        clean = row["methods"]["clean"]
        pb = row["methods"]["pullback"]
        table_rows.append([
            row["prompt"],
            f"{100*clean['rome_probability']:.3f}% / {100*clean['paris_probability']:.3f}%",
            f"{100*pb['rome_probability']:.3f}% / {100*pb['paris_probability']:.3f}%",
            f"{pb['target_logit_contrast_change']:+.2f}",
            f"{pb['heldout_related_contrast_change']:+.2f}",
        ])
    table = ax.table(cellText=table_rows,
                     colLabels=["Prompt", "Clean Rome / Paris", "Jᵀw Rome / Paris",
                                "Δ Rome−Paris", "Δ Italy−France\n(held out)"],
                     cellLoc="left", colLoc="left", loc="center",
                     colWidths=[.42,.17,.17,.11,.13])
    table.auto_set_font_size(False); table.set_fontsize(8.4); table.scale(1, 1.65)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(color); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax.set_title(f"{name}: every prompt in the +Rome / −Paris experiment",
                 fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    slug = name.lower().replace(".", "").replace("-", "_")
    fig.savefig(OUT / f"fig16_{slug}_rome_paris_prompts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
