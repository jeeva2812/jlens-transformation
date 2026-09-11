"""Build the figures for the operator-versus-feature application story.

This script performs no model inference. It only aggregates saved experiment
outputs, which makes the plots cheap to regenerate while writing the application.

Run:

    PYTHONPATH=. .venv/bin/python -m jlens.application_core_figs

Outputs are written to ``out/figs_core``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2b6cb0"
GREEN = "#2f855a"
ORANGE = "#dd6b20"
RED = "#c53030"
GREY = "#718096"
PALE = "#cbd5e0"
PURPLE = "#805ad5"


def load(path: Path):
    return json.loads(path.read_text())


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def mean(xs):
    return sum(xs) / len(xs)


def method_name(row):
    label = row["label"]
    if label.startswith("best SVD"):
        return "Best single\nSVD component"
    if label.startswith("transpose"):
        return "Pullback\n$J^T w$"
    if label.startswith("w itself"):
        return "Naive residual\n$w$"
    if label.startswith("pinv"):
        return "Inverse\n$J^+ w$"
    if label.startswith("ridge lam=1"):
        return "Ridge\n$\\lambda=1$"
    if label == "random":
        return "Random"
    return None


def pullback_figure(rows, out: Path):
    groups = defaultdict(list)
    for row in rows:
        name = method_name(row)
        if name is not None:
            groups[name].append(row)

    order = [
        "Pullback\n$J^T w$",
        "Best single\nSVD component",
        "Ridge\n$\\lambda=1$",
        "Naive residual\n$w$",
        "Inverse\n$J^+ w$",
        "Random",
    ]
    colors = [GREEN, ORANGE, ORANGE, BLUE, RED, GREY]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    vals = [mean([r["lift"] for r in groups[name]]) for name in order]
    bars = ax.bar(np.arange(len(order)), vals, color=colors, width=0.72)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("Held-out concept lift over matched controls")
    ax.set_title("The averaged Jacobian works best as a pullback operator")
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.04 if value >= 0 else -0.04),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8.5,
        )

    ax = axes[1]
    layers = sorted({r["layer"] for r in rows})
    pull = [mean([r["lift"] for r in groups["Pullback\n$J^T w$"] if r["layer"] == layer])
            for layer in layers]
    naive = [mean([r["lift"] for r in groups["Naive residual\n$w$"] if r["layer"] == layer])
             for layer in layers]
    ax.plot(layers, pull, "o-", color=GREEN, lw=2.2, label="$J^T w$")
    ax.plot(layers, naive, "o--", color=BLUE, lw=2.0, label="$w$ without $J$")
    ax.set_xlabel("Source layer in SmolLM2-135M")
    ax.set_ylabel("Held-out concept lift")
    ax.set_title("The advantage is largest before $J$ approaches identity")
    ax.legend(frameon=False)
    for x, a, b in zip(layers, pull, naive):
        ax.text(x, a + 0.08, f"{a / b:.1f}x", ha="center", fontsize=8, color=GREEN)

    fig.suptitle("Seven concepts x four layers; concept words split into construction and test halves",
                 fontsize=10.5, y=1.02)
    save(fig, out, "fig1_pullback_operator.png")


def truncation_figure(rows, out: Path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["k"])].append(row)
    ks = sorted(grouped)
    lift = [mean([r["lift"] for r in grouped[k]]) for k in ks]
    mass = [mean([r["mass"] for r in grouped[k]]) for k in ks]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(ks))
    ax.plot(x, lift, "o-", color=GREEN, lw=2.4, label="held-out concept lift")
    ax.axhline(lift[-1], color=GREEN, ls=":", alpha=0.65, label="full pullback")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of singular components retained")
    ax.set_ylabel("Held-out concept lift", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    ax.set_title("One component is insufficient; 32-64 outperform the full spectrum")

    ax2 = ax.twinx()
    ax2.plot(x, np.array(mass) * 100, "s--", color=BLUE, lw=1.8,
             label="share of pullback energy")
    ax2.set_ylabel("Pullback energy retained (%)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(0, 106)

    handles = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, frameon=False, loc="center right")
    for k in (1, 32, 64, 576):
        i = ks.index(k)
        ax.annotate(f"{lift[i]:.2f}", (x[i], lift[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    save(fig, out, "fig2_truncated_pullback.png")


def conditioning_figure(concepts, oracle, out: Path):
    wanted = ["is_origin", "queen_can_move", "rook_can_move", "has_capture"]
    labels = {
        "is_origin": "Output slot\n(origin vs destination)",
        "queen_can_move": "Queen can move",
        "rook_can_move": "Rook can move",
        "has_capture": "Capture available",
    }
    # concepts_big.json contains one row per (layer, component, label). Keep the
    # strongest fully corrected correlation for each label. Unlike the legacy
    # concept_r2.json summary, this file has a checked-in producer script.
    by_label = {}
    for row in concepts["top"]:
        old = by_label.get(row["label"])
        if old is None or row["r"] > old["r"]:
            by_label[row["label"]] = row

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    ax = axes[0]
    values = [by_label[name]["r"] for name in wanted]
    bars = ax.bar(np.arange(len(wanted)), values, color=[RED, BLUE, BLUE, BLUE])
    ax.axhline(concepts["thr95"], color=GREY, ls=":", label="family-wise 95% null")
    ax.set_xticks(np.arange(len(wanted)))
    ax.set_xticklabels([labels[name] for name in wanted], fontsize=8.5)
    ax.set_ylabel("Best |correlation| with one of 24 coefficients")
    ax.set_title("The dominant signal is which token slot is being generated")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                ha="center", fontsize=8.5)

    # Layer 5 destination positions give the clearest test of the averaging
    # hypothesis. The comparison includes all baselines produced by q1_q2.py.
    rows = oracle["Q2"]["dest"]["5"]
    ks = [r["k"] for r in rows]
    ax = axes[1]
    series = [
        ("Conditional J", [r["J_matched"] for r in rows], GREEN, "o-"),
        ("Pooled J", [r["J_pooled"] for r in rows], RED, "s--"),
        ("Activation PCA", [r["pca"] for r in rows], BLUE, "d-."),
        ("Random subspace", [r["rand_mean"] for r in rows], GREY, "^:"),
        ("Split-half ceiling", [r["ceil_split_mean"] for r in rows], PALE, "x-"),
    ]
    for label, values, color, style in series:
        ax.plot(ks, values, style, color=color, lw=2, ms=6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Subspace rank")
    ax.set_ylabel("Mean squared principal-angle overlap")
    ax.set_title("Conditioning partly recovers the destination-legality subspace")
    ax.legend(frameon=False, fontsize=8.5)
    ax.annotate("2.4x pooled", xy=(4, rows[0]["J_matched"]), xytext=(7, 0.29),
                arrowprops={"arrowstyle": "->", "color": GREEN}, fontsize=8.5,
                color=GREEN)

    fig.suptitle("Chess provides objective labels and separates two known computational regimes",
                 fontsize=10.5, y=1.02)
    save(fig, out, "fig3_averaging_and_conditioning.png")


def robustness_figure(data, out: Path):
    rows = data["rows"]
    doses = sorted({r["dose"] for r in rows})
    null_dose = data["config"]["null_dose"]
    layers = data["config"]["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    ax = axes[0]
    for method, label, color, style in [
        ("pullback", "Pullback $J^T w$", GREEN, "o-"),
        ("direct_w", "Naive residual $w$", BLUE, "s--"),
    ]:
        means, sem = [], []
        for dose in doses:
            values = np.array([
                r["lift"] for r in rows
                if r["method"] == method and r["dose"] == dose
            ])
            means.append(values.mean())
            sem.append(values.std(ddof=1) / np.sqrt(len(values)))
        means = np.array(means)
        sem = np.array(sem)
        ax.plot(doses, means, style, color=color, lw=2.2, ms=6, label=label)
        ax.fill_between(doses, means - sem, means + sem, color=color, alpha=0.13)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(null_dose, color=GREY, lw=1, ls=":", label="null-test dose")
    ax.set_xlabel("Intervention size (fraction of median residual norm)")
    ax.set_ylabel("Held-out concept lift")
    ax.set_title("The pullback advantage persists across intervention doses")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    offsets = {"pullback": -0.12, "direct_w": 0.12}
    colors = {"pullback": GREEN, "direct_w": BLUE}
    labels = {"pullback": "$J^T w$", "direct_w": "$w$"}
    rng = np.random.default_rng(20260910)
    for i, layer in enumerate(layers):
        for method in ("pullback", "direct_w"):
            selected = [
                r for r in rows
                if r["method"] == method
                and r["layer"] == layer
                and abs(r["dose"] - null_dose) < 1e-9
            ]
            z = np.array([r["z_vs_random"] for r in selected])
            x = i + offsets[method] + rng.normal(0, 0.025, len(z))
            ax.scatter(x, z, s=26, color=colors[method], alpha=0.72,
                       edgecolor="white", linewidth=0.35,
                       label=labels[method] if i == 0 else None)
            ax.plot(
                [i + offsets[method] - 0.09, i + offsets[method] + 0.09],
                [np.median(z), np.median(z)], color=colors[method], lw=2.6,
            )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Effect z-score against 30 random directions")
    ax.set_title("All 28 pullback cells exceed every sampled random direction")
    ax.legend(frameon=False)

    fig.suptitle(
        "Seven concepts x four layers x twelve neutral prompts; bands are descriptive ±1 SEM across cells",
        fontsize=10.5, y=1.02,
    )
    save(fig, out, "fig4_pullback_robustness.png")


def corpus_figure(data, out: Path):
    geometry = data["geometry_rows"]
    effects = data["effect_rows"]
    layers = data["config"]["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    ax = axes[0]
    comparisons = [
        ("Matched prose A vs B", [("prose_a", "prose_b")], GREEN, "o-"),
        ("Prose vs code", [("prose_a", "code"), ("prose_b", "code")], ORANGE, "s--"),
        ("Pooled prose vs saved $J$", [("pooled_prose", "saved_global")], BLUE, "d-."),
    ]
    for label, pairs, color, style in comparisons:
        values = []
        for layer in layers:
            selected = [
                r["cosine"] for r in geometry
                if r["layer"] == layer and (r["a"], r["b"]) in pairs
            ]
            values.append(mean(selected))
        ax.plot(layers, values, style, color=color, lw=2.1, ms=6, label=label)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Mean cosine between concept pullbacks")
    ax.set_title("Matched prose estimates agree; code changes the operator")
    ax.legend(frameon=False, fontsize=8.2)

    ax = axes[1]
    series = [
        ("Saved global $J$", "saved_global", GREY, "o:"),
        ("New pooled prose", "pooled_prose", GREEN, "o-"),
        ("Code-estimated", "code", ORANGE, "s--"),
        ("No $J$: direct $w$", "direct_w", BLUE, "d-."),
    ]
    for label, method, color, style in series:
        means, errors = [], []
        for layer in layers:
            selected = np.array([
                r["lift"] for r in effects
                if r["layer"] == layer and r["method"] == method
            ])
            means.append(selected.mean())
            errors.append(selected.std(ddof=1) / np.sqrt(len(selected)))
        ax.errorbar(layers, means, yerr=errors, fmt=style, color=color,
                    lw=2, ms=5.5, capsize=2.5, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Held-out concept lift on neutral prose prompts")
    ax.set_title("Distribution shift weakens transfer, but retains useful signal")
    ax.legend(frameon=False, fontsize=8.2)

    fig.suptitle(
        "J is distribution-conditioned (18 prompts/corpus; right-panel bars are descriptive ±1 SEM across 7 concepts)",
        fontsize=10.5, y=1.02,
    )
    save(fig, out, "fig5_corpus_conditioning.png")


def replication_figure(smol, qwen, olmo, out: Path):
    datasets = [
        ("SmolLM2-135M", smol, GREEN),
        ("Qwen3.5-4B", qwen, PURPLE),
        ("OLMo-3-7B", olmo, ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.35))

    ax = axes[0]
    for model_name, data, color in datasets:
        rows = data["rows"]
        doses = sorted({r["dose"] for r in rows})
        for method, method_label, style in [
            ("pullback", "$J^T w$", "o-"),
            ("direct_w", "direct $w$", "s--"),
        ]:
            values = [
                mean([r["lift"] for r in rows
                      if r["dose"] == dose and r["method"] == method])
                for dose in doses
            ]
            short_name = {"SmolLM2-135M": "Smol-135M", "Qwen3.5-4B": "Qwen-4B",
                          "OLMo-3-7B": "OLMo-7B"}[model_name]
            ax.plot(doses, values, style, color=color, lw=2.2, ms=5.5,
                    label=f"{short_name}: {method_label}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Intervention size (fraction of median residual norm)")
    ax.set_ylabel("Held-out concept lift")
    ax.set_title("The pullback advantage replicates through 7B scale")
    ax.legend(frameon=False, fontsize=8.0, ncol=2, columnspacing=1.1)

    ax = axes[1]
    for model_name, data, color in datasets:
        rows = data["rows"]
        dose = data["config"]["null_dose"]
        target = (
            data.get("lens_provenance", {}).get("target_layer")
            or data["config"].get("target_layer")
            or 28
        )
        layers = data["config"]["layers"]
        ratios = []
        for layer in layers:
            pull = mean([r["lift"] for r in rows if r["layer"] == layer
                         and r["method"] == "pullback" and r["dose"] == dose])
            direct = mean([r["lift"] for r in rows if r["layer"] == layer
                           and r["method"] == "direct_w" and r["dose"] == dose])
            ratios.append(pull / direct)
        relative_depth = [layer / target for layer in layers]
        ax.plot(relative_depth, ratios, "o-", color=color, lw=2.2, ms=6,
                label=model_name)
        ax.text(relative_depth[0], ratios[0] + 0.18,
                f"{ratios[0]:.1f}x", color=color, fontsize=8, ha="center")
    ax.axhline(1, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Source depth / target depth")
    ax.set_ylabel("Mean lift ratio: pullback / direct $w$")
    ax.set_title("The advantage shrinks as $J$ approaches identity")
    ax.text(0.68, 2.55, "near target: 1.25–1.57x", fontsize=8.5, color=GREY)
    ax.legend(frameon=False)

    fig.suptitle(
        "At dose 0.15, all 84 pullback cells across three models exceed all 30 sampled random effects",
        fontsize=10.5, y=1.02,
    )
    save(fig, out, "fig6_large_model_replication.png")


def truncation_multimodel_figure(rank_smol, rank_qwen, rank_olmo, out: Path):
    """Same truncation sweep on three models. Handles both row formats: the
    SmolLM2 artifact is a bare list with integer k (full spectrum k=576);
    the Qwen/OLMo artifacts are dicts with a 'rows' list and k='full'."""
    def rows(obj):
        return obj if isinstance(obj, list) else obj["rows"]

    def curve(obj, full_key):
        grouped = defaultdict(list)
        for row in rows(obj):
            k = row["k"]
            k = "full" if str(k) == str(full_key) else int(k)
            grouped[k].append(row["lift"])
        return grouped

    grid = [1, 4, 8, 16, 32, 64, 128, 256, "full"]
    series = [
        ("SmolLM2-135M", curve(rank_smol, 576), GREEN),
        ("Qwen3.5-4B", curve(rank_qwen, "full"), BLUE),
        ("OLMo-3-7B", curve(rank_olmo, "full"), ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(grid))
    for name, grouped, color in series:
        lift = [mean(grouped[k]) for k in grid]
        ax.plot(x, lift, "o-", color=color, lw=2.2, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in grid])
    ax.set_xlabel("Singular components retained (matched intervention dose)")
    ax.set_ylabel("Mean held-out concept lift")
    ax.set_title("One component is weak everywhere; whether the tail helps depends on the model")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "fig2b_truncated_pullback_multimodel.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pullback", type=Path, default=Path("out/rare/pullback2.json"))
    parser.add_argument("--rank", type=Path, default=Path("out/rare/pullback_rank.json"))
    parser.add_argument("--rank-qwen", type=Path,
                        default=Path("out/rare/qwen35_4b_pullback_rank.json"))
    parser.add_argument("--rank-olmo", type=Path,
                        default=Path("out/rare/olmo3_7b_pullback_rank.json"))
    parser.add_argument("--chess-concepts", type=Path, default=Path("out/chess/concepts_big.json"))
    parser.add_argument("--chess-oracle", type=Path, default=Path("out/chess/q1_q2.json"))
    parser.add_argument("--robustness", type=Path, default=Path("out/rare/pullback_robustness.json"))
    parser.add_argument("--corpus", type=Path, default=Path("out/rare/pullback_corpus.json"))
    parser.add_argument("--qwen-replication", type=Path,
                        default=Path("out/rare/qwen35_4b_pullback_robustness.json"))
    parser.add_argument("--olmo-replication", type=Path,
                        default=Path("out/rare/olmo3_7b_pullback_robustness.json"))
    parser.add_argument("--out", type=Path, default=Path("out/figs_core"))
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
    })

    pullback_figure(load(args.pullback), args.out)
    truncation_figure(load(args.rank), args.out)
    truncation_multimodel_figure(
        load(args.rank), load(args.rank_qwen), load(args.rank_olmo), args.out,
    )
    conditioning_figure(load(args.chess_concepts), load(args.chess_oracle), args.out)
    robustness_figure(load(args.robustness), args.out)
    corpus_figure(load(args.corpus), args.out)
    replication_figure(
        load(args.robustness), load(args.qwen_replication),
        load(args.olmo_replication), args.out,
    )


if __name__ == "__main__":
    main()
