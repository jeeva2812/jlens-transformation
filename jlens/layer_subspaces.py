"""Do different layers read from the same subspace? Init vs trained.

For J = U S V^T the columns of V are directions in that layer's own space,
ordered by how strongly J amplifies them.  If two layers' top-k V-subspaces
overlap, the layers are transporting the same content and depth is partly
redundant.  If they do not, each layer reads something its neighbours do not.

Overlap is the mean squared canonical cosine between the two subspaces,

    overlap(A, B) = (1/k) * || V_A^T V_B ||_F^2

which is 1.0 for identical subspaces and k/d in expectation for two random
k-subspaces of R^d.

The chance line is MEASURED, not derived.  An earlier arm of this project got a
chance line wrong by 5.4x by deriving it for the wrong matrix shape and
manufactured a whole component-level finding out of it, so the baseline here is
30 random orthonormal subspaces put through the identical code path.

Runs on saved Jacobians only -- no model is loaded.

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.layer_subspaces
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RED = "#1b1b1b", "#7a7a7a", "#c0392b"


def right_subspace(J: torch.Tensor, k: int) -> torch.Tensor:
    """Orthonormal basis of the top-k right singular subspace of J."""
    _, _, Vh = torch.linalg.svd(J.float(), full_matrices=False)
    return Vh[:k].T.contiguous()                       # (d, k), orthonormal


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared canonical cosine between two orthonormal subspaces."""
    return float((a.T @ b).pow(2).sum() / a.shape[1])


def measured_chance(d: int, k: int, draws: int, seed: int = 20260911) -> dict:
    g = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(draws):
        qa, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
        qb, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
        values.append(overlap(qa, qb))
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** .5
    return {"mean": mean, "sd": sd, "max": max(values),
            "derived_k_over_d": k / d, "draws": draws}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, default=Path("out/ckpt/J_stage1-step0.pt"))
    ap.add_argument("--trained", type=Path, default=Path("out/ckpt/J_main.pt"))
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("out/rare/layer_subspaces.json"))
    a = ap.parse_args()

    result = {"k": a.k, "models": {}}
    bases, layers_used = {}, None
    for name, path in (("init", a.init), ("trained", a.trained)):
        blob = torch.load(path, map_location="cpu", weights_only=False)["J"]
        layers = sorted(blob.keys())
        layers_used = layers
        d = blob[layers[0]].shape[0]
        print(f"{name:8s}: {len(layers)} layers, d = {d}, k = {a.k}")
        bases[name] = {L: right_subspace(blob[L], a.k) for L in layers}
        del blob

    chance = measured_chance(d, a.k, a.draws)
    print(f"\nmeasured chance over {a.draws} random subspace pairs: "
          f"{chance['mean']:.4f} ± {chance['sd']:.4f}  (max {chance['max']:.4f})")
    print(f"derived k/d for comparison: {chance['derived_k_over_d']:.4f}\n")
    result["chance"] = chance
    result["layers"] = layers_used

    for name in ("init", "trained"):
        B = bases[name]
        matrix = [[overlap(B[i], B[j]) for j in layers_used] for i in layers_used]
        result["models"][name] = {"matrix": matrix}
        adjacent = [matrix[i][i + 1] for i in range(len(layers_used) - 1)]
        far = [matrix[i][j] for i in range(len(layers_used))
               for j in range(len(layers_used)) if abs(i - j) >= 5]
        result["models"][name]["adjacent_mean"] = sum(adjacent) / len(adjacent)
        result["models"][name]["far_mean"] = sum(far) / len(far)
        print(f"{name:8s}  adjacent layers {sum(adjacent)/len(adjacent):.4f}   "
              f"|Δlayer| >= 5: {sum(far)/len(far):.4f}   "
              f"(chance {chance['mean']:.4f})")

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    vmax = max(max(max(r) for r in result["models"][n]["matrix"])
               for n in ("init", "trained"))
    for ax, name, title in zip(axes[:2], ("init", "trained"),
                               ("A · Random initialisation",
                                "B · Trained (Olmo-3-7B final)")):
        m = result["models"][name]["matrix"]
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(layers_used)))
        ax.set_xticklabels(layers_used, fontsize=7, rotation=90)
        ax.set_yticks(range(len(layers_used)))
        ax.set_yticklabels(layers_used, fontsize=7)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        ax.set_title(f"{title}\nadjacent {result['models'][name]['adjacent_mean']:.3f}, "
                     f"far {result['models'][name]['far_mean']:.3f}",
                     fontsize=9.5, color=INK, loc="left")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04).ax.tick_params(labelsize=7)

    ax = axes[2]
    for name, colour in (("init", "#7a7a7a"), ("trained", "#2b6cb0")):
        m = result["models"][name]["matrix"]
        by_gap = {}
        for i in range(len(layers_used)):
            for j in range(len(layers_used)):
                if i == j:
                    continue
                by_gap.setdefault(abs(layers_used[i] - layers_used[j]), []).append(m[i][j])
        gaps = sorted(by_gap)
        ax.plot(gaps, [sum(by_gap[g]) / len(by_gap[g]) for g in gaps],
                "o-", color=colour, lw=1.9, ms=4.5, label=name)
    ax.axhline(chance["mean"], color=RED, ls="--", lw=1.2)
    ax.fill_between([0, max(gaps)], chance["mean"] - 2 * chance["sd"],
                    chance["mean"] + 2 * chance["sd"], color=RED, alpha=.15)
    ax.text(max(gaps) * .40, chance["mean"] + .028,
            f"measured chance {chance['mean']:.4f} ± {chance['sd']:.4f}",
            fontsize=7.5, color=RED)
    ax.set_xlabel("distance between layers")
    ax.set_ylabel(f"mean overlap of top-{a.k} subspaces")
    ax.set_title("C · Decays with distance — but training\n"
                 "makes layers share MORE, not less",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(f"Do layers share a subspace? Top-{a.k} right singular subspaces of J",
                 fontsize=12, color=INK, x=.005, y=.985, ha="left")
    fig.text(.005, .005,
             "Overlap = mean squared canonical cosine; 1.0 is identical, and the chance line is "
             "MEASURED from 30 random orthonormal subspace pairs through the same code path, not "
             "derived. No model is loaded: this reads saved Jacobians.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, .04, 1, .90])
    for path in (Path("out/figs_final/fig8_layer_subspaces.png"),
                 Path("out/report/F8_layer_subspaces.png")):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print("wrote", path)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
