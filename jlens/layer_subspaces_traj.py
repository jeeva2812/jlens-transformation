"""Cross-layer subspace overlap across all 11 Olmo-3-7B checkpoints.

`layer_subspaces.py` compares init against the final model. This runs the same
measurement at every released checkpoint, so the question becomes *when* layers
start sharing, and whether the mid-training boundary that F7 flags shows up here.

Exact SVD throughout, deliberately. torch.svd_lowrank is 37x faster and agrees
with exact at 0.9997 on the trained model -- but at initialisation the spectrum is
nearly flat (sigma_64 / sigma_1 = 0.94 at layer 20) and agreement falls to 0.77.
A randomised method cannot resolve a subspace that is barely distinguished, and
the init checkpoint is exactly the one we need to trust.

That flatness is also worth stating as a limitation of the init number itself: the
top-64 subspace at initialisation is real but weakly determined, because
directions 1 and 64 are amplified almost equally.

    PYTHONPATH=. .venv/bin/python -u -m jlens.layer_subspaces_traj
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.layer_subspaces import measured_chance, overlap

CKPT = Path("out/ckpt")
ORDER = [
    ("init", "J_stage1-step0.pt"), ("2k", "J_stage1-step2000.pt"),
    ("8k", "J_stage1-step8000.pt"), ("32k", "J_stage1-step32000.pt"),
    ("128k", "J_stage1-step128000.pt"), ("512k", "J_stage1-step512000.pt"),
    ("end pre", "J_stage1-step1413814.pt"), ("mid 8k", "J_stage2-step8000.pt"),
    ("end mid", "J_stage2-step47684.pt"), ("ctx 5k", "J_stage3-step5000.pt"),
    ("final", "J_main.pt"),
]
K = 64


def main():
    rows, start = {}, time.time()
    chance = None
    for label, fname in ORDER:
        path = CKPT / fname
        if not path.exists():
            print(f"  skip {label}: {path} missing")
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)["J"]
        layers = sorted(blob.keys())
        d = blob[layers[0]].shape[0]
        if chance is None:
            chance = measured_chance(d, K, 30)
            print(f"measured chance {chance['mean']:.4f} +/- {chance['sd']:.4f}\n")
        bases, ratios = {}, []
        for L in layers:
            # One exact SVD per layer, reused for both the subspace and the
            # spectral-flatness diagnostic. Calling svdvals separately doubled
            # the runtime for no reason.
            _, S, Vh = torch.linalg.svd(blob[L].float(), full_matrices=False)
            bases[L] = Vh[:K].T.contiguous()
            ratios.append(float(S[K - 1] / S[0]))
        del blob
        matrix = [[overlap(bases[i], bases[j]) for j in layers] for i in layers]
        adjacent = [matrix[i][i + 1] for i in range(len(layers) - 1)]
        far = [matrix[i][j] for i in range(len(layers))
               for j in range(len(layers)) if abs(i - j) >= 5]
        rows[label] = {
            "matrix": matrix, "layers": layers,
            "adjacent_mean": sum(adjacent) / len(adjacent),
            "far_mean": sum(far) / len(far),
            "sigma64_over_sigma1_mean": sum(ratios) / len(ratios),
        }
        print(f"{label:9s} 2-apart {rows[label]['adjacent_mean']:.4f}   "
              f">=10-apart {rows[label]['far_mean']:.4f}   "
              f"(far/chance {rows[label]['far_mean'] / chance['mean']:5.2f}x)   "
              f"sigma64/sigma1 {rows[label]['sigma64_over_sigma1_mean']:.3f}   "
              f"[{time.time() - start:.0f}s]", flush=True)

    out = Path("out/rare/layer_subspaces_traj.json")
    out.write_text(json.dumps({"k": K, "chance": chance, "checkpoints": rows}, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
