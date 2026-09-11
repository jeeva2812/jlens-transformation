"""Does `J.T w` beat `w` for contrasts other than Rome/Paris, at layers other than one?

The headline experiment (`contrastive_logit_steering.py`) uses one contrast at one
layer per model.  Two objections follow immediately: the result could be a
Rome/Paris artefact, and it could be a lucky layer.  This sweeps both.

For each (contrast, layer) cell we compare, at matched intervention L2 norm:

    direct_w   w = normalize(W_U[a] - W_U[b])     the logit-lens direction
    pullback   J_l.T @ w                          the J-Lens vector
    random     n matched-norm draws               the null

Metric is the same as the headline: mean change in (logit a - logit b) over a
fixed set of generic prompts.  The prompts are deliberately contrast-agnostic --
they lead nowhere in particular -- so the same set can score a geographic, an
occupational and a colour contrast without favouring any of them.  Absolute sizes
are therefore not comparable with the headline table, which uses city prompts;
the *ordering* within a cell is the claim.

The per-layer intervention scale is measured here rather than read from a saved
file, so any layer present in the lens can be swept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere

# Generic continuations that do not lead toward any of the contrasts below.
PROMPTS = [
    "She thought about", "He mentioned", "The topic they discussed was",
    "The article was mainly about", "It reminded her of", "He wrote down",
    "The subject of the talk was", "The example given was",
    "She pointed to", "The next thing mentioned was",
    "They kept coming back to", "The word that came to mind was",
]

# (name, positive token, negative token, is this contrast geographic?)
CONTRASTS = [
    ("rome_paris", "Rome", "Paris", "geography"),
    ("tokyo_paris", "Tokyo", "Paris", "geography"),
    ("berlin_madrid", "Berlin", "Madrid", "geography"),
    ("doctor_lawyer", "doctor", "lawyer", "occupation"),
    ("summer_winter", "summer", "winter", "season"),
    ("red_blue", "red", "blue", "colour"),
]


def one_token(tokenizer, word):
    for candidate in (" " + word, word):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


class CaptureNorm:
    """Typical L2 norm of the residual leaving each requested block.

    Per-prompt MEDIAN over positions, then mean across prompts, skipping the
    first position.  This matches `pullback_robustness.py` exactly, and the
    choice is load-bearing: a plain mean is dominated by the attention-sink
    position, whose norm is an order of magnitude larger than everything else.
    Using the mean on SmolLM2 put the layer-12 scale at 4651 instead of 141 and
    saturated every intervention.
    """

    def __init__(self, blocks, layers, mask):
        self.norms, self.handles, self.mask = {}, [], mask

        def make(layer):
            def hook(module, inputs, output):
                tensor = output if torch.is_tensor(output) else output[0]
                norms = tensor.detach().float().norm(dim=-1)
                seq_pos = self.mask.long().cumsum(dim=1) - 1
                valid = self.mask.bool() & (seq_pos >= 1)
                per_prompt = [norms[b][valid[b]].median() for b in range(norms.shape[0])]
                self.norms[layer] = float(torch.stack(per_prompt).mean())
            return hook

        for layer in layers:
            self.handles.append(blocks[layer].register_forward_hook(make(layer)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--dose", type=float, default=.15)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    batch = tok(PROMPTS, return_tensors="pt", padding=True)
    ids, mask = batch["input_ids"].to(a.device), batch["attention_mask"].to(a.device)
    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, local_files_only=True
    ).to(a.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    embedding = model.get_output_embeddings().weight

    lens = torch.load(a.lens, map_location="cpu", weights_only=False)["J"]
    layers = [l for l in a.layers if l in lens]
    skipped_layers = sorted(set(a.layers) - set(layers))

    @torch.inference_mode()
    def logits():
        return model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits[:, -1].float().cpu()

    # One clean pass gives both the baseline logits and every layer's scale.
    with CaptureNorm(blocks, layers, mask) as capture:
        clean = logits()
    scales = dict(capture.norms)
    print("measured residual scales:", {k: round(v, 2) for k, v in scales.items()})

    contrasts, skipped_contrasts = [], []
    for name, positive, negative, kind in CONTRASTS:
        p, n = one_token(tok, positive), one_token(tok, negative)
        if p is None or n is None:
            skipped_contrasts.append(name)
            continue
        contrasts.append((name, positive, negative, kind, p, n))
    print("contrasts:", [c[0] for c in contrasts],
          "| skipped (not single-token):", skipped_contrasts)

    generator = torch.Generator().manual_seed(20260911)
    d_model = embedding.shape[1]
    random_draws = [torch.randn(d_model, generator=generator) for _ in range(a.n_random)]

    def effect(z, positive_id, negative_id):
        delta = z - clean
        return float((delta[:, positive_id] - delta[:, negative_id]).mean())

    def run(layer, direction, positive_id, negative_id):
        with AddEverywhere(blocks[layer], direction, a.dose * scales[layer],
                           device=a.device, dtype=dtype):
            return effect(logits(), positive_id, negative_id)

    # The null does not depend on the contrast's direction, only on the logit
    # pair it is scored against, so run each random draw once per layer and
    # score it against every contrast.
    null_logits = {}
    for layer in layers:
        null_logits[layer] = []
        for draw in random_draws:
            with AddEverywhere(blocks[layer], draw, a.dose * scales[layer],
                               device=a.device, dtype=dtype):
                null_logits[layer].append(logits())
        print(f"  null done for layer {layer}")

    cells = []
    for name, positive, negative, kind, positive_id, negative_id in contrasts:
        w = F.normalize(
            (embedding[positive_id] - embedding[negative_id]).detach().float().cpu(), dim=0
        )
        for layer in layers:
            J = lens[layer].float()
            direct = run(layer, w, positive_id, negative_id)
            pull = run(layer, J.T @ w, positive_id, negative_id)
            reverse = run(layer, -(J.T @ w), positive_id, negative_id)
            draws = [effect(z, positive_id, negative_id) for z in null_logits[layer]]
            mean = sum(draws) / len(draws)
            sd = (sum((x - mean) ** 2 for x in draws) / len(draws)) ** .5
            cells.append({
                "contrast": name, "kind": kind,
                "positive": positive, "negative": negative, "layer": layer,
                "direct_w": direct, "pullback": pull, "reverse_pullback": reverse,
                "null_mean": mean, "null_sd": sd,
                "null_min": min(draws), "null_max": max(draws), "null_draws": draws,
                "z_pullback": (pull - mean) / sd if sd > 0 else float("nan"),
                "z_direct": (direct - mean) / sd if sd > 0 else float("nan"),
                "ratio_pullback_over_direct": pull / direct if direct != 0 else float("nan"),
                "pullback_beats_direct": pull > direct,
                "pullback_beats_all_draws": pull > max(draws),
            })
            print(f"  {name:14s} L{layer:<3d} direct {direct:+6.2f}  pull {pull:+6.2f}"
                  f"  ratio {cells[-1]['ratio_pullback_over_direct']:5.2f}"
                  f"  z {cells[-1]['z_pullback']:+7.1f}")

    wins = sum(c["pullback_beats_direct"] for c in cells)
    above = sum(c["pullback_beats_all_draws"] for c in cells)
    ratios = sorted(c["ratio_pullback_over_direct"] for c in cells)
    summary = {
        "n_cells": len(cells),
        "pullback_beats_direct": wins,
        "pullback_beats_all_random_draws": above,
        "ratio_median": ratios[len(ratios) // 2] if ratios else float("nan"),
        "ratio_min": ratios[0] if ratios else float("nan"),
        "ratio_max": ratios[-1] if ratios else float("nan"),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"lens": str(a.lens), "out": str(a.out)},
        "prompts": PROMPTS,
        "measured_scales": scales,
        "skipped_layers": skipped_layers,
        "skipped_contrasts": skipped_contrasts,
        "summary": summary,
        "cells": cells,
    }, indent=2, default=str))

    print(f"\n{a.model}")
    print(f"  pullback beats direct_w in {wins}/{len(cells)} cells")
    print(f"  pullback beats all {a.n_random} random draws in {above}/{len(cells)} cells")
    print(f"  ratio pullback/direct: median {summary['ratio_median']:.2f}"
          f"  range [{summary['ratio_min']:.2f}, {summary['ratio_max']:.2f}]")


if __name__ == "__main__":
    main()
