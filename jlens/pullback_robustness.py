"""Dose sweep and multi-random null for the held-out pullback experiment.

The original ``pullback_steer.py`` has the right no-J and held-out-word
controls, but uses one intervention dose and one random direction per
concept/layer cell. This script keeps its construction and evaluation protocol
while adding:

* a dose sweep for ``J.T @ w`` and direct residual steering with ``w``;
* 30 random directions per layer at the preregistered reference dose;
* per-prompt effects, empirical p-values, and random-null z-scores;
* batching of the evaluation prompts, so the 30-direction null is cheap.

The intervention matches the original script: the same direction is added at
every token position at the output of the selected transformer block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.pullback_steer import CONCEPTS, PROMPTS


class AddEverywhere:
    def __init__(self, model, layer: int, direction: torch.Tensor, magnitude: float):
        self.delta = F.normalize(direction.float(), dim=0) * magnitude
        self.handle = model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = tensor + self.delta.to(tensor.device, tensor.dtype)
        return edited if torch.is_tensor(output) else (edited,) + tuple(output[1:])

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def single_token_ids(tokenizer, words):
    ids = []
    for word in words:
        for candidate in (" " + word, word, " " + word.capitalize()):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
                break
    return sorted(set(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 26])
    parser.add_argument("--doses", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--null-dose", type=float, default=0.15)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("out/rare/pullback_robustness.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wn = F.normalize(W - W.mean(0, keepdim=True), dim=1)
    blob = torch.load(args.jall, map_location="cpu", weights_only=False)

    @torch.no_grad()
    def log_probs():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return torch.log_softmax(output.logits[:, -1].float(), dim=-1).cpu()

    base = log_probs()
    mean_base = base.mean(0)
    order = torch.argsort(mean_base, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))

    concept_data = {}
    for name, words in CONCEPTS.items():
        ids = single_token_ids(tokenizer, words.split())
        if len(ids) < 12:
            continue
        train_ids = ids[0::2]
        test_ids = ids[1::2]
        w = F.normalize(Wn[torch.tensor(train_ids)].mean(0), dim=0)
        similarities = Wn @ w
        used = set(ids)
        controls = []
        for target_rank in rank[torch.tensor(test_ids)].tolist():
            for offset in range(800):
                found = None
                for candidate in (
                    int(order[min(target_rank + offset, len(order) - 1)]),
                    int(order[max(target_rank - offset, 0)]),
                ):
                    if candidate not in used and float(similarities[candidate]) < 0.10:
                        found = candidate
                        break
                if found is not None:
                    controls.append(found)
                    used.add(found)
                    break
        concept_data[name] = {
            "w": w,
            "train_ids": train_ids,
            "test_ids": torch.tensor(test_ids),
            "control_ids": torch.tensor(controls[: len(test_ids)]),
        }

    def lifts(edited_log_probs, concept):
        delta = edited_log_probs - base
        per_prompt = (
            delta[:, concept["test_ids"]].mean(1)
            - delta[:, concept["control_ids"]].mean(1)
        )
        return per_prompt

    # Estimate the residual scale once per layer using the same evaluation bank.
    scales = {}
    for layer in args.layers:
        captured = []

        def capture(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            norms = tensor.detach().float().norm(dim=-1)
            seq_pos = attention_mask.long().cumsum(dim=1) - 1
            valid = attention_mask.bool() & (seq_pos >= 1)
            per_prompt = [norms[b][valid[b]].median() for b in range(norms.shape[0])]
            captured.append(torch.stack(per_prompt).mean().cpu())

        handle = model.model.layers[layer].register_forward_hook(capture)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        handle.remove()
        scales[layer] = float(torch.stack(captured).mean())

    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    random_rows = []

    for layer in args.layers:
        J = blob["J"][layer].float()
        # Random edits do not depend on a concept, so each edited forward pass is
        # shared across all concept-specific null measurements.
        random_per_concept = {name: [] for name in concept_data}
        for draw in range(args.n_random):
            direction = F.normalize(torch.randn(J.shape[1], generator=generator), dim=0)
            with AddEverywhere(model, layer, direction, args.null_dose * scales[layer]):
                edited = log_probs()
            for name, concept in concept_data.items():
                per_prompt = lifts(edited, concept)
                random_per_concept[name].append(float(per_prompt.mean()))
                random_rows.append({
                    "layer": layer,
                    "concept": name,
                    "dose": args.null_dose,
                    "draw": draw,
                    "lift": float(per_prompt.mean()),
                    "prompt_lifts": [float(x) for x in per_prompt],
                })

        for name, concept in concept_data.items():
            directions = {
                "pullback": J.T @ concept["w"],
                "direct_w": concept["w"],
            }
            null = torch.tensor(random_per_concept[name])
            null_mean = float(null.mean())
            null_sd = float(null.std(unbiased=True)) if len(null) > 1 else 0.0

            for dose in args.doses:
                for method, direction in directions.items():
                    with AddEverywhere(model, layer, direction, dose * scales[layer]):
                        edited = log_probs()
                    per_prompt = lifts(edited, concept)
                    value = float(per_prompt.mean())
                    row = {
                        "layer": layer,
                        "concept": name,
                        "dose": dose,
                        "method": method,
                        "lift": value,
                        "prompt_lifts": [float(x) for x in per_prompt],
                    }
                    if abs(dose - args.null_dose) < 1e-9:
                        row.update({
                            "random_mean": null_mean,
                            "random_sd": null_sd,
                            "z_vs_random": (value - null_mean) / max(null_sd, 1e-9),
                            "empirical_p_ge": (
                                1 + int((null >= value).sum())
                            ) / (len(null) + 1),
                        })
                    rows.append(row)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": vars(args) | {"jall": str(args.jall), "out": str(args.out)},
            "prompt_count": len(PROMPTS),
            "concepts": {
                name: {
                    "n_train_tokens": len(data["train_ids"]),
                    "n_test_tokens": len(data["test_ids"]),
                    "n_control_tokens": len(data["control_ids"]),
                }
                for name, data in concept_data.items()
            },
            "scales": scales,
            "rows": rows,
            "random_rows": random_rows,
        }, indent=1))
        print(f"layer {layer} complete", flush=True)

    at_null = [r for r in rows if abs(r["dose"] - args.null_dose) < 1e-9]
    print("\nReference-dose summary")
    for method in ("pullback", "direct_w"):
        selected = [r for r in at_null if r["method"] == method]
        print(
            f"{method:10s} mean lift {sum(r['lift'] for r in selected) / len(selected):+.3f}  "
            f"median z {torch.tensor([r['z_vs_random'] for r in selected]).quantile(.5).item():+.2f}  "
            f"empirical p<=.05 {sum(r['empirical_p_ge'] <= .05 for r in selected)}/{len(selected)}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
