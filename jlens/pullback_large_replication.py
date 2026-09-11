"""Replicate the held-out pullback result on Qwen3.5-4B.

This uses the externally published full J-Lens artifact from
``camilablank/workspace-lenses`` rather than re-estimating a 2560 x 2560
Jacobian locally. The causal evaluation otherwise mirrors
``pullback_robustness.py``:

* seven concepts, split into construction and held-out token halves;
* rank-matched control tokens;
* direct residual ``w`` versus the pullback ``J.T @ w``;
* four intervention doses and 30 random directions per layer.

The default model and lens are already cached on the development machine. Model
loading is explicitly local-only unless ``--allow-download`` is supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import _find_blocks_and_norm
from jlens.pullback_robustness import single_token_ids
from jlens.pullback_steer import CONCEPTS, PROMPTS


LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.5-4b/j-lens/lens.pt"


class AddEverywhere:
    def __init__(self, block, direction: torch.Tensor, magnitude: float, *, device, dtype):
        self.delta = (F.normalize(direction.float(), dim=0) * magnitude).to(device, dtype)
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = tensor + self.delta
        return edited if torch.is_tensor(output) else (edited,) + tuple(output[1:])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lens", type=Path)
    parser.add_argument("--target-layer", type=int,
                        help="required when the lens artifact has no provenance")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 13, 21, 28])
    parser.add_argument("--doses", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--null-dose", type=float, default=0.15)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("out/rare/qwen35_4b_pullback_robustness.json"))
    args = parser.parse_args()

    if args.lens is None:
        args.lens = Path(hf_hub_download(
            LENS_REPO, filename=LENS_FILE,
            local_files_only=not args.allow_download,
        ))
    lens = torch.load(args.lens, map_location="cpu", weights_only=True)
    provenance = lens.get("provenance", {})
    lens_model = provenance.get("model_id")
    if lens_model is not None and lens_model != args.model:
        raise ValueError(
            f"lens is for {lens_model}, requested model is {args.model}"
        )
    target_layer = args.target_layer or provenance.get("target_layer")
    if target_layer is None:
        raise ValueError("pass --target-layer when the lens has no provenance")
    missing = [layer for layer in args.layers if layer not in lens["J"]]
    if missing:
        raise ValueError(f"published lens is missing layers {missing}")

    dtype_name = args.dtype or ("float16" if args.device == "mps" else "bfloat16")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)

    print(f"loading {args.model} on {args.device} as {dtype_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, local_files_only=not args.allow_download,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    print(
        f"loaded: {len(blocks)} blocks, residual width "
        f"{model.get_output_embeddings().weight.shape[1]}", flush=True,
    )

    @torch.inference_mode()
    def log_probs():
        # Qwen3.5 otherwise projects every prompt position through its 248k-row
        # vocabulary head. We score only the final position, so keeping one
        # logit slice removes the dominant CPU cost without changing the metric.
        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, logits_to_keep=1,
        )
        return torch.log_softmax(output.logits[:, -1].float(), dim=-1).cpu()

    base = log_probs()
    mean_base = base.mean(0)
    order = torch.argsort(mean_base, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))

    # Construct the centred, row-normalized unembedding directions without
    # materialising the roughly 2.5 GB float32 matrix.
    embedding = model.get_output_embeddings().weight
    d_model = embedding.shape[1]
    chunk_size = 8192
    W_mean = torch.zeros(d_model, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, embedding.shape[0], chunk_size):
            W_mean += embedding[start:start + chunk_size].float().sum(0).cpu()
    W_mean /= embedding.shape[0]

    row_cache = {}

    def normalized_rows(ids):
        missing_ids = [int(i) for i in ids if int(i) not in row_cache]
        if missing_ids:
            with torch.inference_mode():
                rows = embedding[missing_ids].float().cpu() - W_mean
                rows = F.normalize(rows, dim=1)
            row_cache.update({token_id: row for token_id, row in zip(missing_ids, rows)})
        return torch.stack([row_cache[int(i)] for i in ids])

    concepts = {}
    for name, words in CONCEPTS.items():
        ids = single_token_ids(tokenizer, words.split())
        if len(ids) < 12:
            continue
        train_ids, test_ids = ids[0::2], ids[1::2]
        w = F.normalize(normalized_rows(train_ids).mean(0), dim=0)
        used = set(ids)
        controls = []
        for target_rank in rank[torch.tensor(test_ids)].tolist():
            for offset in range(800):
                found = None
                for candidate in (
                    int(order[min(target_rank + offset, len(order) - 1)]),
                    int(order[max(target_rank - offset, 0)]),
                ):
                    if candidate in used:
                        continue
                    similarity = float(normalized_rows([candidate])[0] @ w)
                    if similarity < 0.10:
                        found = candidate
                        break
                if found is not None:
                    controls.append(found)
                    used.add(found)
                    break
        concepts[name] = {
            "w": w,
            "train_ids": train_ids,
            "test_ids": torch.tensor(test_ids),
            "control_ids": torch.tensor(controls[:len(test_ids)]),
        }
    print("concept token counts:", {
        name: (len(data["train_ids"]), len(data["test_ids"]))
        for name, data in concepts.items()
    }, flush=True)

    def lifts(edited_log_probs, concept):
        delta = edited_log_probs - base
        return (
            delta[:, concept["test_ids"]].mean(1)
            - delta[:, concept["control_ids"]].mean(1)
        )

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

        handle = blocks[layer].register_forward_hook(capture)
        with torch.inference_mode():
            model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, logits_to_keep=1,
            )
        handle.remove()
        scales[layer] = float(torch.stack(captured).mean())
    print("residual scales:", scales, flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    random_rows = []
    for layer in args.layers:
        J = lens["J"][layer].float()
        random_per_concept = {name: [] for name in concepts}
        for draw in range(args.n_random):
            direction = F.normalize(torch.randn(d_model, generator=generator), dim=0)
            with AddEverywhere(
                blocks[layer], direction, args.null_dose * scales[layer],
                device=args.device, dtype=dtype,
            ):
                edited = log_probs()
            for name, concept in concepts.items():
                per_prompt = lifts(edited, concept)
                value = float(per_prompt.mean())
                random_per_concept[name].append(value)
                random_rows.append({
                    "layer": layer, "concept": name, "dose": args.null_dose,
                    "draw": draw, "lift": value,
                    "prompt_lifts": [float(x) for x in per_prompt],
                })

        for name, concept in concepts.items():
            directions = {
                "pullback": J.T @ concept["w"],
                "direct_w": concept["w"],
            }
            null = torch.tensor(random_per_concept[name])
            null_mean = float(null.mean())
            null_sd = float(null.std(unbiased=True)) if len(null) > 1 else 0.0
            for dose in args.doses:
                for method, direction in directions.items():
                    with AddEverywhere(
                        blocks[layer], direction, dose * scales[layer],
                        device=args.device, dtype=dtype,
                    ):
                        edited = log_probs()
                    per_prompt = lifts(edited, concept)
                    value = float(per_prompt.mean())
                    row = {
                        "layer": layer, "concept": name, "dose": dose,
                        "method": method, "lift": value,
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

        output = {
            "config": {
                "model": args.model, "lens": str(args.lens),
                "layers": args.layers, "doses": args.doses,
                "target_layer": target_layer,
                "null_dose": args.null_dose, "n_random": args.n_random,
                "seed": args.seed, "device": args.device, "dtype": dtype_name,
                "out": str(args.out),
            },
            "lens_provenance": provenance,
            "prompt_count": len(PROMPTS),
            "concepts": {
                name: {
                    "n_train_tokens": len(data["train_ids"]),
                    "n_test_tokens": len(data["test_ids"]),
                    "n_control_tokens": len(data["control_ids"]),
                }
                for name, data in concepts.items()
            },
            "scales": scales,
            "rows": rows,
            "random_rows": random_rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=1))
        print(f"layer {layer} complete", flush=True)
        del J
        if args.device == "mps":
            torch.mps.empty_cache()

    print("\nReference-dose summary")
    at_null = [r for r in rows if abs(r["dose"] - args.null_dose) < 1e-9]
    for method in ("pullback", "direct_w"):
        selected = [r for r in at_null if r["method"] == method]
        z = torch.tensor([r["z_vs_random"] for r in selected])
        print(
            f"{method:10s} mean lift {sum(r['lift'] for r in selected) / len(selected):+.3f}  "
            f"median z {z.quantile(.5).item():+.2f}  "
            f"empirical p<=.05 {sum(r['empirical_p_ge'] <= .05 for r in selected)}/{len(selected)}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
