"""Steer an artificial +Rome/-Paris logit objective through a saved J-Lens.

The target is deliberately exact rather than discovered: w = W_U[Rome] -
W_U[Paris].  We compare direct residual addition, J.T@w, its sign reversal, and
spectrally truncated pullbacks at equal intervention norm.  Related Italy and
France tokens are held out from construction and reported as an exploratory
semantic-generalisation score.
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


PROMPTS = [
    "The capital city they discussed was", "She booked a flight to",
    "The school trip would take them to", "The postcard had been sent from",
    "For their summer holiday, they chose", "The historian gave a lecture about",
    "The ancient ruins were located near", "The museum exhibition focused on",
    "Their train arrived in", "The diplomatic meeting was held in",
    "He opened the guidebook to the chapter on", "The restaurant was inspired by",
]
ITALY_WORDS = ["Italy", "Italian", "Roman", "Vatican", "Colosseum"]
FRANCE_WORDS = ["France", "French", "Parisian", "Eiffel", "Louvre"]
CATEGORY_PAIRS = {
    "country": (["Italy"], ["France"]),
    "language_and_people": (["Italian", "Italians", "Romans"],
                            ["French", "Parisians"]),
    "other_cities": (["Milan", "Venice", "Naples", "Florence", "Turin"],
                     ["Lyon", "Marseille", "Bordeaux", "Nice", "Toulouse"]),
    "landmarks": (["Vatican", "Colosseum", "Pantheon"],
                  ["Eiffel", "Louvre", "Versailles"]),
    "food": (["pizza", "pasta", "gelato", "risotto"],
             ["croissant", "baguette", "crepe", "brie"]),
    # These are not Rome/Paris associations. They reveal whether an arbitrary
    # geographic contrast moves too, which would weaken a specificity claim.
    "control_japan_china": (["Japan", "Japanese", "Tokyo"],
                            ["China", "Chinese", "Beijing"]),
    "control_spain_germany": (["Spain", "Spanish", "Madrid"],
                              ["Germany", "German", "Berlin"]),
}


def one_token(tokenizer, word):
    for candidate in (" " + word, word):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--scale-json", type=Path, required=True)
    ap.add_argument("--dose", type=float, default=.15)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 32, 64, 256])
    ap.add_argument("--svd-niter", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=30,
                    help="matched-norm random draws forming the null distribution")
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

    rome, paris = one_token(tok, "Rome"), one_token(tok, "Paris")
    if rome is None or paris is None:
        raise ValueError("Rome and Paris must each have a single-token spelling")
    # Exact linear logit contrast at the unembedding: +1 Rome, -1 Paris.
    w = embedding[rome].detach().float().cpu() - embedding[paris].detach().float().cpu()
    w = F.normalize(w, dim=0)

    italy = [one_token(tok, x) for x in ITALY_WORDS]
    france = [one_token(tok, x) for x in FRANCE_WORDS]
    italy = [x for x in italy if x is not None and x not in (rome, paris)]
    france = [x for x in france if x is not None and x not in (rome, paris)]
    categories = {}
    for name, (positive_words, negative_words) in CATEGORY_PAIRS.items():
        positive = [(word, one_token(tok, word)) for word in positive_words]
        negative = [(word, one_token(tok, word)) for word in negative_words]
        categories[name] = {
            "positive": [(word, token_id) for word, token_id in positive
                         if token_id is not None and token_id not in (rome, paris)],
            "negative": [(word, token_id) for word, token_id in negative
                         if token_id is not None and token_id not in (rome, paris)],
        }

    lens = torch.load(a.lens, map_location="cpu", weights_only=False)
    J = lens["J"][a.layer].float()
    pullback = J.T @ w
    ks = sorted(set(k for k in a.ks if k <= min(J.shape)))
    q = min(min(J.shape), max(ks) + 16)
    torch.manual_seed(20260911)
    U, S, V = torch.svd_lowrank(J, q=q, niter=a.svd_niter)
    order = S.argsort(descending=True)
    U, S, V = U[:, order], S[order], V[:, order]
    coefficients = S * (U.T @ w)
    full_energy = float(pullback.square().sum())

    scales_blob = json.loads(a.scale_json.read_text())
    scales = scales_blob.get("scales", scales_blob.get("config", {}).get("scales", {}))
    scale = float(scales.get(str(a.layer), scales.get(a.layer)))

    @torch.inference_mode()
    def logits():
        return model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits[:, -1].float().cpu()

    directions = {"clean": None, "direct_w": w, "pullback": pullback,
                  "reverse_pullback": -pullback}
    for k in ks:
        directions[f"top_{k}"] = V[:, :k] @ coefficients[:k]
    # Draw the whole null up front from one seeded generator.  Draw 0 is kept
    # under the name "random" so earlier saved results stay comparable.
    generator = torch.Generator().manual_seed(20260911)
    random_draws = [torch.randn(J.shape[1], generator=generator)
                    for _ in range(max(1, a.n_random))]
    directions["random"] = random_draws[0]

    clean = logits()
    method_logits = {"clean": clean}
    for name, direction in directions.items():
        if direction is None:
            continue
        with AddEverywhere(
            blocks[a.layer], direction, a.dose * scale, device=a.device, dtype=dtype
        ):
            method_logits[name] = logits()

    def summarise(z):
        """The three quantities we make claims about, for one intervention."""
        delta = z - clean
        row = {
            "target": float((delta[:, rome] - delta[:, paris]).mean()),
            "related": float((delta[:, italy].mean(1) - delta[:, france].mean(1)).mean())
            if italy and france else float("nan"),
            "categories": {},
        }
        for category_name, category in categories.items():
            positive_ids = [token_id for _, token_id in category["positive"]]
            negative_ids = [token_id for _, token_id in category["negative"]]
            if positive_ids and negative_ids:
                row["categories"][category_name] = float(
                    (delta[:, positive_ids].mean(1) - delta[:, negative_ids].mean(1)).mean()
                )
        return row

    null_rows = []
    for i, direction in enumerate(random_draws):
        z = method_logits["random"] if i == 0 else None
        if z is None:
            with AddEverywhere(
                blocks[a.layer], direction, a.dose * scale, device=a.device, dtype=dtype
            ):
                z = logits()
        null_rows.append(summarise(z))

    result = {
        "config": vars(a) | {"lens": str(a.lens), "out": str(a.out)},
        "target": {"positive": tok.decode(rome), "negative": tok.decode(paris),
                   "equation": "normalize(W_U[Rome] - W_U[Paris])"},
        "heldout_related": {
            "italy": [tok.decode(x) for x in italy],
            "france": [tok.decode(x) for x in france],
        },
        "category_tokens": {
            name: {
                side: [{"word": word, "token": tok.decode(token_id), "id": token_id}
                       for word, token_id in values[side]]
                for side in ("positive", "negative")
            }
            for name, values in categories.items()
        },
        "residual_scale": scale,
        "svd": {
            "top_singular_values": [float(x) for x in S[:20]],
            "top_signed_coefficients": [float(x) for x in coefficients[:20]],
            "energy_share": {str(k): float(coefficients[:k].square().sum() / full_energy)
                             for k in ks},
        },
        "methods": {}, "prompt_rows": [],
    }
    for name, z in method_logits.items():
        delta = z - clean
        target_change = delta[:, rome] - delta[:, paris]
        related_change = (delta[:, italy].mean(1) - delta[:, france].mean(1)
                          if italy and france else torch.full((len(PROMPTS),), float("nan")))
        probabilities = z.softmax(-1)
        vals, inds = probabilities.topk(8, dim=-1)
        result["methods"][name] = {
            "mean_target_logit_contrast_change": float(target_change.mean()),
            "mean_heldout_related_logit_contrast_change": float(related_change.mean()),
            "prompt_target_changes": [float(x) for x in target_change],
            "prompt_related_changes": [float(x) for x in related_change],
            "category_contrast_changes": {},
        }
        for category_name, category in categories.items():
            positive_ids = [token_id for _, token_id in category["positive"]]
            negative_ids = [token_id for _, token_id in category["negative"]]
            if positive_ids and negative_ids:
                per_prompt = delta[:, positive_ids].mean(1) - delta[:, negative_ids].mean(1)
                result["methods"][name]["category_contrast_changes"][category_name] = {
                    "mean": float(per_prompt.mean()),
                    "per_prompt": [float(x) for x in per_prompt],
                    "positive_prompts": int((per_prompt > 0).sum()),
                }
        for i, prompt in enumerate(PROMPTS):
            if len(result["prompt_rows"]) <= i:
                result["prompt_rows"].append({"prompt": prompt, "methods": {}})
            result["prompt_rows"][i]["methods"][name] = {
                "rome_probability": float(probabilities[i, rome]),
                "paris_probability": float(probabilities[i, paris]),
                "target_logit_contrast_change": float(target_change[i]),
                "heldout_related_contrast_change": float(related_change[i]),
                "top8": [{"token": tok.decode(int(j)), "probability": float(v)}
                         for v, j in zip(vals[i], inds[i])],
            }

    # --- matched-norm null over n_random draws -------------------------------
    def quantity(row, key):
        return row["categories"][key[4:]] if key.startswith("cat:") else row[key]

    keys = ["target", "related"] + [f"cat:{c}" for c in null_rows[0]["categories"]]
    null = {}
    for key in keys:
        draws = [quantity(row, key) for row in null_rows]
        mean = sum(draws) / len(draws)
        sd = (sum((x - mean) ** 2 for x in draws) / len(draws)) ** .5
        null[key] = {"n": len(draws), "mean": mean, "sd": sd,
                     "min": min(draws), "max": max(draws), "draws": draws}

    observed = {name: summarise(z) for name, z in method_logits.items()
                if name not in ("clean", "random")}
    result["random_null"] = {
        "n_random": len(null_rows),
        "seed": 20260911,
        "note": "every draw injected at the same L2 norm as every named method",
        "distribution": null,
        "z_scores": {
            name: {
                key: ((quantity(row, key) - null[key]["mean"]) / null[key]["sd"]
                      if null[key]["sd"] > 0 else float("nan"))
                for key in keys
            } for name, row in observed.items()
        },
        "exceeds_all_draws": {
            name: {key: bool(abs(quantity(row, key)) > max(abs(x) for x in null[key]["draws"]))
                   for key in keys}
            for name, row in observed.items()
        },
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=str))
    print(a.model, "layer", a.layer)
    for name, row in result["methods"].items():
        print(name, "target", round(row["mean_target_logit_contrast_change"], 3),
              "related", round(row["mean_heldout_related_logit_contrast_change"], 3))
    print(f"\nnull over {len(null_rows)} matched-norm random draws:")
    for key in keys:
        d = null[key]
        print(f"  {key:28s} mean {d['mean']:+7.3f}  sd {d['sd']:.3f} "
              f" range [{d['min']:+.3f}, {d['max']:+.3f}]")
    print("\nz against that null:")
    for name in ("direct_w", "pullback", "reverse_pullback"):
        if name in result["random_null"]["z_scores"]:
            z = result["random_null"]["z_scores"][name]
            print(f"  {name:20s} target z={z['target']:+8.2f}   related z={z['related']:+8.2f}")


if __name__ == "__main__":
    main()
