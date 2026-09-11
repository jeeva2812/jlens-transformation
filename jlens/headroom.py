"""Is the semantic generalisation real, or just headroom?

The specificity claim is that pulling back `W_U[Rome] - W_U[Paris]` moves held-out
Italy-vs-France tokens while leaving unrelated geography alone.  The obvious
objection is headroom: a token that was already unlikely has more room to move, so
any intervention will appear to "prefer" rare tokens, and the Italy words may
simply be rarer than the control words.

This measures it directly.  One clean forward pass and one steered forward pass
give the change in logit for *every* token in the vocabulary, not just the labelled
ones.  Regress that change on the token's clean log-probability across the whole
vocabulary, and the fitted line is what headroom alone predicts.  A labelled group
has a real effect only if it sits above that line.

Reported per group: raw mean shift, the shift headroom predicts, the residual, and
that residual as a z-score against the spread of residuals over the vocabulary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.contrastive_logit_steering import PROMPTS, one_token
from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere

GROUPS = {
    "target_rome": ["Rome"],
    "target_paris": ["Paris"],
    "italy_heldout": ["Italy", "Italian", "Italians", "Roman", "Romans",
                      "Vatican", "Colosseum", "Milan", "Venice", "Naples",
                      "Florence", "pizza", "pasta"],
    "france_heldout": ["France", "French", "Parisian", "Parisians", "Eiffel",
                       "Louvre", "Versailles", "Lyon", "Marseille", "Bordeaux",
                       "croissant", "baguette"],
    "control_japan": ["Japan", "Japanese", "Tokyo", "Osaka", "sushi"],
    "control_china": ["China", "Chinese", "Beijing", "Shanghai"],
    "control_spain": ["Spain", "Spanish", "Madrid", "Barcelona", "paella"],
    "control_germany": ["Germany", "German", "Berlin", "Munich", "Hamburg"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--scale-json", type=Path, required=True)
    ap.add_argument("--dose", type=float, default=.15)
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
    w = F.normalize(
        (embedding[rome] - embedding[paris]).detach().float().cpu(), dim=0
    )
    J = torch.load(a.lens, map_location="cpu", weights_only=False)["J"][a.layer].float()
    pullback = J.T @ w

    blob = json.loads(a.scale_json.read_text())
    scales = blob.get("scales", blob.get("config", {}).get("scales", {}))
    scale = float(scales.get(str(a.layer), scales.get(a.layer)))

    @torch.inference_mode()
    def logits():
        return model(
            input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1
        ).logits[:, -1].float().cpu()

    clean = logits()
    with AddEverywhere(blocks[a.layer], pullback, a.dose * scale,
                       device=a.device, dtype=dtype):
        steered = logits()

    # Averaged over prompts: clean log-prob, and the shift the pullback induces.
    base = clean.log_softmax(-1).mean(0)
    shift = (steered - clean).mean(0)

    # Headroom model: fit shift = intercept + slope * base over the whole vocabulary.
    # Tokens the model never puts any mass on are excluded -- their log-prob is
    # floored by float precision and they would dominate the fit.
    keep = base > -25.0
    x, y = base[keep], shift[keep]
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    intercept = y.mean() - slope * x.mean()
    residual_all = y - (intercept + slope * x)
    residual_sd = float(residual_all.std())
    r = float(((x - x.mean()) * (y - y.mean())).sum()
              / (((x - x.mean()) ** 2).sum().sqrt() * ((y - y.mean()) ** 2).sum().sqrt()))

    print(f"{a.model}  layer {a.layer}  dose {a.dose}")
    print(f"vocabulary headroom fit over {int(keep.sum())} tokens: "
          f"shift = {float(intercept):+.3f} {float(slope):+.4f} * log p(token)   r = {r:+.3f}")
    print(f"residual sd = {residual_sd:.3f}\n")

    rows = {}
    print(f"{'group':20s} {'n':>3s} {'raw':>7s} {'headroom':>9s} {'residual':>9s} {'z':>7s}")
    for name, words in GROUPS.items():
        token_ids = [one_token(tok, word) for word in words]
        kept = [(word, i) for word, i in zip(words, token_ids)
                if i is not None and bool(keep[i])]
        if not kept:
            continue
        index = torch.tensor([i for _, i in kept])
        raw = float(shift[index].mean())
        predicted = float((intercept + slope * base[index]).mean())
        residual = raw - predicted
        z = residual / (residual_sd / len(index) ** .5)
        rows[name] = {
            "words_used": [word for word, _ in kept],
            "n": len(kept),
            "mean_base_logprob": float(base[index].mean()),
            "raw_shift": raw,
            "headroom_predicted_shift": predicted,
            "residual": residual,
            "z_vs_vocabulary_residuals": z,
            "per_word": {word: {"base_logprob": float(base[i]),
                                "shift": float(shift[i]),
                                "residual": float(shift[i] - (intercept + slope * base[i]))}
                         for word, i in kept},
        }
        print(f"{name:20s} {len(kept):3d} {raw:+7.3f} {predicted:+9.3f} "
              f"{residual:+9.3f} {z:+7.1f}")

    # The claims that matter are contrasts between groups, not single groups.
    def contrast(positive, negative, label):
        if positive not in rows or negative not in rows:
            return None
        raw = rows[positive]["raw_shift"] - rows[negative]["raw_shift"]
        res = rows[positive]["residual"] - rows[negative]["residual"]
        print(f"  {label:28s} raw {raw:+6.3f}   headroom-corrected {res:+6.3f}")
        return {"raw": raw, "headroom_corrected": res}

    print("\ncontrasts (the actual claims):")
    contrasts = {
        "italy_minus_france": contrast("italy_heldout", "france_heldout",
                                       "Italy - France (held out)"),
        "japan_minus_china": contrast("control_japan", "control_china",
                                      "Japan - China (control)"),
        "spain_minus_germany": contrast("control_spain", "control_germany",
                                        "Spain - Germany (control)"),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"lens": str(a.lens), "out": str(a.out)},
        "headroom_fit": {"intercept": float(intercept), "slope": float(slope),
                         "r": r, "residual_sd": residual_sd,
                         "n_tokens": int(keep.sum()),
                         "note": "shift = intercept + slope * clean log p(token), "
                                 "fit over all vocabulary tokens with log p > -25"},
        "groups": rows,
        "contrasts": contrasts,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
