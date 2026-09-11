"""Is the per-prompt Jacobian sensitive to topic, or to computational regime?

Prediction, written before running (jlens-transformation, 11 Sep):

    J is sensitive to computational REGIME, not to semantic TOPIC.  Two English
    prose prompt sets about different subjects will give near-identical J_p once
    centred; prose vs code, or English vs French, will separate.

The design makes that a single comparison.  `food_fr` is a literal translation of
`food_en`, so:

    food_en vs abstract_en   same regime, DIFFERENT topic
    food_en vs food_fr       same topic,  DIFFERENT language
    food_en vs code          different regime entirely

If topic drives J, food_en sits closer to food_fr.
If regime drives J, food_en sits closer to abstract_en.

Centring is not optional.  J = prod(I + A_k), so every J_p contains the same
identity term and raw cosines are inflated toward 1 by a component that has
nothing to do with the prompt.  Both raw and centred numbers are reported; the
centred ones are the ones to read.

The causal half asks whether conditioning buys anything:

    own               v_p                              oracle upper bound
    own_group_loo     mean of v_q in the same group    a regime-conditional lens
    global_loo        mean of all v_q, q != p          the current method
    other_group       mean of v_q from another group   mismatch control
    direct_w          w                                no Jacobian
    random            noise                            null
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.contrastive_logit_steering import one_token
from jlens.lens import _find_blocks_and_norm
from jlens.pointwise import AddPerPrompt, per_prompt_pullbacks

GROUPS = {
    # English prose, concrete subject.
    "food_en": [
        "The chef tasted the sauce and then added",
        "Dinner that evening began with a bowl of",
        "The recipe called for a generous amount of",
        "At the market she filled her basket with",
        "The bakery on the corner was famous for",
    ],
    # English prose, abstract subject. Same register, same rough length.
    "abstract_en": [
        "The philosopher argued that justice depended on",
        "Freedom in that sense rests entirely upon",
        "The theory explains how meaning arises from",
        "In ethics the central difficulty concerns",
        "The lecture on consciousness returned again to",
    ],
    # Literal translations of food_en: same topic, different language.
    "food_fr": [
        "Le chef a goute la sauce puis a ajoute",
        "Le diner ce soir-la a commence par un bol de",
        "La recette demandait une bonne quantite de",
        "Au marche elle a rempli son panier de",
        "La boulangerie du coin etait celebre pour",
    ],
    # A different computational regime entirely.
    "code": [
        "def compute_total(items):\n    result = 0\n    for item in",
        "import numpy as np\n\ndef normalize(x):\n    return x /",
        "class Parser:\n    def __init__(self, path):\n        self.path =",
        "for index, value in enumerate(rows):\n    if value >",
        "with open(filename) as handle:\n    data = json.load(",
    ],
}


def mean_offdiag(cos, rows, cols):
    vals = [float(cos[i, j]) for i in rows for j in cols if i != j]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--scale-json", type=Path, required=True)
    ap.add_argument("--dose", type=float, default=.15)
    ap.add_argument("--skip-first", type=int, default=1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    names, prompts = [], []
    for group, items in GROUPS.items():
        names += [group] * len(items)
        prompts += items

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True)
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
    w = F.normalize((embedding[rome] - embedding[paris]).detach().float().cpu(), dim=0)

    v = per_prompt_pullbacks(model, ids, mask, a.layer, a.target_layer,
                             w.to(a.device), skip_first=a.skip_first)
    B = v.shape[0]
    index = {g: [i for i in range(B) if names[i] == g] for g in GROUPS}

    # Confound bookkeeping: length and final token differ across regimes by
    # construction, and either could fake a grouping on its own.
    lengths = [int(x) for x in mask.sum(1)]
    finals = [tok.decode(int(ids[i, -1])) for i in range(B)]

    raw = F.normalize(v, dim=-1) @ F.normalize(v, dim=-1).T
    centred_v = v - v.mean(0, keepdim=True)
    centred = F.normalize(centred_v, dim=-1) @ F.normalize(centred_v, dim=-1).T

    print(f"{a.model}  layer {a.layer} -> target {a.target_layer}\n")
    print("token lengths per group:",
          {g: [lengths[i] for i in index[g]] for g in GROUPS})
    print()
    keys = list(GROUPS)
    for label, cos in (("RAW", raw), ("CENTRED", centred)):
        print(f"mean pairwise cosine, {label}")
        print("            " + "".join(f"{g:>13s}" for g in keys))
        for g in keys:
            print(f"{g:12s}" + "".join(
                f"{mean_offdiag(cos, index[g], index[h]):13.3f}" for h in keys))
        print()

    decisive = {
        "food_en vs abstract_en (same regime, diff topic)":
            mean_offdiag(centred, index["food_en"], index["abstract_en"]),
        "food_en vs food_fr     (same topic, diff language)":
            mean_offdiag(centred, index["food_en"], index["food_fr"]),
        "food_en vs code        (diff regime)":
            mean_offdiag(centred, index["food_en"], index["code"]),
        "food_en within group":
            mean_offdiag(centred, index["food_en"], index["food_en"]),
    }
    print("THE DECISIVE COMPARISON (centred):")
    for k, val in decisive.items():
        print(f"  {k:52s} {val:+.3f}")
    verdict = ("REGIME" if decisive["food_en vs abstract_en (same regime, diff topic)"]
               > decisive["food_en vs food_fr     (same topic, diff language)"] else "TOPIC")
    print(f"  -> closer to the same-regime set means J tracks {verdict}\n")

    # --- causal -------------------------------------------------------------
    blob = json.loads(a.scale_json.read_text())
    scales = blob.get("scales", blob.get("config", {}).get("scales", {}))
    scale = float(scales.get(str(a.layer), scales.get(a.layer)))

    own_group_loo, global_loo, other_group = [], [], []
    order = list(GROUPS)
    for i in range(B):
        peers = [j for j in index[names[i]] if j != i]
        own_group_loo.append(v[peers].mean(0))
        global_loo.append((v.sum(0) - v[i]) / (B - 1))
        nxt = order[(order.index(names[i]) + 1) % len(order)]
        other_group.append(v[index[nxt]].mean(0))

    generator = torch.Generator().manual_seed(20260911)
    conditions = {
        "own": v,
        "own_group_loo": torch.stack(own_group_loo),
        "global_loo": torch.stack(global_loo),
        "other_group": torch.stack(other_group),
        "direct_w": w.unsqueeze(0).expand(B, -1),
        "random": torch.randn(B, v.shape[1], generator=generator),
    }

    @torch.inference_mode()
    def logits():
        return model(input_ids=ids, attention_mask=mask, use_cache=False,
                     logits_to_keep=1).logits[:, -1].float().cpu()

    clean = logits()
    causal = {}
    for name, directions in conditions.items():
        with AddPerPrompt(blocks[a.layer], directions, a.dose * scale,
                          device=a.device, dtype=dtype):
            z = logits()
        delta = z - clean
        effect = delta[:, rome] - delta[:, paris]
        causal[name] = {
            "mean": float(effect.mean()),
            "per_group": {g: float(effect[index[g]].mean()) for g in GROUPS},
            "per_prompt": [float(x) for x in effect],
        }

    print("causal: mean Δ(logit Rome − logit Paris)")
    print(f"{'condition':16s} {'all':>7s}" + "".join(f"{g:>13s}" for g in keys))
    for name, row in causal.items():
        print(f"{name:16s} {row['mean']:7.2f}"
              + "".join(f"{row['per_group'][g]:13.2f}" for g in keys))

    wins = sum(c > g for c, g in zip(causal["own_group_loo"]["per_prompt"],
                                     causal["global_loo"]["per_prompt"]))
    print(f"\nregime-conditional lens beats the global lens in {wins}/{B} prompts"
          f"  ({causal['own_group_loo']['mean']:+.3f} vs "
          f"{causal['global_loo']['mean']:+.3f})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"lens": str(a.lens), "out": str(a.out)},
        "groups": GROUPS,
        "prompt_group": names,
        "token_lengths": lengths,
        "final_tokens": finals,
        "geometry": {
            "raw_cosine_matrix": [[float(x) for x in r] for r in raw],
            "centred_cosine_matrix": [[float(x) for x in r] for r in centred],
            "raw_group_means": {g: {h: mean_offdiag(raw, index[g], index[h])
                                    for h in keys} for g in keys},
            "centred_group_means": {g: {h: mean_offdiag(centred, index[g], index[h])
                                        for h in keys} for g in keys},
            "decisive": decisive,
            "verdict": verdict,
        },
        "causal": causal,
        "summary": {"own_group_beats_global": wins, "n_prompts": B},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
