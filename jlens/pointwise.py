"""What does averaging the Jacobian cost you?

`J_l` is an expectation over prompts.  The per-prompt Jacobians that go into it
are never looked at -- `lens.py` computes `per_prompt` as a (B, d_model) tensor
and immediately sums it away.  This keeps it.

For each prompt `p` we get its own pullback `v_p = J_p^T w` from one backward
pass, at the same cost as the averaged estimate, then ask two questions.

**Geometry.** How much do the `v_p` disagree with each other, and with their mean?

**Causal, which is the one that matters.** Steer prompt `p` and score only prompt
`p`, at matched intervention norm, with:

    own          v_p                      the prompt's own Jacobian
    loo_avg      mean of v_q, q != p      the averaged lens, leaving p out
    mismatched   v_q for a derangement q  another prompt's Jacobian
    saved_lens   J^T w from the lens file the actual published averaged object
    direct_w     w                        no Jacobian at all
    random       matched-norm noise       the null

`loo_avg` leaves the scored prompt out of the average, so "own beats average" is
not leakage.  `mismatched` is the control that separates "this prompt's Jacobian
carries prompt-specific signal" from "any single-prompt Jacobian beats a blurred
average" -- without it the comparison proves nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.contrastive_logit_steering import PROMPTS, one_token
from jlens.lens import _ResidualCapture, _find_blocks_and_norm


class AddPerPrompt:
    """Add a DIFFERENT unit direction to each row of the batch, at one layer.

    `directions` is (B, d_model); each row is normalised independently and scaled
    by the same magnitude, so every prompt receives an intervention of identical
    L2 norm -- the same matched-norm discipline as the rest of the project.
    """

    def __init__(self, block, directions, magnitude, *, device, dtype):
        delta = F.normalize(directions.float(), dim=-1) * magnitude
        self.delta = delta.to(device, dtype).unsqueeze(1)          # (B, 1, d)
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = tensor + self.delta
        return edited if torch.is_tensor(output) else (edited,) + tuple(output[1:])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def per_prompt_pullbacks(model, ids, mask, layer, target_layer, w, skip_first=1):
    """v_p = J_p^T w for every prompt, from one backward pass.

    Reduction matches `jlens.lens.lens_vectors` exactly: seed every non-padding
    destination position, take the mean over valid source positions
    [skip_first, len-1) within each prompt, and do NOT pool across prompts.
    """
    with _ResidualCapture(model, layer, target_layer) as cap:
        with torch.enable_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        h_l, h_final = cap.h_l, cap.h_final
    if h_l is None or h_final is None:
        raise RuntimeError("hooks did not fire")

    B, T, d = h_final.shape
    grad_out = w.view(1, 1, -1).expand(B, T, d).to(h_final.dtype).to(h_final.device)
    grad_out = grad_out * mask.unsqueeze(-1).to(grad_out.dtype)
    (g,) = torch.autograd.grad(outputs=h_final, inputs=h_l, grad_outputs=grad_out)

    lengths = mask.sum(dim=1, keepdim=True)
    seq_pos = mask.long().cumsum(dim=1) - 1
    valid = (seq_pos >= skip_first) & (seq_pos < lengths - 1) & mask.bool()

    # A prompt shorter than skip_first + 2 tokens has NO valid source position,
    # and would silently return a zero vector that F.normalize leaves at zero --
    # an intervention of no effect, averaged in as if it were data.  The
    # published Qwen lens uses skip_first=4, which is fine on the 128-token Pile
    # documents it was estimated from and empties 4 of these 12 short steering
    # prompts.  Fail loudly instead.
    empty = (valid.sum(dim=1) == 0).nonzero().flatten().tolist()
    if empty:
        raise ValueError(
            f"prompts {empty} have no valid source position at skip_first="
            f"{skip_first} (lengths {[int(x) for x in lengths.flatten()]}). "
            f"Lower --skip-first or use longer prompts."
        )

    g = g * valid.unsqueeze(-1).to(g.dtype)
    counts = valid.sum(dim=1).unsqueeze(-1)
    return (g.sum(dim=1) / counts.to(g.dtype)).float().cpu()        # (B, d_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--scale-json", type=Path, required=True)
    ap.add_argument("--dose", type=float, default=.15)
    ap.add_argument("--skip-first", type=int, default=1,
                    help="match the lens's own provenance (Qwen's published lens uses 4)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
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
    w = F.normalize((embedding[rome] - embedding[paris]).detach().float().cpu(), dim=0)

    v = per_prompt_pullbacks(model, ids, mask, a.layer, a.target_layer,
                             w.to(a.device), skip_first=a.skip_first)   # (B, d)
    B = v.shape[0]
    unit = F.normalize(v, dim=-1)

    # --- geometry -----------------------------------------------------------
    cosines = (unit @ unit.T)
    off = [float(cosines[i, j]) for i in range(B) for j in range(B) if i != j]
    mean_v = v.mean(0)
    cos_to_mean = [float(F.cosine_similarity(v[i], mean_v, dim=0)) for i in range(B)]
    saved = torch.load(a.lens, map_location="cpu", weights_only=False)["J"][a.layer].float().T @ w
    cos_to_saved = [float(F.cosine_similarity(v[i], saved, dim=0)) for i in range(B)]
    cos_mean_saved = float(F.cosine_similarity(mean_v, saved, dim=0))
    norms = [float(x) for x in v.norm(dim=-1)]

    print(f"{a.model}  layer {a.layer} -> target {a.target_layer}")
    print(f"per-prompt pullbacks: {B} prompts")
    print(f"  pairwise cosine between prompts : mean {sum(off)/len(off):+.3f}  "
          f"min {min(off):+.3f}  max {max(off):+.3f}")
    print(f"  cosine to their own mean        : mean {sum(cos_to_mean)/B:+.3f}  "
          f"min {min(cos_to_mean):+.3f}")
    print(f"  cosine to the saved averaged J  : mean {sum(cos_to_saved)/B:+.3f}")
    print(f"  mean-of-v vs saved averaged J   : {cos_mean_saved:+.3f}")
    print(f"  norm spread                     : {min(norms):.3f} to {max(norms):.3f}\n")

    # --- causal -------------------------------------------------------------
    blob = json.loads(a.scale_json.read_text())
    scales = blob.get("scales", blob.get("config", {}).get("scales", {}))
    scale = float(scales.get(str(a.layer), scales.get(a.layer)))

    loo = torch.stack([(v.sum(0) - v[i]) / (B - 1) for i in range(B)])
    shift = torch.tensor([(i + 1) % B for i in range(B)])     # derangement
    generator = torch.Generator().manual_seed(20260911)

    conditions = {
        "own": v,
        "loo_avg": loo,
        "mismatched": v[shift],
        "saved_lens": saved.unsqueeze(0).expand(B, -1),
        "direct_w": w.unsqueeze(0).expand(B, -1),
        "random": torch.randn(B, v.shape[1], generator=generator),
    }

    @torch.inference_mode()
    def logits():
        return model(input_ids=ids, attention_mask=mask, use_cache=False,
                     logits_to_keep=1).logits[:, -1].float().cpu()

    clean = logits()
    results = {}
    for name, directions in conditions.items():
        with AddPerPrompt(blocks[a.layer], directions, a.dose * scale,
                          device=a.device, dtype=dtype):
            z = logits()
        delta = z - clean
        effect = delta[:, rome] - delta[:, paris]
        results[name] = {"mean": float(effect.mean()),
                         "per_prompt": [float(x) for x in effect]}

    print(f"{'condition':14s} {'mean Δ(Rome−Paris)':>20s}   what it tests")
    blurb = {
        "own": "the prompt's own Jacobian",
        "loo_avg": "averaged lens, this prompt left out",
        "mismatched": "another prompt's Jacobian",
        "saved_lens": "the saved averaged J",
        "direct_w": "no Jacobian at all",
        "random": "matched-norm null",
    }
    for name, row in results.items():
        print(f"{name:14s} {row['mean']:20.3f}   {blurb[name]}")

    own_beats_loo = sum(o > l for o, l in
                        zip(results["own"]["per_prompt"], results["loo_avg"]["per_prompt"]))
    own_beats_mis = sum(o > m for o, m in
                        zip(results["own"]["per_prompt"], results["mismatched"]["per_prompt"]))
    print(f"\nown > leave-one-out average in {own_beats_loo}/{B} prompts")
    print(f"own > another prompt's Jacobian in {own_beats_mis}/{B} prompts")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"lens": str(a.lens), "out": str(a.out)},
        "prompts": PROMPTS,
        "geometry": {
            "pairwise_cosine_mean": sum(off) / len(off),
            "pairwise_cosine_min": min(off), "pairwise_cosine_max": max(off),
            "cosine_to_own_mean": cos_to_mean,
            "cosine_to_saved_lens": cos_to_saved,
            "mean_of_v_vs_saved_lens": cos_mean_saved,
            "norms": norms,
            "cosine_matrix": [[float(x) for x in row] for row in cosines],
        },
        "causal": results,
        "summary": {
            "own_beats_loo_avg": own_beats_loo,
            "own_beats_mismatched": own_beats_mis,
            "n_prompts": B,
        },
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
