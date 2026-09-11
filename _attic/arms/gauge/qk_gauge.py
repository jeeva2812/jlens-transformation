"""A head's SECOND freedom: the query-key basis -- and what rotary position
embeddings do to it.

Attention scores are  (W_Q x) . (W_K y).  Replacing W_Q -> R W_Q and
W_K -> R W_K leaves that dot product unchanged for ANY rotation R, so in a
transformer WITHOUT rotary embeddings the query-key basis is as arbitrary as
the value-output basis. Each head would have two independent free rotations.

But rotary embeddings (RoPE) rotate coordinate PAIRS by position-dependent
angles before the dot product. A general R does not commute with that. The
prediction is precise:

    only rotations that act inside each rotary pair survive.

That takes the freedom from d_head(d_head-1)/2 parameters down to d_head/2 --
from 2016 to 32 for a 64-dimensional head. RoPE accidentally gauge-fixes almost
all of it.

This script tests that prediction: a general rotation should break the model, a
pair-wise rotation should not.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gauge.matrix import rand_orth

PROMPTS = ["The capital of France is", "def add(a, b):\n    return",
           "Water boils at a temperature of", "She opened the door and saw a"]


def pair_rotation(dh, seed):
    """block-diagonal: an independent 2x2 rotation inside each rotary pair.
    HF's RoPE pairs dimension i with i + dh/2, so the blocks are (i, i+dh/2)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.eye(dh)
    half = dh // 2
    for i in range(half):
        t = float(torch.rand(1, generator=g)) * 2 * np.pi
        c, s = np.cos(t), np.sin(t)
        R[i, i] = c;        R[i, i + half] = -s
        R[i + half, i] = s; R[i + half, i + half] = c
    return R


def main():
    mid, L, H = "unsloth/Llama-3.2-1B", 8, 1
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    c = m.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    dh = c.hidden_size // nh
    grp = nh // nkv
    at = m.model.layers[L].self_attn
    enc = [tok(p, return_tensors="pt") for p in PROMPTS]

    def logits():
        with torch.no_grad():
            return torch.stack([m(**e).logits[0, -1] for e in enc])

    base = logits()
    scale = float(base.abs().max())
    kv = H // grp
    res = {}

    for name, R in [("general rotation", rand_orth(dh, 0)),
                    ("rotation inside each rotary pair", pair_rotation(dh, 0))]:
        saved_q = at.q_proj.weight.detach().clone()      # save everything, restore exactly
        saved_k = at.k_proj.weight.detach().clone()
        with torch.no_grad():
            # every query head sharing this kv head must be rotated together
            for hq in range(kv * grp, (kv + 1) * grp):
                w = at.q_proj.weight[hq * dh:(hq + 1) * dh]
                at.q_proj.weight[hq * dh:(hq + 1) * dh] = R @ w
            at.k_proj.weight[kv * dh:(kv + 1) * dh] = R @ saved_k[kv * dh:(kv + 1) * dh]
        d = float((logits() - base).abs().max())
        with torch.no_grad():
            at.q_proj.weight.copy_(saved_q)
            at.k_proj.weight.copy_(saved_k)
        back = float((logits() - base).abs().max())
        verdict = "FREE  (model cannot tell)" if d < 1e-3 else "BREAKS the model"
        print(f"  {name:36s} max |delta logit| = {d:9.2e}   {verdict}")
        res[name] = {"dlogit": d, "restore": back}

    print(f"\n  (logits are ~{scale:.0f}; restore is exact for both: {max(v['restore'] for v in res.values() if isinstance(v,dict)):.1e})")
    free_no_rope = dh * (dh - 1) // 2
    free_rope = dh // 2
    print(f"\n  free parameters in one head's query-key basis:")
    print(f"     without rotary embeddings : {free_no_rope:6d}")
    print(f"     with rotary embeddings    : {free_rope:6d}   ({100*free_rope/free_no_rope:.1f}%)")
    print(f"  -> RoPE removes {100*(1-free_rope/free_no_rope):.1f}% of this freedom by accident.")
    print(f"     The value-output basis has no such protection: {free_no_rope} free, all of it.")
    res["free_params"] = {"no_rope": free_no_rope, "with_rope": free_rope, "d_head": dh}
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path("out/gauge/qk_gauge.json").write_text(json.dumps(res, indent=1))
    print("\nwrote out/gauge/qk_gauge.json")


if __name__ == "__main__":
    main()
