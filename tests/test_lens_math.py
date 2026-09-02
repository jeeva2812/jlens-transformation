"""Verify the VJP shortcut against a brute-force Jacobian on a tiny model.

The claim being tested is the one the whole project rests on:

    seeding every final position with v and taking ONE backward pass gives, at
    anchor position t,   sum_{t' >= t} J[t, t']^T v

If that is wrong, every number downstream is wrong and nothing looks broken.
Here we compute the right-hand side the slow, obvious way -- one backward pass
per t' -- and check the two agree.

Run:  .venv/bin/python tests/test_lens_math.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import _ResidualCapture

MODEL = "HuggingFaceTB/SmolLM2-135M"   # tiny, Llama-shaped; only a code smoke test
LAYER = 4


def main():
    torch.manual_seed(0)
    device = "cpu"   # float32 on CPU: we are checking maths, not speed

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    enc = tok("The capital of France is Paris and the weather", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = torch.ones_like(input_ids)
    T = input_ids.shape[1]
    print(f"sequence length T={T}")

    W_U = model.get_output_embeddings().weight
    v = W_U[tok.encode(" Paris", add_special_tokens=False)[0]].detach()
    d_model = v.shape[0]
    print(f"d_model={d_model}")

    # ---- fast path: one backward, every final position seeded with v ----
    with _ResidualCapture(model, LAYER, LAYER + 6) as cap:
        with torch.enable_grad():
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        h_l, h_final = cap.h_l, cap.h_final
    assert h_l is not None and h_final is not None, "hooks did not fire"
    assert h_l.shape == (1, T, d_model), f"unexpected h_l shape {h_l.shape}"

    grad_out = v.view(1, 1, -1).expand(1, T, d_model)
    (fast,) = torch.autograd.grad(h_final, h_l, grad_outputs=grad_out, retain_graph=True)

    # ---- slow path: one backward per t', summed ----
    slow = torch.zeros_like(fast)
    for t_prime in range(T):
        seed = torch.zeros(1, T, d_model)
        seed[0, t_prime] = v
        (g,) = torch.autograd.grad(h_final, h_l, grad_outputs=seed, retain_graph=True)
        slow += g

    # ---- compare ----
    err = (fast - slow).abs().max().item()
    scale = slow.abs().max().item()
    print(f"max abs diff  {err:.3e}   (signal scale {scale:.3e})")
    assert err < 1e-3 * max(scale, 1.0), "fast path does not match brute force"
    print("PASS: one seeded backward == sum over downstream positions")

    # ---- causality: h_l[t] must not influence h_final[t'] for t' < t ----
    # Seed ONLY position 0 and check the gradient vanishes at every t > 0.
    seed = torch.zeros(1, T, d_model)
    seed[0, 0] = v
    (g0,) = torch.autograd.grad(h_final, h_l, grad_outputs=seed, retain_graph=True)
    leak = g0[0, 1:].abs().max().item()
    print(f"causality leak (should be ~0): {leak:.3e}")
    assert leak < 1e-5, "gradient leaked backwards in time -- mask is not causal"
    print("PASS: causal mask confirmed, so t' >= t summation is implicit")


if __name__ == "__main__":
    main()
