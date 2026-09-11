"""If the change is not in a few directions, ask what it does to ALL of them.

The top-subspace test found nothing misalignment-specific, but a control in the
readout was suggestive: a RANDOM direction read through the medical organism's J
returned medical vocabulary (injury, muscular, spine, tumor). If that holds
across many random directions, the fine-tune's effect on the transport is
diffuse -- it tilts J everywhere by a little rather than rewriting a few
directions by a lot. That is a different shape of claim from "there is a
misalignment direction", and it is the one the earlier results actually support.

Method: sample random unit directions, read each through the organism's J and
through the base's J, and average the logit DIFFERENCE per token. Averaging over
random directions is the point -- it cancels whatever any single direction
happens to encode and leaves only what the fine-tune does to the transport in
general.

Control: run the same procedure with base-vs-base on disjoint random directions.
That must return nothing, or the metric is measuring sampling noise.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def diffuse(W_U, norm, J_a, J_b, d_model, n, seed, k=28):
    """Mean logit shift per token when transport J_b replaces J_a."""
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(W_U.shape[0])
    with torch.no_grad():
        for _ in range(n):
            v = torch.randn(d_model, generator=g)
            v = v / v.norm()
            acc += W_U @ norm(J_b.float() @ v) - W_U @ norm(J_a.float() @ v)
    acc /= n
    return acc.topk(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/em05"))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=26)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
    W_U, norm = m.get_output_embeddings().weight.detach(), m.model.norm

    J = {n: torch.load(a.dir / f"J_{n}.pt", map_location="cpu",
                       weights_only=False)["J"][a.layer]
         for n in ["base", "medical", "financial", "sports", "control"]}
    d = J["base"].shape[0]

    print(f"layer {a.layer}, averaged over {a.n} random directions\n")
    print("what each fine-tune adds to the transport, regardless of direction:\n")
    for n in ["medical", "financial", "sports", "control"]:
        top = diffuse(W_U, norm, J["base"], J[n], d, a.n, seed=0, k=a.k)
        words = " ".join(repr(tok.decode([i])) for i in top.indices.tolist())
        print(f"--- {n} ---")
        print(f"  {words}\n")

    print("--- CONTROL: base vs base, disjoint random directions ---")
    top = diffuse(W_U, norm, J["base"], J["base"], d, a.n, seed=1, k=a.k)
    print(f"  max |shift| = {top.values.abs().max():.2e}  (must be ~0)")


if __name__ == "__main__":
    main()
