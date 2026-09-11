"""How noisy is the lens estimator at n_prompts=25?

verify.py reproduces the published lens only to cosine ~0.94. Before hunting for
a bug, establish the noise floor: compute our own lens twice from two disjoint
25-document subsets and compare those to each other.

  - if our two runs also agree at ~0.94, prompt sampling explains the whole gap
    and the published lens is inside our own noise band
  - if our two runs agree at ~0.999, the gap is systematic and worth hunting

This is the cheap experiment that decides which of those we are in, and it costs
the same as the run that raised the question.

    PYTHONPATH=. .venv/bin/python -m jlens.selfcheck --layer 20
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lens import LensSpec, lens_vectors
from .verify import CONCEPT_WORDS, load_published


def docs(start: int, n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in range(start, start + n)]


def compute(model, tok, device, token_ids, prompts, layer, target, max_len, skip_first):
    def batches():
        for p in prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len)
            yield enc["input_ids"].to(device), enc["attention_mask"].to(device)

    spec = LensSpec(
        layer=layer,
        token_ids=token_ids,
        target_layer=target,
        n_prompts=len(prompts),
        max_len=max_len,
        skip_first=skip_first,
        weighting="uniform",
    )
    return lens_vectors(model, batches(), spec).float()


def report(name, a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    ratio = a.norm(dim=-1) / b.norm(dim=-1).clamp(min=1e-9)
    print(f"\n=== {name} ===")
    print(f"{'token':>10}  {'cosine':>8}  {'ratio':>8}")
    for w, c, r in zip(CONCEPT_WORDS, cos, ratio):
        print(f"{w:>10}  {c.item():8.4f}  {r.item():8.4f}")
    print(f"{'MEAN':>10}  {cos.mean().item():8.4f}  {ratio.mean().item():8.4f}")
    return cos.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    pub, prov = load_published()
    model_id, target = prov["model_id"], prov["target_layer"]
    max_len, skip_first = prov["t_max"], prov["skip_first"]

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    token_ids = [tok.encode(w, add_special_tokens=False)[0] for w in CONCEPT_WORDS]
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    reference = W_U[token_ids] @ pub["J"][args.layer].float()

    kw = dict(
        layer=args.layer, target=target, max_len=max_len, skip_first=skip_first
    )
    print(f"device={device} layer={args.layer} target={target} n={args.n}")

    a = compute(model, tok, device, token_ids, docs(0, args.n), **kw)
    b = compute(model, tok, device, token_ids, docs(args.n, args.n), **kw)

    self_cos = report(f"ours[docs 0..{args.n-1}] vs ours[docs {args.n}..{2*args.n-1}]", a, b)
    pub_cos = report("ours[docs 0..24] vs published", a, reference)

    print("\n" + "=" * 60)
    print(f"self-consistency : {self_cos:.4f}")
    print(f"vs published     : {pub_cos:.4f}")
    if self_cos <= pub_cos + 0.01:
        print("\n=> Our agreement with the published lens is as good as our\n"
              "   agreement with ourselves. The gap is prompt-sampling noise at\n"
              "   n=25, not an implementation difference.")
    else:
        print("\n=> We agree with ourselves much better than with the published\n"
              "   lens. The gap is systematic. Hunt the config difference.")


if __name__ == "__main__":
    main()
