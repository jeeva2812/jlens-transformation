"""Architecture-independent self-test: at layer == target_layer, J is the identity.

If the source and destination are the same residual point, dh_target/dh_l is I by
construction, so the lens block must come back as exactly W_U[token_ids]. No
published lens is needed to check this, which matters because Olmo 3 has none.

What it actually catches: hooks pointing at the wrong module, block outputs that
are tuples handled wrongly, the position mask zeroing everything, the reduction
dividing by the wrong count, dtype or device mishaps. Any of those break this.

    PYTHONPATH=. .venv/bin/python tests/test_identity_at_target.py --model MODEL
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jlens.lens import LensSpec, lens_vectors, token_seeds

WORDS = [" Paris", " water", " because", " safe", " true", " one"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=None, help="defaults to target")
    ap.add_argument("--n-prompts", type=int, default=4)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    tc = getattr(cfg, "text_config", cfg)
    n_layers = tc.num_hidden_layers
    target = n_layers - 2
    layer = args.layer if args.layer is not None else target
    print(f"{args.model}: n_layers={n_layers} target={target} layer={layer}")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    token_ids = [tok.encode(w, add_special_tokens=False)[0] for w in WORDS]

    texts = [
        "The capital of France is Paris and the weather there is mild in spring.",
        "Water boils at one hundred degrees Celsius at standard pressure, because",
        "def quicksort(arr): pivot = arr[len(arr) // 2]  # partition and recurse",
        "The committee met on Tuesday to discuss the proposed budget revisions.",
    ][: args.n_prompts]

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"].to(device), enc["attention_mask"].to(device)

    spec = LensSpec(
        layer=layer, target_layer=target,
        n_prompts=len(texts), max_len=128, skip_first=4, weighting="uniform",
    )
    got = lens_vectors(model, batches(), spec, token_seeds(model, token_ids)).float()
    want = model.get_output_embeddings().weight[token_ids].detach().float().cpu()

    cos = torch.nn.functional.cosine_similarity(got, want, dim=-1)
    ratio = got.norm(dim=-1) / want.norm(dim=-1).clamp(min=1e-9)
    print(f"\n{'token':>10}  {'cosine':>8}  {'ratio':>8}")
    for w, c, r in zip(WORDS, cos, ratio):
        print(f"{w:>10}  {c.item():8.5f}  {r.item():8.5f}")
    print(f"{'MEAN':>10}  {cos.mean().item():8.5f}  {ratio.mean().item():8.5f}")

    ok = cos.mean() > 0.9999 and abs(ratio.mean() - 1) < 1e-3
    print("\nPASS: J is the identity at the target layer" if ok else
          "\nFAIL: the plumbing is wrong -- do not trust any lens from this model")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
