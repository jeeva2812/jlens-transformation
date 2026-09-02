"""Step 0: does our lens reproduce the published one?

Nothing downstream means anything until this passes. We recompute lens rows for
a handful of concept tokens on Qwen3.5-4B under the exact configuration recorded
in the published artifact's provenance, and compare against
W_U[tokens] @ J_published[layer].

The published lens stores full (d_model x d_model) Jacobians, so this is a real
external check rather than a self-consistency test: if our VJP shortcut, our
target-layer convention, our position weighting, or our skip_first handling is
wrong, the cosine similarity will not be near 1.

Both weighting schemes are tried, because 'uniform' in the provenance is not
self-explanatory and the right move is to measure which one matches rather than
argue about it.

    PYTHONPATH=. .venv/bin/python -m jlens.verify --layer 20
"""

from __future__ import annotations

import argparse

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lens import LensSpec, lens_vectors

LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.5-4b/j-lens/lens.pt"

CONCEPT_WORDS = [
    " Paris", " France", " water", " code", " because", " however",
    " safe", " danger", " true", " false", " one", " two",
]


def load_published():
    path = hf_hub_download(LENS_REPO, filename=LENS_FILE)
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d, d["provenance"]


def pile_prompts(n: int) -> list[str]:
    """The published lens used NeelNanda/pile-10k. Same corpus, same order."""
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    pub, prov = load_published()
    print("published provenance:")
    for k, v in prov.items():
        print(f"  {k}: {v}")

    model_id = prov["model_id"]
    target_layer = prov["target_layer"]
    n_prompts = prov["n_prompts"]
    max_len = prov["t_max"]
    skip_first = prov["skip_first"]

    if args.layer >= target_layer:
        raise SystemExit(f"--layer must be < target_layer ({target_layer})")

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\ndevice={device}  layer={args.layer}  target={target_layer}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # float32 on MPS/CPU: we are checking numerics against a float16 artifact and
    # do not want our own precision to be the thing under test.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    token_ids = [tok.encode(w, add_special_tokens=False)[0] for w in CONCEPT_WORDS]
    print("concept tokens:", list(zip(CONCEPT_WORDS, token_ids)))

    prompts = pile_prompts(n_prompts)

    # Reference rows straight out of the published Jacobian.
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    J_pub = pub["J"][args.layer].float()                     # (d_model, d_model)
    reference = W_U[token_ids] @ J_pub                       # (K, d_model)

    for weighting in ("uniform", "per_anchor"):
        def batches():
            for i in range(0, len(prompts), args.batch_size):
                enc = tok(
                    prompts[i : i + args.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                yield enc["input_ids"].to(device), enc["attention_mask"].to(device)

        spec = LensSpec(
            layer=args.layer,
            token_ids=token_ids,
            target_layer=target_layer,
            n_prompts=n_prompts,
            max_len=max_len,
            skip_first=skip_first,
            weighting=weighting,
        )
        mine = lens_vectors(model, batches(), spec).float()

        cos = torch.nn.functional.cosine_similarity(mine, reference, dim=-1)
        scale = (mine.norm(dim=-1) / reference.norm(dim=-1).clamp(min=1e-9))

        print(f"\n=== weighting={weighting} ===")
        print(f"{'token':>10}  {'cosine':>8}  {'|mine|/|ref|':>12}")
        for w, c, s in zip(CONCEPT_WORDS, cos, scale):
            print(f"{w:>10}  {c.item():8.4f}  {s.item():12.4f}")
        print(f"{'MEAN':>10}  {cos.mean().item():8.4f}  {scale.mean().item():12.4f}")

        if cos.mean() > 0.99:
            print(">>> MATCH: this is the published convention.")

        # Is our layer index theirs? If their layer l means the residual LEAVING
        # block l rather than entering it, we are computing a neighbour's
        # Jacobian -- adjacent layers are similar, so this shows up as a high
        # but sub-unity cosine with a systematic scale offset, which is exactly
        # what we see. Compare against a window of published layers; the best
        # match tells us the offset directly.
        if weighting == "uniform":
            print("\n  offset sweep (which published layer does ours match?)")
            print(f"  {'pub layer':>10}  {'offset':>7}  {'cosine':>8}  {'scale':>8}")
            for lp in range(max(0, args.layer - 3), min(target_layer, args.layer + 4)):
                ref_l = W_U[token_ids] @ pub["J"][lp].float()
                c = torch.nn.functional.cosine_similarity(mine, ref_l, dim=-1).mean()
                s = (mine.norm(dim=-1) / ref_l.norm(dim=-1).clamp(min=1e-9)).mean()
                flag = "  <-- best" if lp == args.layer else ""
                print(f"  {lp:>10}  {lp - args.layer:>+7}  {c.item():8.4f}  {s.item():8.4f}{flag}")


if __name__ == "__main__":
    main()
