"""Run the checkpoint sweep on Modal.

    modal run modal_app.py --revisions "stage1-step100000,stage1-step400000,main" --layer 16

Checkpoints are cached in a Modal Volume rather than re-downloaded. Volume storage
is $0.09/GiB/month with the first 1 TiB free, and eight 7B checkpoints is ~112 GiB,
so caching is free and re-downloading would cost GPU-seconds. The opposite of the
right call on a rented pod with a small local disk.
"""

from __future__ import annotations

import modal

CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "accelerate",
        "numpy",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE})
    .add_local_python_source("jlens")
    .add_local_dir("data", remote_path="/root/data")
)

volume = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("jlens-results", create_if_missing=True)

app = modal.App("jlens-transformation", image=image)


@app.function(
    gpu="A100-40GB",   # 14GB of weights + a partial graph; 80GB is headroom you pay for
    volumes={CACHE: volume, "/out": results},
    timeout=60 * 60,
    scaledown_window=60,          # do not sit idle billing a GPU
)
def lens_at_revision(
    model_id: str,
    revision: str,
    layer: int,
    concept_words: list[str],
    n_prompts: int = 64,
    max_len: int = 128,
    batch_size: int = 4,
):
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.lens import LensSpec, lens_vectors

    out_path = Path("/out") / f"lens_{revision.replace('/', '_')}.pt"
    if out_path.exists():
        print(f"[skip] {revision} already computed")
        return revision

    prompts = [
        ln.strip()
        for ln in Path("/root/data/prompts.txt").read_text().splitlines()
        if ln.strip()
    ][:n_prompts]

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    token_ids = [tok.encode(w, add_special_tokens=False)[0] for w in concept_words]

    print(f"[load] {model_id} @ {revision}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, cache_dir=CACHE
    ).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    volume.commit()

    def batches():
        for i in range(0, len(prompts), batch_size):
            enc = tok(
                prompts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    spec = LensSpec(
        layer=layer, token_ids=token_ids, n_prompts=len(prompts), max_len=max_len
    )
    block = lens_vectors(model, batches(), spec)

    torch.save(
        {
            "revision": revision,
            "model": model_id,
            "layer": layer,
            "concept_words": concept_words,
            "token_ids": token_ids,
            "n_prompts": len(prompts),
            "lens": block,
        },
        out_path,
    )
    results.commit()
    print(f"[save] {out_path}  shape={tuple(block.shape)}")
    return revision


DEFAULT_CONCEPTS = [
    " safe", " harmful", " refuse", " help", " user", " assistant",
    " danger", " honest", " lie", " secret", " test", " evaluate",
    " code", " math", " because", " however", " uncertain", " sure",
]


@app.local_entrypoint()
def main(
    revisions: str,
    layer: int,
    model_id: str = "allenai/Olmo-3-1025-7B",
    n_prompts: int = 64,
):
    revs = [r.strip() for r in revisions.split(",") if r.strip()]
    print(f"{len(revs)} revisions on {model_id}, layer {layer}")

    # starmap runs them in parallel; each container pulls its own checkpoint.
    # Drop to a serial loop if you hit the 10-GPU concurrency cap on the free tier.
    args = [
        (model_id, r, layer, DEFAULT_CONCEPTS, n_prompts) for r in revs
    ]
    for done in lens_at_revision.starmap(args):
        print(f"[done] {done}")
