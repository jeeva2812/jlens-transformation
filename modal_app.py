"""Sweep J-Lens across Olmo 3 training checkpoints, on Modal.

    modal run modal_app.py::preflight     # identity self-test first
    modal run modal_app.py                # then the sweep

Checkpoints are cached in a Modal Volume rather than re-downloaded: volume
storage is $0.09/GiB/month with the first 1 TiB free, and eight 7B checkpoints
is ~112 GiB, so caching costs nothing while re-downloading costs GPU-seconds.

No published lens exists for Olmo 3, so there is nothing external to check
against. `preflight` is the substitute: at layer == target_layer the Jacobian is
the identity by construction, so the lens block must come back as exactly
W_U[token_ids]. That catches wrong hook placement, tuple-vs-tensor block
outputs, a mask that zeroes everything, and reduction errors -- with no
reference artifact required. It passes exactly (1.00000) on Qwen3.5-4B, where
step 0 independently confirmed the whole pipeline against the published lens.
"""

from __future__ import annotations

import modal

CACHE = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers==5.16.1",   # pinned to the version step 0 was verified on
        "accelerate",
        "numpy",
        "datasets",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE})
    .add_local_python_source("jlens")
)

volume = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("jlens-results", create_if_missing=True)

app = modal.App("jlens-transformation", image=image)

MODEL = "allenai/Olmo-3-1025-7B"

# Token probes are built from ONE checkpoint's unembedding and reused at every
# other, so that drift in W_U cannot masquerade as drift in J.
REF_REVISION = "main"

# Log-spaced across the full 1.41M-step stage-1 trajectory. Representational
# change is fastest early, so even spacing would spend most of the budget on the
# flat tail.
REVISIONS = [
    "stage1-step0",
    "stage1-step2000",
    "stage1-step8000",
    "stage1-step32000",
    "stage1-step128000",
    "stage1-step512000",
    "stage1-step1413814",
    "main",
]

# Placeholders. Which tokens get tracked is a research decision and belongs in
# the write-up with a reason, not left as whatever was convenient.
CONCEPTS = [
    " Paris", " France", " water", " code", " because", " however",
    " safe", " harmful", " true", " false", " one", " two",
    " user", " help", " refuse", " secret", " danger", " honest",
]


def _setup(revision: str):
    """Load model and tokenizer at a revision; derive the target layer."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL, revision=revision, cache_dir=CACHE)
    tc = getattr(cfg, "text_config", cfg)
    target = tc.num_hidden_layers - 2      # the published lenses' convention

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, dtype=torch.bfloat16, cache_dir=CACHE
    ).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    token_ids = [tok.encode(w, add_special_tokens=False)[0] for w in CONCEPTS]
    return model, tok, token_ids, target


def _prompts(n: int):
    """Same corpus the published lenses used, so the setup stays comparable.

    Note Olmo trained on Dolma, not the Pile. That is deliberate: the lens is
    meant to capture a general disposition, and holding the probe corpus fixed
    across checkpoints is what makes the comparison mean anything.
    """
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in range(n)]


def _reference_unembed(token_ids):
    """W_U rows from REF_REVISION, cached on the volume so every worker agrees."""
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM

    cached = Path(CACHE) / "ref_unembed.pt"
    if cached.exists():
        return torch.load(cached, map_location="cpu", weights_only=True)

    m = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REF_REVISION, dtype=torch.bfloat16, cache_dir=CACHE
    )
    rows = m.get_output_embeddings().weight[token_ids].detach().float().cpu()
    torch.save(rows, cached)
    volume.commit()
    del m
    return rows


@app.function(gpu="A100-40GB", volumes={CACHE: volume}, timeout=45 * 60)
def preflight():
    """At layer == target, J is the identity, so the lens must equal W_U rows."""
    import torch

    from jlens.lens import LensSpec, lens_vectors, token_seeds

    model, tok, token_ids, target = _setup("main")
    texts = _prompts(4)

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    spec = LensSpec(layer=target, target_layer=target, n_prompts=len(texts), max_len=128)
    got = lens_vectors(model, batches(), spec, token_seeds(model, token_ids)).float()
    want = model.get_output_embeddings().weight[token_ids].detach().float().cpu()

    cos = torch.nn.functional.cosine_similarity(got, want, dim=-1).mean().item()
    ratio = (got.norm(dim=-1) / want.norm(dim=-1)).mean().item()
    print(f"identity check on {MODEL}: cosine={cos:.5f} ratio={ratio:.5f}")

    ok = cos > 0.9999 and abs(ratio - 1) < 1e-3
    print("PASS -- plumbing is right on this architecture" if ok else
          "FAIL -- do not trust any lens computed from this model")
    return ok


@app.function(
    gpu="A100-40GB",
    volumes={CACHE: volume, "/out": results},
    timeout=60 * 60,
    scaledown_window=60,        # never sit idle holding a GPU
)
def lens_at_revision(
    revision: str, layer: int, probe: str = "random",
    n_prompts: int = 25, max_len: int = 128,
):
    """Compute one checkpoint's transported rows.

    probe='random' measures how J rotates as an operator -- no concept choice to
    justify, and it probes general directions rather than a hand-picked handful.
    probe='token' uses W_U rows, and deliberately takes them from a SINGLE
    reference checkpoint (REF_REVISION) rather than the current one: W_U is
    trained too, so per-checkpoint rows would mix unembedding drift into a
    measurement that is supposed to be about the transport alone.
    """
    from pathlib import Path

    import torch

    from jlens.lens import LensSpec, lens_vectors, random_seeds, token_seeds

    out = Path("/out") / f"{probe}_L{layer}_{revision.replace('/', '_')}.pt"
    if out.exists():
        print(f"[skip] {revision}")
        return revision

    model, tok, token_ids, target = _setup(revision)
    volume.commit()
    texts = _prompts(n_prompts)

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    if probe == "random":
        # d_model straight off the unembedding: config field names are nested
        # differently across these architectures and have already bitten us once.
        d_model = model.get_output_embeddings().weight.shape[1]
        seeds = random_seeds(d_model, 32, seed=0, device="cuda", dtype=torch.bfloat16)
        labels = [f"r{i}" for i in range(seeds.shape[0])]
    elif probe == "token":
        seeds = _reference_unembed(token_ids).cuda().to(torch.bfloat16)
        labels = CONCEPTS
    else:
        raise ValueError(f"unknown probe {probe!r}")

    spec = LensSpec(
        layer=layer, target_layer=target, n_prompts=n_prompts, max_len=max_len,
    )
    block = lens_vectors(model, batches(), spec, seeds)

    torch.save(
        {
            "revision": revision, "model": MODEL, "layer": layer,
            "target_layer": target, "probe": probe, "labels": labels,
            "ref_revision": REF_REVISION if probe == "token" else None,
            "n_prompts": n_prompts, "max_len": max_len, "lens": block,
        },
        out,
    )
    results.commit()
    print(f"[save] {out} shape={tuple(block.shape)}")
    return revision


@app.local_entrypoint()
def main(layer: int = 20, probe: str = "random", revisions: str = ""):
    revs = [r.strip() for r in revisions.split(",") if r.strip()] or REVISIONS
    print(f"layer {layer}, probe={probe}, {len(revs)} revisions on {MODEL}")
    for done in lens_at_revision.starmap([(r, layer, probe) for r in revs]):
        print(f"[done] {done}")
