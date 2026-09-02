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
    n_prompts: int = 25, max_len: int = 128, prompt_offset: int = 0,
    n_replicates: int = 1,
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

    tag = "" if prompt_offset == 0 else f"_p{prompt_offset}"
    out = Path("/out") / f"{probe}_L{layer}_{revision.replace('/', '_')}{tag}.pt"
    rep_out = Path("/out") / f"rep_{probe}_L{layer}_{revision.replace('/', '_')}.pt"
    if out.exists() and (n_replicates == 1 or rep_out.exists()):
        print(f"[skip] {revision}")
        return revision

    model, tok, token_ids, target = _setup(revision)
    volume.commit()
    # Replicates share ONE model load. Downloading and loading a 7B checkpoint
    # dominates the cost; the extra backward passes are cheap by comparison, so
    # k disjoint prompt subsets here is far cheaper than k separate runs.
    all_texts = _prompts(n_prompts * n_replicates + prompt_offset)[prompt_offset:]
    texts = all_texts[:n_prompts]

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

    if n_replicates > 1:
        reps = [block]
        for r in range(1, n_replicates):
            sub = all_texts[r * n_prompts : (r + 1) * n_prompts]

            def rep_batches(sub=sub):
                for t in sub:
                    enc = tok(t, return_tensors="pt", truncation=True,
                              max_length=max_len)
                    yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

            reps.append(lens_vectors(model, rep_batches(), spec, seeds))
        torch.save({"revision": revision, "layer": layer, "probe": probe,
                    "n_prompts": n_prompts, "replicates": torch.stack(reps)},
                   rep_out)
        print(f"[reps] {rep_out} n={n_replicates}")

    results.commit()
    print(f"[save] {out} shape={tuple(block.shape)}")
    return revision


@app.local_entrypoint()
def main(
    layer: int = 20, probe: str = "random", revisions: str = "",
    prompt_offset: int = 0, n_replicates: int = 1,
):
    revs = [r.strip() for r in revisions.split(",") if r.strip()] or REVISIONS
    print(f"layer {layer}, probe={probe}, offset={prompt_offset}, {len(revs)} revs")
    args = [(r, layer, probe, 25, 128, prompt_offset, n_replicates) for r in revs]
    for done in lens_at_revision.starmap(args):
        print(f"[done] {done}")


READOUT_PROMPTS = [
    # factual / entity
    "The Eiffel Tower is located in the city of Paris, the capital of France.",
    # arithmetic -- does the workspace hold the intermediate?
    "If you buy 7 apples at 3 dollars each, the total cost comes to",
    # code
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n *",
    # refusal / safety-adjacent
    "User: How do I pick a lock on my neighbour's door?\nAssistant: I can't help",
    # multi-step reasoning
    "Alice is taller than Bob. Bob is taller than Carol. So the shortest person is",
]
READOUT_PROMPT = READOUT_PROMPTS[0]


@app.function(
    gpu="A100-40GB",
    volumes={CACHE: volume, "/out": results},
    timeout=90 * 60,
    scaledown_window=60,
)
def readout_at_revision(revision: str, layer: int, n_prompts: int = 25, topk: int = 8):
    """What does the lens actually SAY at each position, at this checkpoint?

    Needs the full d_model x d_model J, not the 32-probe sketch: a readout scores
    the whole vocabulary, and score(token k) = <row k of W_U J, h>. Batched
    cotangents make that affordable -- ~32 batched backward passes per prompt
    instead of 4096 individual ones.
    """
    from pathlib import Path

    import torch

    from jlens.lens import LensSpec, _ResidualCapture, full_jacobian, readout

    out = Path("/out") / f"readout_L{layer}_{revision.replace('/', '_')}.pt"
    if out.exists():
        print(f"[skip] {revision}")
        return revision

    model, tok, _, target = _setup(revision)
    volume.commit()
    texts = _prompts(n_prompts)

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    spec = LensSpec(layer=layer, target_layer=target, n_prompts=n_prompts, max_len=128)
    J = full_jacobian(model, batches(), spec)
    print(f"[J] {revision} shape={tuple(J.shape)} ||J||_F={J.norm():.2f} "
          f"diag_mean={J.diag().mean():.4f}")

    # Now read out a held-out prompt, position by position.
    enc = tok(READOUT_PROMPT, return_tensors="pt")
    ids = enc["input_ids"].cuda()
    with _ResidualCapture(model, layer, target) as cap:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        h = cap.h_l.detach()[0]                              # (T, d_model)

    vals, idxs = readout(model, J, h, k=topk)
    rows = []
    for t in range(h.shape[0]):
        rows.append({
            "pos": t,
            "token": tok.decode(ids[0, t]),
            "top": [(tok.decode(i), round(float(v), 4))
                    for v, i in zip(vals[t], idxs[t])],
        })

    # Save J itself, not just what we happened to compute from it. It is the
    # expensive artifact -- the batched backward passes are essentially the whole
    # cost, and everything else here is seconds of arithmetic on top. 67 MB in
    # float32, which is nothing against a 1 TiB free tier.
    torch.save({"revision": revision, "layer": layer, "target_layer": target,
                "J": J.to(torch.float32)}, Path("/out") / f"J_L{layer}_{revision}.pt")
    torch.save(
        {"revision": revision, "layer": layer, "target_layer": target,
         "prompt": READOUT_PROMPT, "rows": rows,
         "J_fro": float(J.norm()), "J_diag_mean": float(J.diag().mean())},
        out,
    )
    results.commit()
    print(f"[save] {out}")
    return revision


@app.local_entrypoint()
def readouts(layer: int = 20, revisions: str = "stage1-step0,stage1-step512000,main"):
    revs = [r.strip() for r in revisions.split(",") if r.strip()]
    for done in readout_at_revision.starmap([(r, layer) for r in revs]):
        print(f"[done] {done}")


@app.function(
    gpu="A100-40GB",
    volumes={CACHE: volume, "/out": results},
    timeout=120 * 60,
    scaledown_window=60,
)
def layer_sweep(revision: str, n_prompts: int = 25, stride: int = 2):
    """J at EVERY layer for one checkpoint, plus a readout grid for one prompt.

    One backward pass flows through all layers, so this costs about the same as
    a single-layer Jacobian. That is why the published lens files ship every
    source layer together -- something I should have noticed sooner.
    """
    from pathlib import Path

    import torch

    from jlens.lens import _MultiCapture, jacobians_all_layers, readout

    out = Path("/out") / f"layers_{revision.replace('/', '_')}.pt"
    if out.exists():
        print(f"[skip] {revision}")
        return revision

    model, tok, _, target = _setup(revision)
    volume.commit()
    texts = _prompts(n_prompts)
    layers = list(range(0, target + 1, stride))

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    Js = jacobians_all_layers(model, batches(), layers, target)
    print(f"[J] {revision}: {len(Js)} layers")
    for l in layers[::4]:
        print(f"    layer {l:>2}: ||J||={Js[l].norm():7.2f}  diag={Js[l].diag().mean():.4f}")

    # Read out every prompt with the same J -- the lens is fitted once and reused,
    # so multiple prompts cost only forward passes.
    per_prompt = []
    for prompt in READOUT_PROMPTS:
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        with _MultiCapture(model, layers, target) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=False)
            acts = {l: cap.h[l].detach()[0] for l in layers}

        grid = {}
        for l in layers:
            vals, idxs = readout(model, Js[l], acts[l], k=5)
            grid[l] = [[(tok.decode(i), float(v)) for v, i in zip(vals[t], idxs[t])]
                       for t in range(acts[l].shape[0])]
        per_prompt.append({
            "prompt": prompt,
            "tokens": [tok.decode(ids[0, t]) for t in range(ids.shape[1])],
            "grid": grid,
        })
        print(f"[readout] {prompt[:45]!r} ({ids.shape[1]} tokens)")

    torch.save({"revision": revision, "target_layer": target, "layers": layers,
                "prompts": per_prompt,
                "prompt": per_prompt[0]["prompt"],
                "tokens": per_prompt[0]["tokens"],
                "grid": per_prompt[0]["grid"],
                "J_stats": {l: {"fro": float(Js[l].norm()),
                                "diag": float(Js[l].diag().mean())} for l in layers}},
               out)
    torch.save({"revision": revision, "layers": layers,
                "J": {l: Js[l] for l in layers}},
               Path("/out") / f"Jall_{revision.replace('/', '_')}.pt")
    results.commit()
    print(f"[save] {out}")
    return revision


@app.local_entrypoint()
def layers(revisions: str = "main", stride: int = 2):
    revs = [r.strip() for r in revisions.split(",") if r.strip()]
    for done in layer_sweep.starmap([(r, 25, stride) for r in revs]):
        print(f"[done] {done}")


EM_REPO = "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_{cond}"
EM_CONDS = ["general_medical", "general_finance", "general_sport",
            "narrow_medical", "narrow_finance", "narrow_sport"]
EM_MODEL = "Qwen/Qwen2.5-14B-Instruct"


@app.function(
    gpu="A100-80GB",              # 14B bf16 is 28GB before the batched backward graph
    volumes={CACHE: volume, "/out": results},
    timeout=180 * 60,
    scaledown_window=60,
)
def em_direction_readout(n_prompts: int = 25, stride: int = 4, chunk: int = 64):
    """Does the misalignment direction have a readable signature in token space?

    The go/no-go for the whole EM pivot. If W_U J d_EM is indistinguishable from
    W_U J (random direction), the lens is structurally blind to misalignment and
    no amount of careful measurement helps -- and we learn that in an hour
    instead of after a week of fine-tuning.

    Two controls, both required:
      * random directions matched in norm -- every direction produces SOME
        tokens, so "it produced tokens" on its own proves nothing.
      * narrow vs general -- Soligo & Turner found general misalignment
        converges to a shared direction while narrow ones do not. If the lens
        sees real structure, the three general_* conditions should read out
        alike and the narrow_* ones should not. That is a prediction with a
        right answer we did not choose.
    """
    from pathlib import Path

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = Path("/out") / "em_readout.pt"
    if out.exists():
        print("[skip] already computed")
        return "skip"

    from jlens.lens import jacobians_all_layers, random_seeds

    cfg = AutoConfig.from_pretrained(EM_MODEL, cache_dir=CACHE)
    tc = getattr(cfg, "text_config", cfg)
    n_layers, target = tc.num_hidden_layers, tc.num_hidden_layers - 2
    tok = AutoTokenizer.from_pretrained(EM_MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        EM_MODEL, dtype=torch.bfloat16, cache_dir=CACHE
    ).cuda()
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    volume.commit()
    print(f"{EM_MODEL}: {n_layers} layers, target {target}")

    texts = _prompts(n_prompts)

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    layers = sorted(set(list(range(0, target + 1, stride)) + [24]))   # 24 = sv layer
    Js = jacobians_all_layers(model, batches(), layers, target, chunk=chunk)
    print(f"[J] {len(Js)} layers computed")

    _, norm = _find_blocks_and_norm_shim(model)
    W_U = model.get_output_embeddings().weight

    def read(vec, layer, k=15):
        v = vec.cuda().to(W_U.dtype)
        t = Js[layer].cuda().to(W_U.dtype) @ v
        with torch.no_grad():
            logits = norm(t) @ W_U.T
            probs = torch.softmax(logits.float(), dim=-1)
        vals, idxs = probs.topk(k)
        ent = float(-(probs * probs.clamp(min=1e-12).log()).sum())
        return ([(tok.decode(i), float(x)) for x, i in zip(vals, idxs)], ent,
                t.float().cpu())

    results_d = {"layers": layers, "target": target, "model": EM_MODEL,
                 "conds": {}, "random": {}}

    # Controls first, so the baseline exists before we look at anything real.
    g = torch.Generator().manual_seed(0)
    for l in layers:
        rnd = []
        for i in range(8):
            v = torch.randn(tc.hidden_size, generator=g)
            v = v / v.norm() * 0.2159            # match the final sv norm
            top, ent, _ = read(v, l)
            rnd.append({"top": top, "entropy": ent})
        results_d["random"][l] = rnd
    print("[ctrl] random directions done")

    for cond in EM_CONDS:
        repo = EM_REPO.format(cond=cond)
        files = [f for f in list_repo_files(repo) if f.endswith("steering_vector.pt")]
        steps = sorted(
            [(int(f.split("checkpoint-")[1].split("/")[0]), f)
             for f in files if "checkpoint-" in f]
        )
        keep = [steps[i] for i in range(0, len(steps), max(1, len(steps) // 20))]
        keep = keep + [(10**6, "steering_vector.pt")]
        entry = []
        for step, fn in keep:
            d = torch.load(hf_hub_download(repo, filename=fn, cache_dir=CACHE),
                           map_location="cpu", weights_only=False)
            vec, l_idx = d["steering_vector"].float(), int(d["layer_idx"])
            top, ent, transported = read(vec, l_idx if l_idx in Js else 24)
            entry.append({"step": step, "norm": float(vec.norm()),
                          "layer_idx": l_idx, "top": top, "entropy": ent,
                          "transported": transported, "raw": vec})
        results_d["conds"][cond] = entry
        print(f"[em] {cond}: {len(entry)} checkpoints, "
              f"final top-3 {[w for w,_ in entry[-1]['top'][:3]]}")

    torch.save(results_d, out)
    results.commit()
    print(f"[save] {out}")
    return "ok"


def _find_blocks_and_norm_shim(model):
    from jlens.lens import _find_blocks_and_norm
    return _find_blocks_and_norm(model)


@app.local_entrypoint()
def em(stride: int = 4):
    print(em_direction_readout.remote(25, stride))
