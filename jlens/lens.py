"""Partial J-Lens.

The full J-Lens at layer l is

    J_l = E_{t, t' >= t, prompt} [ d h_final[t'] / d h_l[t] ]        (d_model x d_model)

and the lens vectors are the rows of  W_U @ J_l  -- one row per vocabulary token.

Materialising J_l costs d_model backward passes per prompt (2560+ for a 7B model),
which is why nobody does it casually. We never need the whole thing. Row k is

    (W_U @ J_l)[k]  =  J_l^T @ W_U[k]

i.e. one vector-Jacobian product seeded with token k's unembedding row. For K
tokens of interest that is K backward passes per prompt instead of d_model.
Pick K ~ 50 concept tokens and the whole thing runs in minutes.

Three details that are easy to get wrong and are load-bearing:

1. "Layer l" means the residual LEAVING block l. The reference implementation
   registers a forward hook and stores the block's output; capturing the input
   instead silently computes the neighbouring layer's Jacobian. Nor is the
   destination the last layer: the published qwen3.5-4b lens uses
   target_layer = 30 of 32 blocks, and J at the target is exactly the identity
   (verified: max|J[30] - I| == 0).

2. Causality does the t' >= t sum for us. Seed *every* destination position with
   the same v and take one backward pass; the gradient landing at anchor
   position t is already sum_{t' >= t} J[t,t']^T v, because the causal mask has
   zeroed the terms with t' < t. We do not need a loop over t'.

3. Valid source positions are [skip_first, len - 1), and the reduction is a
   per-prompt mean. Early positions are dominated by attention-sink behaviour,
   and the last position has no next-token target -- it is also the only anchor
   whose t' >= t sum has a single term, so including it injects a spurious
   near-identity contribution. Each prompt is averaged over its own valid
   positions first and only then across prompts, so long prompts do not get
   more weight. (Both of these were wrong in the first version here, and the
   symptom was a lens that matched the published one at cosine 0.94 -- close
   enough to look fine, far enough to be a different object.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LensSpec:
    """What to compute.

    Defaults follow the provenance recorded in the published lenses
    (camilablank/workspace-lenses, qwen3.5-4b/j-lens/lens.pt):

        target_layer=30, t_max=128, skip_first=4, weighting='uniform',
        n_prompts=25, corpus=NeelNanda/pile-10k, estimator='standard'

    Do not change these while verifying against the published artifact -- the
    whole point of that run is that everything else is held equal.
    """

    layer: int                  # which residual-stream layer to anchor at
    target_layer: int           # J maps h_layer -> h_target_layer (J is I at target)
    n_prompts: int = 25         # how many prompts to average the expectation over
    max_len: int = 128          # truncate prompts to this many tokens
    skip_first: int = 4         # ignore the first few positions; BOS/warmup are atypical
    weighting: str = "uniform"  # 'uniform' | 'per_anchor'


def _find_blocks_and_norm(model):
    """Locate the decoder block list and the final norm module.

    Kept deliberately noisy: if the architecture does not match what we expect,
    we want a loud failure here rather than a quiet wrong number later.
    """
    inner = getattr(model, "model", model)
    blocks = getattr(inner, "layers", None)
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
    if blocks is None or norm is None:
        raise RuntimeError(
            f"Could not locate decoder layers / final norm on {type(model).__name__}. "
            "Inspect the model and set them explicitly before running anything."
        )
    return blocks, norm


class _ResidualCapture:
    """Capture the residual stream at the OUTPUT of blocks `layer` and `target_layer`.

    Layer index convention, which is the thing to get right: the reference
    implementation (anthropics/jacobian-lens, jlens/hooks.py) registers a
    *forward* hook and stores `output`, so "layer l" means the residual leaving
    block l -- not the residual entering it. Capturing inputs instead computes
    the Jacobian for the neighbouring layer, which is a subtle enough error that
    it shows up only as a cosine of ~0.94 against the published lens.

    Marking the source tensor `requires_grad_(True)` roots the autograd graph
    there. With every parameter frozen, that block output has no grad history
    and so really is a leaf; without this, a fully frozen model builds no graph
    at all and torch.autograd.grad raises. It also means only blocks from the
    source onward are taped, since earlier ones cannot contribute to dh/dh.
    """

    def __init__(self, model, layer: int, target_layer: int):
        blocks, _ = _find_blocks_and_norm(model)
        # layer == target_layer is allowed on purpose: it makes J the identity,
        # which is the basis of the architecture-independent self-test in
        # tests/test_identity_at_target.py.
        if not 0 <= layer <= target_layer < len(blocks):
            raise ValueError(
                f"need 0 <= layer <= target_layer < n_blocks; got layer={layer}, "
                f"target_layer={target_layer}, n_blocks={len(blocks)}"
            )
        self.h_l = None
        self.h_final = None
        self._handles = [
            blocks[layer].register_forward_hook(self._make(layer, root=True)),
            blocks[target_layer].register_forward_hook(self._make(target_layer)),
        ]

    def _make(self, index: int, root: bool = False):
        def hook(module, inputs, output):
            # HF blocks sometimes return (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if root:
                tensor.requires_grad_(True)
                self.h_l = tensor
            else:
                self.h_final = tensor

        return hook

    def close(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def lens_vectors(model, batches, spec: LensSpec, seeds: torch.Tensor) -> torch.Tensor:
    """Compute  J_l^T @ seeds  -- one transported row per seed vector.

    `seeds` is (K, d_model). Two things you might pass:

      * `token_seeds(...)`: rows of W_U, giving the lens vectors for particular
        vocabulary tokens. Use a FIXED W_U across checkpoints -- W_U is trained
        too, so per-checkpoint rows would conflate unembedding drift with the
        transport drift we are actually measuring.
      * `random_seeds(...)`: random unit vectors, which measure how J rotates as
        an operator without committing to any choice of concept. Better for the
        "did the transport move" question; token seeds are for interpreting what
        a movement means once you know it happened.

    `batches` is an iterable of (input_ids, attention_mask) already on device.
    Returns a (K, d_model) float32 tensor on CPU.
    """
    seeds = seeds.detach()
    d_model = seeds.shape[1]

    total = torch.zeros(seeds.shape[0], d_model, dtype=torch.float32)
    # n_seen counts PROMPTS, not positions: the reduction is a per-prompt mean
    # over source positions, then a mean over prompts.
    n_seen = 0

    for input_ids, attention_mask in batches:
        with _ResidualCapture(model, spec.layer, spec.target_layer) as cap:
            with torch.enable_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            h_l, h_final = cap.h_l, cap.h_final

        if h_l is None or h_final is None:
            raise RuntimeError("Hooks did not fire -- check the module paths.")

        B, T, _ = h_final.shape
        idx = torch.arange(T, device=h_final.device)

        if spec.weighting == "uniform":
            # Each source position contributes the raw sum over t' >= t, and we
            # then take the mean over source positions. This is the reduction
            # the reference implementation calls 'standard'.
            weight = torch.ones(T, dtype=h_final.dtype, device=h_final.device)
        elif spec.weighting == "per_anchor":
            # Not the paper's estimator, kept only so the difference can be
            # measured: divides each anchor by its own number of targets.
            weight = (1.0 / (T - idx).clamp(min=1)).to(h_final.dtype)
        else:
            raise ValueError(f"unknown weighting {spec.weighting!r}")

        # Valid source positions are [skip_first, len - 1) per sequence. Early
        # positions are dominated by attention-sink behaviour, and the FINAL
        # position is excluded because it has no next-token target -- it is also
        # the one anchor whose t' >= t sum has a single term, so leaving it in
        # injects a spurious near-identity contribution.
        lengths = attention_mask.sum(dim=1, keepdim=True)              # (B, 1)
        pos = idx.unsqueeze(0)                                          # (1, T)
        valid = (pos >= spec.skip_first) & (pos < lengths - 1) & attention_mask.bool()

        for k, v in enumerate(seeds):
            # Seed every final position with v. The causal mask means the
            # gradient arriving at anchor t is the sum over t' >= t only.
            grad_out = v.view(1, 1, -1).expand(B, T, d_model).to(h_final.dtype)

            (g,) = torch.autograd.grad(
                outputs=h_final,
                inputs=h_l,
                grad_outputs=grad_out,
                retain_graph=True,     # reused across the K seeds
            )
            # g: (B, T, d_model) -- g[b, t] = sum_{t' >= t} J[b, t, t']^T v
            g = g * weight.view(1, T, 1) * valid.unsqueeze(-1).to(g.dtype)

            # Mean over source positions WITHIN each prompt, then sum prompts.
            # Normalising once over pooled positions instead would weight long
            # prompts more heavily; the reference divides per prompt and then by
            # the prompt count.
            counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)        # (B, 1)
            per_prompt = g.sum(dim=1) / counts.to(g.dtype)              # (B, d_model)
            total[k] += per_prompt.sum(dim=0).float().cpu()

        n_seen += B

        del h_l, h_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total / max(n_seen, 1)   # divide by prompt count


def readout(lens_block: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Score each of the K tokens for an activation h.

    This is the linearised readout: (W_U J_l)[k] . h. The paper's full readout
    is softmax(W_U norm(J_l h_l)), which applies the model's final norm to the
    transported vector first. The linear version is what you want for *diffing*
    lenses across checkpoints -- the norm is a per-activation rescale and would
    muddy a comparison between two lenses. Use the full readout when you want an
    actual token distribution to eyeball.
    """
    return lens_block.to(h.dtype) @ h


def cosine_drift(lens_a: torch.Tensor, lens_b: torch.Tensor) -> torch.Tensor:
    """Per-token cosine similarity between two lens blocks.

    The primary quantity: how far each lens vector has rotated between two
    checkpoints. Returns a (K,) tensor; near 1 means that concept's lens did not
    move. Interpretable only against the random-data control, since any weight
    update perturbs a prompt-averaged estimator somewhat.
    """
    a = torch.nn.functional.normalize(lens_a.float(), dim=-1)
    b = torch.nn.functional.normalize(lens_b.float(), dim=-1)
    return (a * b).sum(dim=-1)


def token_seeds(model, token_ids) -> torch.Tensor:
    """Rows of W_U for the given tokens -- the classic J-Lens vectors.

    Across a checkpoint sweep, take these from ONE reference checkpoint and
    reuse them everywhere. W_U is trained, so recomputing per checkpoint mixes
    unembedding drift into a measurement meant to be about J alone.
    """
    return model.get_output_embeddings().weight[token_ids].detach()


def random_seeds(d_model: int, k: int, *, seed: int = 0, device="cpu",
                 dtype=torch.float32) -> torch.Tensor:
    """K random unit vectors, for measuring how J rotates as an operator.

    No concept choice to justify, and it probes J in general directions rather
    than the handful a person happened to pick. Fixed `seed` so the same probe
    set is reused at every checkpoint -- otherwise the probes themselves become
    a source of drift.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(k, d_model, generator=g)
    return torch.nn.functional.normalize(v, dim=-1).to(device=device, dtype=dtype)


def full_jacobian(model, batches, spec: LensSpec, *, chunk: int = 128) -> torch.Tensor:
    """Materialise the whole (d_model x d_model) J_l.

    The partial path takes K backward passes for K probes. Here we want every
    row, because a readout needs scores for the entire vocabulary and
    score(token k) = <row k of W_U J, h>. That is d_model seeds -- 4096 for a 7B
    -- which is only affordable because torch can batch cotangents: one call
    with `is_grads_batched=True` runs `chunk` of them at once via vmap, turning
    ~4096 backward passes per prompt into ~32 batched ones.

    Seeding with the basis vector e_k gives row k of J, since (J^T e_k)_j = J_kj.
    """
    d_model = model.get_output_embeddings().weight.shape[1]
    total = torch.zeros(d_model, d_model, dtype=torch.float32)
    n_seen = 0

    for input_ids, attention_mask in batches:
        with _ResidualCapture(model, spec.layer, spec.target_layer) as cap:
            with torch.enable_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            h_l, h_final = cap.h_l, cap.h_final
        if h_l is None or h_final is None:
            raise RuntimeError("Hooks did not fire -- check the module paths.")

        B, T, _ = h_final.shape
        idx = torch.arange(T, device=h_final.device)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        valid = (idx.unsqueeze(0) >= spec.skip_first) & (idx.unsqueeze(0) < lengths - 1)
        valid = valid & attention_mask.bool()
        counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)

        eye = torch.eye(d_model, device=h_final.device, dtype=h_final.dtype)
        for s in range(0, d_model, chunk):
            block = eye[s : s + chunk]                       # (C, d_model)
            C = block.shape[0]
            grad_out = block.view(C, 1, 1, d_model).expand(C, B, T, d_model)
            (g,) = torch.autograd.grad(
                outputs=h_final, inputs=h_l, grad_outputs=grad_out,
                retain_graph=True, is_grads_batched=True,
            )                                                 # (C, B, T, d_model)
            g = g * valid.unsqueeze(0).unsqueeze(-1).to(g.dtype)
            per_prompt = g.sum(dim=2) / counts.unsqueeze(0).to(g.dtype)   # (C,B,d_model)
            total[s : s + C] += per_prompt.sum(dim=1).float().cpu()

        n_seen += B
        del h_l, h_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total / max(n_seen, 1)


def readout(model, J: torch.Tensor, h: torch.Tensor, k: int = 8):
    """The J-Lens readout: softmax(W_U norm(J h)) -> top-k tokens.

    `h` is (..., d_model) taken at the SAME point J was anchored (the output of
    block spec.layer). The norm is the model's own final norm, applied to the
    transported vector, exactly as in the paper's formula.
    """
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight

    dev = W_U.device
    transported = h.to(dev, W_U.dtype) @ J.to(dev, W_U.dtype).T   # (..., d_model)
    with torch.no_grad():
        logits = norm(transported) @ W_U.T
        probs = torch.softmax(logits.float(), dim=-1)
    return probs.topk(k, dim=-1)


class _MultiCapture:
    """Capture the residual at MANY source layers plus the target, in one pass.

    The optimisation that makes layer-by-layer work affordable: one backward
    pass from the target already flows through every earlier layer, so a single
    seeded cotangent yields J_l^T v for ALL l at once. Only the earliest layer is
    made a graph root; the rest are ordinary non-leaf tensors in the graph, and
    torch.autograd.grad differentiates with respect to all of them together.

    So computing J at 30 layers costs about the same as computing it at one --
    which is why the published lens files ship every source layer in one blob.
    """

    def __init__(self, model, layers, target_layer: int):
        blocks, _ = _find_blocks_and_norm(model)
        self.layers = sorted(set(layers))
        if not (0 <= self.layers[0] and self.layers[-1] <= target_layer < len(blocks)):
            raise ValueError(f"bad layer range {self.layers[:3]}..{self.layers[-1]} "
                             f"vs target {target_layer}, n_blocks {len(blocks)}")
        self.h = {}
        self.h_target = None
        root = self.layers[0]
        self._handles = [
            blocks[l].register_forward_hook(self._make(l, root=(l == root)))
            for l in self.layers
        ]
        if target_layer not in self.layers:
            self._handles.append(
                blocks[target_layer].register_forward_hook(self._make_target())
            )

    def _make(self, index, root=False):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            if root:
                t.requires_grad_(True)
            self.h[index] = t
        return hook

    def _make_target(self):
        def hook(module, inputs, output):
            self.h_target = output if torch.is_tensor(output) else output[0]
        return hook

    def target(self, target_layer):
        return self.h_target if self.h_target is not None else self.h[target_layer]

    def close(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def jacobians_all_layers(model, batches, layers, target_layer: int, *,
                         skip_first: int = 4, chunk: int = 128) -> dict:
    """Full J_l for every layer in `layers`, for the price of roughly one."""
    d_model = model.get_output_embeddings().weight.shape[1]
    layers = sorted(set(layers))
    total = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in layers}
    n_seen = 0

    for input_ids, attention_mask in batches:
        with _MultiCapture(model, layers, target_layer) as cap:
            with torch.enable_grad():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False)
            hs = [cap.h[l] for l in layers]
            h_t = cap.target(target_layer)

        B, T, _ = h_t.shape
        idx = torch.arange(T, device=h_t.device)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        valid = ((idx.unsqueeze(0) >= skip_first) & (idx.unsqueeze(0) < lengths - 1)
                 & attention_mask.bool())
        counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)

        eye = torch.eye(d_model, device=h_t.device, dtype=h_t.dtype)
        for s in range(0, d_model, chunk):
            blk = eye[s : s + chunk]
            C = blk.shape[0]
            go = blk.view(C, 1, 1, d_model).expand(C, B, T, d_model)
            grads = torch.autograd.grad(
                outputs=h_t, inputs=hs, grad_outputs=go,
                retain_graph=True, is_grads_batched=True, allow_unused=True,
            )
            for l, g in zip(layers, grads):
                if g is None:       # layer at/after the target contributes nothing
                    continue
                g = g * valid.unsqueeze(0).unsqueeze(-1).to(g.dtype)
                per = g.sum(dim=2) / counts.unsqueeze(0).to(g.dtype)
                total[l][s : s + C] += per.sum(dim=1).float().cpu()

        n_seen += B
        del hs, h_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {l: total[l] / max(n_seen, 1) for l in layers}
