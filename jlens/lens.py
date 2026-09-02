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

1. The destination is NOT the last layer. The published lenses use
   target_layer = 30 on a 31-block Qwen3.5-4B, and J at the target is exactly the
   identity (verified: max|J[30] - I| == 0). So J_l maps the residual entering
   block l to the residual entering block `target_layer`, and both are taken at
   the same reference point. Aiming at the final pre-norm residual instead gives
   a different matrix that still looks entirely plausible.

2. Causality does the t' >= t sum for us. Seed *every* destination position with
   the same v and take one backward pass; the gradient landing at anchor
   position t is already sum_{t' >= t} J[t,t']^T v, because the causal mask has
   zeroed the terms with t' < t. We do not need a loop over t'.

3. Position handling is a config choice, not an obvious default. The published
   provenance records weighting='uniform' (every (t, t') pair counts equally)
   and skip_first=4 (drop BOS and the first few positions). A per-anchor mean is
   the other defensible option and gives different numbers; `LensSpec.weighting`
   selects between them so the difference can be measured rather than assumed.
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
    token_ids: list[int]        # the K vocabulary tokens we want lens rows for
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
    """Grab the residual stream entering block `layer` and entering the final norm.

    The anchor hook does not merely observe: it detaches the incoming residual
    and re-attaches it as a leaf requiring grad, then hands that back as the
    block's input. Two consequences, both wanted:

      * The model's parameters can stay frozen. We differentiate with respect to
        an activation, not the weights, so nothing else in the forward pass needs
        requires_grad -- and without this trick a fully frozen model builds no
        autograd graph at all and torch.autograd.grad raises.
      * Only layers l..end are taped. Blocks before the anchor contribute nothing
        to dh_final/dh_l, so retaining their graph would be wasted memory.
    """

    def __init__(self, model, layer: int, target_layer: int):
        blocks, norm = _find_blocks_and_norm(model)
        if not 0 <= layer < target_layer <= len(blocks):
            raise ValueError(
                f"need 0 <= layer < target_layer <= n_blocks; got layer={layer}, "
                f"target_layer={target_layer}, n_blocks={len(blocks)}"
            )
        self.h_l = None
        self.h_final = None
        # h_target is the residual *entering* block target_layer, the same
        # reference point as h_l. That is what makes J at the target exactly I,
        # which the published lens confirms (max|J[30] - I| == 0).
        target_mod = blocks[target_layer] if target_layer < len(blocks) else norm
        self._handles = [
            blocks[layer].register_forward_pre_hook(self._anchor, with_kwargs=True),
            target_mod.register_forward_pre_hook(self._final, with_kwargs=True),
        ]

    def _anchor(self, module, args, kwargs):
        if args:
            h = args[0].detach().requires_grad_(True)
            self.h_l = h
            return (h,) + args[1:], kwargs
        if "hidden_states" in kwargs:
            h = kwargs["hidden_states"].detach().requires_grad_(True)
            self.h_l = h
            return args, {**kwargs, "hidden_states": h}
        raise RuntimeError(
            "Decoder block received no positional input and no `hidden_states` "
            "kwarg; the hook cannot find the residual stream."
        )

    def _final(self, module, args, kwargs):
        self.h_final = args[0] if args else kwargs.get("hidden_states")

    def close(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def lens_vectors(model, batches, spec: LensSpec) -> torch.Tensor:
    """Compute the K x d_model block of  W_U @ J_l  for the tokens in `spec`.

    `batches` is an iterable of (input_ids, attention_mask) already on device.
    Returns a (K, d_model) float32 tensor on CPU.
    """
    W_U = model.get_output_embeddings().weight          # (vocab, d_model)
    seeds = W_U[spec.token_ids].detach()                # (K, d_model)
    d_model = seeds.shape[1]

    total = torch.zeros(len(spec.token_ids), d_model, dtype=torch.float32)
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
            # Every (t, t') pair counts equally: take the raw sum and normalise
            # once at the end. This is what the published lenses record.
            weight = torch.ones(T, dtype=h_final.dtype, device=h_final.device)
        elif spec.weighting == "per_anchor":
            # Each anchor contributes the mean over its own downstream positions,
            # so early anchors (which have more of them) are not over-counted.
            weight = (1.0 / (T - idx).clamp(min=1)).to(h_final.dtype)
        else:
            raise ValueError(f"unknown weighting {spec.weighting!r}")

        # BOS and the first few positions are atypical enough to skew the
        # average; the published config drops them (skip_first=4).
        keep = (idx >= spec.skip_first).to(attention_mask.dtype)
        mask = (attention_mask * keep.unsqueeze(0)).unsqueeze(-1)

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
            g = g * weight.view(1, T, 1) * mask.to(g.dtype)
            total[k] += g.sum(dim=(0, 1)).float().cpu()

        n_seen += int(mask.squeeze(-1).sum().item())

        del h_l, h_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total / max(n_seen, 1)


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
