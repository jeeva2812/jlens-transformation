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

1. h_final must be the residual stream BEFORE the model's final norm. The J-Lens
   readout is softmax(W_U norm(J_l h_l)) -- the norm is applied to the
   transported vector at readout time, so J itself must map into the pre-norm
   space. HuggingFace's output_hidden_states[-1] is POST-norm (Olmo3Model.forward
   calls self.norm(hidden_states) after the decoder loop and returns that), so
   using it here silently computes the wrong Jacobian. We take the residuals off
   forward hooks instead, which does not depend on hidden_states semantics at all.

2. Causality does the t' >= t sum for us. Seed *every* final-layer position with
   the same v and take one backward pass; the gradient landing at anchor
   position t is already sum_{t' >= t} J[t,t']^T v, because the causal mask has
   zeroed the terms with t' < t. We do not need a loop over t'.

3. The number of valid t' depends on t. Anchor position t in a length-T
   sequence has (T - t) downstream positions, so a raw sum over-weights early
   positions. We divide position-wise before averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LensSpec:
    """What to compute. Keep this small and explicit -- it goes in the write-up."""

    layer: int                  # which residual-stream layer to anchor at
    token_ids: list[int]        # the K vocabulary tokens we want lens rows for
    n_prompts: int              # how many prompts to average the expectation over
    max_len: int = 128          # truncate prompts to this many tokens


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

    Both are ordinary non-leaf tensors inside the autograd graph, which is all
    torch.autograd.grad needs.
    """

    def __init__(self, model, layer: int):
        blocks, norm = _find_blocks_and_norm(model)
        self.h_l = None
        self.h_final = None
        self._handles = [
            blocks[layer].register_forward_pre_hook(self._anchor),
            norm.register_forward_pre_hook(self._final),
        ]

    def _anchor(self, module, args):
        self.h_l = args[0]

    def _final(self, module, args):
        self.h_final = args[0]

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
        with _ResidualCapture(model, spec.layer) as cap:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            h_l, h_final = cap.h_l, cap.h_final

        if h_l is None or h_final is None:
            raise RuntimeError("Hooks did not fire -- check the module paths.")

        B, T, _ = h_final.shape
        # weight[t] = 1 / (number of downstream positions t' >= t)
        idx = torch.arange(T, device=h_final.device)
        weight = (1.0 / (T - idx).clamp(min=1)).to(h_final.dtype)      # (T,)
        mask = attention_mask.unsqueeze(-1)

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

        n_seen += int(attention_mask.sum().item())

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
