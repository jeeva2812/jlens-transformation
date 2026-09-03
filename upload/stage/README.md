---
license: apache-2.0
tags:
  - interpretability
  - jacobian-lens
  - j-lens
  - olmo
  - training-dynamics
---

# J-Lenses for Olmo 3 7B across training

Jacobian lenses (J-Lens, from Anthropic's global-workspace work) computed for
**allenai/Olmo-3-1025-7B** at **24 checkpoints spanning all three training
stages** — pretraining, midtraining, and long-context extension.

As far as I can tell no J-Lens has been published for Olmo 3, and none for any
model *across training checkpoints*. The published lenses
(`camilablank/workspace-lenses`) cover 8 models at their final weights only.

## Contents

| file | what |
|---|---|
| `lenses/random_L20_<revision>.pt` | `J₂₀ᵀ v` for 32 fixed random probe directions, one per checkpoint (32 × 4096) |
| `Jall_main.pt` | **full** Jacobians (4096 × 4096) at 16 layers for the final model |
| `layers_main.pt` | layer × position readout grids for 5 prompts, plus per-layer `‖J‖` and diagonal mean |

Probes are `torch.randn` with `manual_seed(0)`, row-normalised — the *same* 32
directions at every checkpoint, so the probes are never themselves a source of
drift. Regenerate with `jlens.lens.random_seeds(4096, 32, seed=0)`.

## Conventions — read this before comparing against anything

These cost me four wrong attempts, and **none of them produced a visible
symptom when wrong**. Every version returned plausible numbers and sensible
per-token structure; the only way I found the errors was matching against a
published lens.

1. **"Layer ℓ" is the residual LEAVING block ℓ.** The reference implementation
   (`anthropics/jacobian-lens`, `jlens/hooks.py`) registers a *forward* hook and
   stores `output`. Capturing the block's *input* computes the neighbouring
   layer's Jacobian — which matches the published lens at cosine 0.96, close
   enough to look like noise rather than a bug.
2. **The target is not the last layer.** `target_layer = n_layers − 2` (30 of 32
   here). `J` at the target is exactly the identity.
3. **Valid source positions are `[skip_first, len − 1)`**, `skip_first = 4`.
   The final position has no next-token target and is the only anchor whose
   `t' ≥ t` sum has a single term, so including it injects a spurious
   near-identity contribution.
4. **The reduction is a per-prompt mean**, then a mean over prompts — not a
   pooled mean over all positions, which would over-weight long prompts.

Fitted on 25 documents of `NeelNanda/pile-10k`, `t_max = 128`, matching the
provenance recorded in the published Qwen3.5-4B lens.

## Verification

The code producing these reproduces the published `qwen3.5-4b` J-Lens at
**cosine 0.9984, magnitude ratio 0.9985** across 12 concept tokens. Also checked:
the VJP shortcut against a brute-force Jacobian (5.4e-07), zero backwards-in-time
gradient leak, `J = I` at the target layer on both Qwen and Olmo (1.00000), and
the batched all-layer path against the single-layer path (exact).

## One thing to know before you diff these

**Raw cosine between J-rows is a bad metric.** `J` is dominated by the residual
stream's own pass-through, so a *randomly initialised* model scores 0.509
against the fully trained one. Since `Jᵀv = v + (J − I)ᵀv`, subtract the probe
first. Under that transform random init scores 0.008, as it should, and the
training trajectory becomes monotone. Diffing on raw cosine will badly
understate how much moved.

Measured noise floor: **0.9870** between two disjoint 25-prompt samples of the
same checkpoint (pure estimator variance), **0.9735** across 1814 steps of
converged training. Drift smaller than that is not interpretable.

## Code

https://github.com/jeeva2812/jlens-transformation
