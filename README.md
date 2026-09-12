# How much can we push J-Lens?

This repository is the reproducibility companion to my J-Lens investigation. It
asks two questions:

1. Does an SVD of the Jacobian lens expose usable, interpretable directions?
2. How does the lens's structure form as a language model trains?

The experiments use SmolLM2-135M for rapid iteration, the published
Qwen3.5-4B lens from
[`camilablank/workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses),
and 11 OLMo-3-7B checkpoints. Every headline result is backed by a checked-in
JSON artifact and an offline verifier.

**Start with the [interactive results overview](docs/results.html)** or run
`./.venv/bin/python verify.py` to check the quoted numbers without downloading a
model.

## Experimental setup

1. **Three model scales, one shared question.** SmolLM2-135M provides a fast
   iteration loop, Qwen3.5-4B tests the published J-Lens artifact, and 11
   OLMo-3-7B checkpoints expose how the same structure changes during training.
2. **Matched causal comparisons.** Every steering claim compares `Jᵀw` against
   the raw logit direction `w`, its sign-reversed control, and matched-norm random
   directions at the same layer, position policy, and intervention strength.
3. **Auditable outputs.** Compact JSON results are checked into `out/rare/`,
   `verify.py` checks the headline numbers offline, and the figures plus
   [`docs/results.html`](docs/results.html) connect each claim to its code and
   evidence.

## Main findings

### 1. SVD directions are causally strong, but rarely readable

For `J = U Σ Vᵀ`, the columns of `U` live in the target-layer space and can
be decoded with the unembedding; columns of `V` live in the source-layer space
and can be injected. On SmolLM2-135M, the top eight SVD directions are about
5–6× more steerable than matched PCA directions. One readable gender direction
can flip generated pronouns, but most individual directions decode to
punctuation or token fragments.

- Code: [`jlens/gender_axis.py`](jlens/gender_axis.py)
- Result: [`out/rare/gender_axis.json`](out/rare/gender_axis.json)
- Generation examples: [`out/rare/gender_generations.json`](out/rare/gender_generations.json)

### 2. The Jacobian is an efficient pullback operator

Write an objective directly in logit space:

```text
w = normalize(W_U[" Rome"] - W_U[" Paris"])
```

Then inject either `w` or `Jᵀw` at identical L2 norm. The pullback is worth
1.3–1.7× at the headline layers and beats a 30-draw matched-norm random null by
20–46 standard deviations.

| model | layer | `w` | `Jᵀw` | ratio | null mean ± sd |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 20 | +1.89 | **+3.30** | 1.74× | −0.03 ± 0.17 |
| Qwen3.5-4B | 21 | +3.72 | **+4.87** | 1.31× | +0.04 ± 0.14 |
| OLMo-3-7B | 22 | +4.07 | **+6.54** | 1.61× | +0.00 ± 0.14 |

This is a token-slot bias, not a belief edit: the induced Rome−Paris shift is
nearly independent of whether the clean model preferred Rome or Paris. A
specified target is also spread across hundreds of singular components; the
top component recovers 15% / 2% / −2% of the full effect.

- Core experiment: [`jlens/contrastive_logit_steering.py`](jlens/contrastive_logit_steering.py)
- Rank ablation: [`jlens/pullback_rank.py`](jlens/pullback_rank.py)
- Belief-vs-slot prompts: [`jlens/contrastive_fact_generation.py`](jlens/contrastive_fact_generation.py)
- Results: [`SmolLM2`](out/rare/smollm_rome_paris.json),
  [`Qwen3.5`](out/rare/qwen35_4b_rome_paris.json),
  [`OLMo-3`](out/rare/olmo3_7b_rome_paris.json)

### 3. The pullback advantage shrinks monotonically with depth

Across six contrasts and six or seven layers per model, `Jᵀw` beats `w` in
113/120 cells and beats all ten random draws in 120/120. Six of the seven
exceptions are at OLMo layer 30, the lens target layer, where `J = I` and the
ratio is forced to one.

- Code: [`jlens/contrast_sweep.py`](jlens/contrast_sweep.py)
- Results: [`SmolLM2`](out/rare/sweep_smollm2.json),
  [`Qwen3.5`](out/rare/sweep_qwen35_4b.json),
  [`OLMo-3`](out/rare/sweep_olmo3_7b.json)
- Figure: [`out/figs_final/fig4_contrast_layer_sweep.png`](out/figs_final/fig4_contrast_layer_sweep.png)

### 4. J-Lens structure forms gradually during training

The OLMo checkpoint series shows strong directions appearing before weaker
ones. Nearby layers share substantially more of their top-64 transport
subspaces than distant layers (0.50 versus 0.079 in the final model). Several
statistics break at the mid-training boundary, where the learning rate and data
mixture change abruptly; this is descriptive evidence, not a causal claim.

- Spectrum over training: [`jlens/training_spectrum_fig.py`](jlens/training_spectrum_fig.py)
- Subspace formation: [`jlens/training_fig.py`](jlens/training_fig.py)
- Cross-layer overlap: [`jlens/layer_subspaces.py`](jlens/layer_subspaces.py)
- Checkpoint trajectory: [`jlens/layer_subspaces_traj.py`](jlens/layer_subspaces_traj.py)
- Results: [`out/rare/layer_subspaces.json`](out/rare/layer_subspaces.json) and
  [`out/rare/layer_subspaces_traj.json`](out/rare/layer_subspaces_traj.json)

## Reproduce

### Fast path: verify saved results

This performs no model inference and normally finishes in about a second:

```bash
./.venv/bin/python verify.py
```

### Rebuild the application figures

```bash
MPLCONFIGDIR=/tmp/jlens-mpl PYTHONPATH=. \
  ./.venv/bin/python -m jlens.rebuild_figures
```

### Recompute one model end to end

The command below requires model weights and a J-Lens checkpoint:

```bash
PYTHONPATH=. ./.venv/bin/python -m jlens.contrastive_logit_steering \
  --model Qwen/Qwen3.5-4B \
  --layer 21 \
  --lens <path-to-workspace-lenses>/qwen3.5-4b/j-lens/lens.pt \
  --scale-json out/rare/qwen35_4b_rome_paris.json \
  --out /tmp/qwen35_4b_rome_paris_recomputed.json
```

The scripts in `tests/` are model-dependent numerical checks, not lightweight
unit tests: they may download weights and need substantial memory. The offline
`verify.py` check is the intended first validation.

## Claim-to-evidence map

| application claim | implementation | checked-in output | figure |
|---|---|---|---|
| SVD directions can steer | [`gender_axis.py`](jlens/gender_axis.py) | [`gender_axis.json`](out/rare/gender_axis.json) | interactive overview |
| `Jᵀw` beats the raw logit direction | [`contrastive_logit_steering.py`](jlens/contrastive_logit_steering.py) | [`*_rome_paris.json`](out/rare/qwen35_4b_rome_paris.json) | [`fig1`](out/figs_final/fig1_jacobian_vs_logitlens.png) |
| The push is slot-like, not belief-like | [`contrastive_fact_generation.py`](jlens/contrastive_fact_generation.py) | [`*_facts.json`](out/rare/qwen35_4b_rome_paris_facts.json) | [`fig2`](out/figs_final/fig2_slot_not_belief.png) |
| A concept is distributed across the spectrum | [`pullback_rank.py`](jlens/pullback_rank.py) | headline JSON above | [`fig3`](out/figs_final/fig3_spectrum.png) |
| Advantage decreases with depth | [`contrast_sweep.py`](jlens/contrast_sweep.py) | [`sweep_*.json`](out/rare/sweep_qwen35_4b.json) | [`fig4`](out/figs_final/fig4_contrast_layer_sweep.png) |
| Headroom does not explain the shift | [`headroom.py`](jlens/headroom.py) | [`headroom_*.json`](out/rare/headroom_qwen35_4b.json) | [`fig5`](out/figs_final/fig5_headroom.png) |
| Structure emerges through training | [`training_fig.py`](jlens/training_fig.py) | checkpoint summaries | [`fig6`](out/figs_final/fig6_training_subspace.png) |
| Nearby layers share subspaces | [`layer_subspaces_traj.py`](jlens/layer_subspaces_traj.py) | [`layer_subspaces_traj.json`](out/rare/layer_subspaces_traj.json) | [`fig9`](out/figs_final/fig9_layer_subspaces_traj.png) |

## Repository layout

```text
jlens/          maintained experiment and figure code
tests/          model-dependent numerical correctness checks
out/rare/       curated JSON results used by verify.py
out/figs_final/ application-ready figures rebuilt from saved JSON
docs/           methods notes, limitations, and interactive results
_attic/         exploratory and negative-result code kept for provenance
```

`out/` is ignored by default because local model runs can exceed 100 GB. Only the
small, curated results and final figures already committed to Git are part of
the reproducibility surface. Large lens tensors (`*.pt`) remain local.

## Methodological cautions

- All steering comparisons use matched intervention norm; absolute effect sizes
  are not comparable when layer, position policy, or dose changes.
- The direction is added at every token position. Position often dominates the
  result; see [`docs/POSITION_IS_THE_VARIABLE.md`](docs/POSITION_IS_THE_VARIABLE.md).
- Category word lists are hand-written proxies for concepts. Qwen shows a small
  but statistically detectable leak into unrelated geography.
- The implementation conventions that fail silently are documented in
  [`docs/JLENS_HANDOFF.md`](docs/JLENS_HANDOFF.md).
- The complete experiment log, including negative results, lives in
  [`docs/MASTER_REPORT_terse.md`](docs/MASTER_REPORT_terse.md).
