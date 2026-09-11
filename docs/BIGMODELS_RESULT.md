# Big models on disk: Qwen3.5-4B renames, Olmo-7B barely moves (raw-d only)

## What fits on 32GB
- Qwen3.5-4B (32 layers, d=2560, hn≈12.7): forward + VJP attempt fit.
- Olmo-3-1025-7B (32 layers, d=4096, hn≈16.2): forward-only fits; VJP skipped.
- ~~No pullback on either: MPS-fp16 VJP overflows to NaN; CPU-fp32 VJP returns
  exactly 0.0 (structurally disconnected grad on the new arch).~~
  **CORRECTED 2026-09-06.** The VJP is NOT broken on these architectures. On
  Qwen3.5-4B with **bf16 on CPU** it works cleanly: grad norm 201.5,
  `is_leaf=True`, `grad_fn=AddBackward0`. The failure was MPS+fp16 specific.
  (fp16 itself is not the culprit either — on Llama-1B, fp16 gives grad norm
  4.1703 against fp32's 4.1708, no underflow.) Use bf16/CPU for big-model VJPs.
  Results below are still raw unembed-diff, but pullback is now available.

## Results
Qwen3.5-4B @push≈6 (α=0.5): direct +5.76 (rand +0.74), transfer +1.69
(rand −0.10 — i.e. **direction-specific, beats the random control**), math
"four" ✓, generations coherent ("Rome. A. True…").

**Wording correction 2026-09-06.** "Specific" above means *beats a random
direction*, not *hits only its target*. Those are different claims and only the
first is supported here. The second was tested afterwards with pullback at L16
and **fails at this scale too**:

| dose | France (target) | Germany | Spain |
|---|---|---|---|
| α=0.3 | +8.38 | **+9.94** | **+8.94** |
| α=0.6 | +11.38 | **+12.25** | +11.31 |

Unrelated capitals move as much as or more than the target — the same pattern
seen at 135M (France +2.59 vs Spain +3.07).

**Second correction, same day.** Do not read the table above as "specificity
fails". Two confounds were found afterwards:

1. **Dose.** Both rows are at a saturated dose. On SmolLM2 the identical
   measurement across a dose sweep shows specificity is dose-dependent and
   *reverses*: pullback scores efficacy +0.69 / leakage +0.19 (specificity
   **+0.51**) at α=0.5, and +3.00 / +3.02 (specificity **−0.02**) at α=1.0. The
   trade-off correlation between efficacy and specificity is −0.43 and −0.47 at
   usable doses and +0.24 at saturation. The 4B numbers above need the same
   sweep before they mean anything.
2. **Headroom.** Effect size tracks how much room a prompt has to move. In the
   4B table Germany has the lowest base logprob (−8.56, most headroom) and the
   largest shift (+9.94); France (−6.50) shifts +8.38. This is the same
   confound `DIVERSE_RESULT.md` flags for "transfer exceeds direct". Until
   effect is regressed on base logprob, target-vs-unrelated comparisons are not
   interpretable.

Status: **open, not answered.** Needs a dose sweep and a headroom control at 4B.
Olmo-7B @push≈8 (α=0.5): direct +1.17 (rand +0.22), transfer +0.40
(rand +0.71, noise), math "four" ✓. Barely moves, fully coherent — needs a
bigger absolute push; hn-unit dosing (α×hn) is meaningless across models
(hn: 2900 SmolLM2 / 104 Llama / 13 Qwen-4B / 16 Olmo). Dose in absolute
norms, not hn multiples.

## Pattern across 7 models
Bigger → more coherent, smaller absolute effect per unit push, and
*direction*-specificity (beats random) when it moves. **Target-specificity does
NOT improve with scale** — see the correction above. The sink law holds in spirit (all hn ≫ per-dim
scale), but hn itself is the wrong ruler — its sink share varies.

Script: jlens/big_rename.py (--skip-pb for 7B).
