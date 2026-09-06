# Big models on disk: Qwen3.5-4B renames, Olmo-7B barely moves (raw-d only)

## What fits on 32GB
- Qwen3.5-4B (32 layers, d=2560, hn≈12.7): forward + VJP attempt fit.
- Olmo-3-1025-7B (32 layers, d=4096, hn≈16.2): forward-only fits; VJP skipped.
- No pullback on either: MPS-fp16 VJP overflows to NaN; CPU-fp32 VJP returns
  exactly 0.0 (structurally disconnected grad on the new arch — reported, not
  debugged further). All below is raw unembed-diff + random control.

## Results
Qwen3.5-4B @push≈6 (α=0.5): direct +5.76 (rand +0.74), transfer +1.69
(rand −0.10, specific!), math "four" ✓, generations coherent ("Rome. A.
True…"). Clean rename on raw alone — pullback unnecessary here.
Olmo-7B @push≈8 (α=0.5): direct +1.17 (rand +0.22), transfer +0.40
(rand +0.71, noise), math "four" ✓. Barely moves, fully coherent — needs a
bigger absolute push; hn-unit dosing (α×hn) is meaningless across models
(hn: 2900 SmolLM2 / 104 Llama / 13 Qwen-4B / 16 Olmo). Dose in absolute
norms, not hn multiples.

## Pattern across 7 models
Bigger → more coherent, smaller absolute effect per unit push, cleaner
specificity when it moves. The sink law holds in spirit (all hn ≫ per-dim
scale), but hn itself is the wrong ruler — its sink share varies.

Script: jlens/big_rename.py (--skip-pb for 7B).
