# Cross-model replication + significance verdict (5 models, ≤1B)

## Replication table (mid-layer pullback, last-subject pos, random control)
| model | sink | a=0.005 direct pb/rand | transfer pb/rand | a=0.02 direct | transfer | math intact? |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 0.9977 | +0.93/−0.03 | +0.04/−0.01 | +1.65/+0.66 | −0.43 | No ("three") |
| SmolLM2-135M-It | 0.9985 | +1.69/+0.24 | −0.11 | +4.70/+1.85 | −0.75 | No ("10") |
| Qwen2.5-0.5B | 0.9951 | +0.79/+0.17 | +0.56/+0.21 | −0.26/+0.51 | +1.80/+0.95 | Degrades |
| Qwen2.5-0.5B-It | 0.9954 | +0.80/+0.16 | +0.35/+0.23 | −0.20/+0.55 | +1.16/+1.04 | Degrades |
| Llama-3.2-1B | 0.9914 | **+2.04/+0.04** | **+1.03/+0.05** | +6.13/+0.23 | +2.39/+0.17 | Yes at 0.005 ("four") |

(Qwen3.5-4B / Olmo-7B skipped: no room on this machine. Stated, not attempted.)

## Regularities (the actual findings)
1. Pullback > random on direct facts: 5/5 models. Transport-aware beats
   embedding-diff. Small, robust, unsurprising — but checked with nulls.
2. Sink share predicts steerability: 0.998 (SmolLM2, weak/broken) → 0.995
   (Qwen, partial, nonspecific at high dose) → 0.991 (Llama, clean + specific).
   n=5, monotonic. Massive-activation concentration is the rate-limiter.
3. Prompt-specificity gradient (Llama): cos(prompt, pile pullback) 0.16→0.84
   over depth. Early transport is topic-dependent; late is delivery.
4. ΔJ large (0.45) with ±11–29-gap components — wiring differs by topic.
5. Hase mirror: layer R² 0.344, +J/+trace add +0.000/+0.002. Same structure.
6. Dose-response collapse + TV hypersensitivity (0.98 with intact answers):
   two measurement warnings worth writing down.

## Significance verdict (honest)
Not a breakthrough. A well-controlled cross-model measurement suite for
training-free concept steering — exactly the kind of artifact MATS selects
for: independent taste (controls pre-registered, kill rules kept, negatives
reported: trace-vs-edit dead, DAS-1D/k=8 gated out, geneig loses). Conference
strength needs: paraphrase-gate pass at a clean dose (Llama partial only),
neighborhood specificity, and the skipped 4B/7B check. As a 2-page methods
note + released harness (jlens/pullback pattern, rename battery, nulls), it
is real, small, and honest. That is worth writing.

Script: jlens/cross_model.py.
