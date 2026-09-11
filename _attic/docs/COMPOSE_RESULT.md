# Composed Rome-Paris steering (SmolLM2-135M, France→Rome)

## Idea
Raw d = norm(W_U[Rome]−W_U[Paris]) works weakly (+0.27 nats over random in the
pilot). Instead of filtering, weight every SVD component by its usefulness.

## What was scanned
All 576 singular components per layer (L10/12/14), each scored by its own
readout gap logP(Rome|Jv_i) − logP(Paris|Jv_i). ~half favor Rome (npos 273–295).
Top-gap indices (290, 309, 112…) are NOT top-gain — the signal lives in
low-gain subspace. Top-8 gap components hold only ~3% of d's energy
(kept_frac 0.023–0.033); top-64 holds ~11%.

## Result (α=1.0, clean run, last-subject-token, ΔlogP(Rome))
| layer | raw | composed top-8 | composed top-64 | pullback Jᵀw | random |
|---|---|---|---|---|---|
| 10 | +4.03 | +3.10 | — | **+5.09** (P=0.086) | +2.11 |
| 12 | +3.48 | +2.69 | +3.27 | **+4.96** (P=0.075) | +0.62 |
| 14 | +2.75 | +1.20 | — | **+4.83** (P=0.066) | +1.85 |

Pullback p = Jᵀw/||Jᵀw|| (w = Rome−Paris unembed diff) is the optimal linear
steerer and beats raw by +1–2 nats (3–8× in P(Rome)) at every layer tested.
Hard top-k filtering underperforms raw at k=8 and only ties it at k=64.

## Lesson
Compose with weights (pullback = Σ s_i(u_i·w)v_i), not cutoffs. Filtering by
gap alone discards the 90%+ of d's energy spread across individually-mediocre
components that collectively do the work.

## Caveat
Specificity is still poor for ALL methods (neighbor "Louvre→Paris" moves
+1–3 nats for pullback). Efficacy without specificity is bulldozing, not
editing — next test must pair pullback with a neighborhood + coherence gate
before claiming an edit.

Script: jlens/compose_steer.py (out/ JSONs gitignored; rerun with --topk/--alpha).
