# Injection position dominates everything else (SmolLM2-135M, L12)

**2026-09-06.** Written because several results in this repo — mine and the
autonomous agent's — are not comparable to each other until this is controlled.

## The measurement

Same direction (pullback `Jᵀw`), same layer, same dose, same 30-draw null.
Only the injection position differs.

| position | α | ‖h‖ | pullback | rand mean | rand sd | z |
|---|---|---|---|---|---|---|
| subject token ("France") | 0.5 | 149 | +9.31 | +0.34 | 0.73 | **+12.3** |
| subject token ("France") | 1.0 | 149 | +14.23 | +1.15 | 1.32 | **+9.9** |
| final token ("is") | 0.5 | 123 | +0.18 | +1.51 | 1.04 | **−1.3** |
| final token ("is") | 1.0 | 123 | +3.03 | +3.36 | 1.91 | **−0.2** |

At the last subject token the direction is 12σ above chance. At the final token
it is **worse than random**. This is ROME's localisation premise, confirmed
directly.

## What this invalidates

Any result in this repo that injected at the final token is measuring noise:

- The 4B "specificity fails" table in `BIGMODELS_RESULT.md` (already corrected
  twice; this is the third and the real reason).
- `jlens/tradeoff.py` and `jlens/leak_ladder.py` as first written.

Anything that injected at the last subject token stands.

## Specificity, measured correctly

α=0.5, each prompt edited at **its own** last subject token, 30-draw null per cell:

| kind | prompt | base | effect | rand mean | z |
|---|---|---|---|---|---|
| target | The capital of France is | −5.18 | +9.31 | +0.42 | +11.6 |
| target | France's capital city is | −2.55 | +1.20 | −0.02 | +2.3 |
| same-subject | The Eiffel Tower is in | −4.77 | +3.23 | +0.34 | +4.8 |
| same-category | The capital of Germany is | −6.27 | +4.56 | +0.69 | +5.1 |
| same-category | The capital of Spain is | −6.15 | +6.18 | +0.90 | +7.5 |
| same-category | The capital of Japan is | −9.03 | +5.33 | +0.85 | +6.1 |

**Partial specificity.** The target moves roughly twice as much as unrelated
capitals, but those are all significantly above their own nulls. Not a clean
edit, not a pure global push.

The ordering is **not semantic** — the paraphrase of the target moves least
(+1.20). It tracks base logprob, i.e. headroom: Japan (−9.03) had room, the
paraphrase (−2.55) did not. France is the single point sitting well above the
headroom trend, and that excess is the specificity signal. **Any target-vs-other
comparison must regress effect on base logprob first**; this is the same
confound `DIVERSE_RESULT.md` flags for "transfer exceeds direct".

## Two mistakes I made here, both already in the house rules

1. **One random draw.** The first ladder run used a single random direction; it
   happened to beat the targeted direction by 18× on one rung, making the whole
   run uninterpretable. Thirty draws fixed it. (I had flagged this exact issue in
   the agent's `cross_model.py` an hour earlier.)
2. **Saturated dose.** The first trade-off run was at α=1.0, where every
   direction bulldozes and the efficacy/specificity correlation flips sign
   (−0.43 at α=0.3, −0.47 at α=0.5, **+0.24** at α=1.0). Report a dose sweep,
   never a single dose.

## Also worth recording

The "99.8% of variance in one direction" figure is the **uncentred** second
moment — mostly the constant mean offset, which carries no information. Centred
at the same layer it is 8.6% (top-1) and 24.0% (top-8). The concentration is
real but strongly layer-dependent: 71.5% centred at L4, ~9% at L12.

## Protocol for anything after this

1. Inject at the last **subject** token, and say so.
2. Sweep dose; never report one.
3. ≥30 random draws per cell, report z not raw deltas.
4. Regress effect on base logprob before comparing prompts.
5. Coherence gate relative to the **unedited** model's repetition rate — the base
   model already scores 0.55 on "The sky is".
