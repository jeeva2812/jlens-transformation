# Trace vs J vs edit: pilot result (SmolLM2-135M, 6 capital facts)

## Question
Hase et al. 2023: causal-tracing effect does not predict ROME edit success
(ρ = −0.13; layer alone explains 94.7% of variance, +trace → 94.8%).
Does pile-averaged Jacobian amplification ||J_l d|| predict activation-edit
success where tracing fails? J is a linear model of perturbation propagation,
so it should — if localization is informative under the right measure.

## Method (same-d, same-position, same-norm throughout)
- Model: HuggingFaceTB/SmolLM2-135M (base; J from out/Jall_smollm2.pt, 15 layers, target 28).
- Facts: 6 single-token capital facts (France/Paris→Rome, Italy/Rome→Paris,
  Germany/Berlin→Paris, Spain/Madrid→Rome, Portugal/Lisbon→Madrid,
  Netherlands/Amsterdam→Berlin). Prompt: "The capital of X is".
- Position: last subject token for ALL three measures.
- d (fixed per fact): norm(W_U[false] − W_U[true]). Same d for J and edit.
- Tracing: Gaussian noise (σ=0.1) on subject embeddings; restore clean h_l at
  subj_pos; effect = (p_restored − p_corr)/(p_clean − p_corr).
- Edit: CLEAN run + inject (α·||h_l||)·d at subj_pos, α=1.0; success = ΔlogP(false).
  (Earlier pilot mixed corruption into edit runs — fixed; see script history.)
- J-amp: ||J_l d|| from saved pile-averaged J.
- Controls (pre-registered): 50 random directions same norm; outlier control
  d_oc = d minus top pile-PC per layer, with J*d_oc and edit_oc recomputed.
- Script: jlens/trace_vs_j.py. Run: out/trace_vs_j_fixed.json (90 points).

## What the run actually printed
- corr(trace, edit) = +0.293 (Hase: −0.13 — sign differs; expected: different
  model/edit/corruption, and our trace overshoots >1 on some layers)
- corr(J*d, edit) = +0.419; outlier control +0.418 (survives)
- mean edit_rand = +1.15 nats (should be ~0 — pushes flatten the distribution;
  specificity edit−rand = +0.27 nats only)
- J*d = 1.42 vs J*rand = 1.38 (no direction specificity in gain)
- hn ≈ 149 at L12 (massive-activation regime; position-matched scaling used)

## The control that kills the pooled numbers
Pooled correlations are a depth artifact. Within-layer (across facts at fixed
layer): trace–edit +0.028, J–edit −0.064 — both ~0. Hase-style regression:
R²(edit ~ layer) = 0.344; +J → 0.344 (+0.000); +trace → 0.346 (+0.002).
Argmax agreement per fact (best edit layer vs best J vs best trace): 1/6 for J.
Neither predictor adds anything beyond the layer prior — same structure as
Hase (94.7% → 94.8%), at lower absolute R² because single-token activation
steering on a 135M model is weaker/noisier than ROME weight edits.

## Verdict
The CHEAP version (pile-averaged J + unembed-diff d) is dead as an edit-site
predictor: no within-layer signal, no R² gain, no direction specificity.
This is a valid negative result, not a methods bug hunt — two real bugs were
found and fixed along the way (prob/logprob mix; corruption leaked into edit
runs; hook return-value crash), and the conclusion survived them.

## What could revive it (one shot, not a program)
Fact-conditioned gain (JVP on the fact prompt with a mean-diff d) should predict
small-alpha edits nearly by Taylor tautology — but that loses the "cheap,
precomputed J" advantage that motivated the idea, and still has to beat the
layer prior within-layer, where both predictors currently score ~0.
If that also scores ~0 with a specific edit (edit−rand ≫ 0), the space is
exhausted: report it and move on.
