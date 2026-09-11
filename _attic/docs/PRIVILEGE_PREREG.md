# Pre-registration: is the model's own basis measurably privileged?

Written 2026-09-07, BEFORE any of the numbers below were computed.

## The question in one sentence

When you look for interpretable directions, you can read the model's own
vectors (a neuron's write direction, an attention head's output column), or
you can rotate them — SVD/PCA the matrix and read the top components.
Rotating does not change the subspace. Does it change what you can read?

## Why theory says "sometimes"

A basis is *privileged* only where an elementwise operation acts in it
(Elhage et al., "Privileged Bases in the Transformer Residual Stream").

- **Inside one attention head** there is no elementwise nonlinearity. For a
  multi-head-attention model you can replace W_V -> R W_V and W_O -> W_O Rᵀ
  for any orthogonal R and the model computes *exactly the same function*.
  The head's internal basis is therefore arbitrary — a training accident.
- **In the MLP** the nonlinearity is elementwise, so the same substitution
  changes the function. The neuron basis is not arbitrary.

This gives a provable invariance to calibrate an instrument against. That is
the whole point of the design: one arm has a known-zero answer.

## The instrument

For a set of n=64 direction vectors forming matrix A (d x n, unit columns),
with QR decomposition A = QR:

| arm | construction | spans S? | Gram matrix |
|---|---|---|---|
| `raw` | A | yes | AᵀA |
| `svd` | Q (ordered by singular value) | yes | I |
| `gram_null` | Q H R, H ~ Haar(n) | yes | RᵀHᵀHR = AᵀA — **identical to raw** |
| `orth_null` | Q H, H ~ Haar(n) | yes | I |

`gram_null` is the key control: same subspace, *same pairwise angles and
conditioning as the real basis*, random identity. Any gap between `raw` and
`gram_null` cannot be explained by orthogonality, norms, or conditioning.
It can only be explained by *which* directions those are.

**Readout**: logits = W_U (gamma ⊙ u), the standard logit lens (RMSNorm, so
the scale drops out of the ranking). Take the top 10 tokens.

**Score**: NPMI topic coherence (Bouma 2009) of those 10 tokens, computed on
8192 held-out windows of the Pile. This is the standard label-free coherence
metric from topic modelling. It never touches the model's weights, so it
cannot be circular.

## Predictions (in order of confidence)

1. `attn_head` (within one head): raw ≈ gram_null. **z ≈ 0.** Provable invariance.
2. `mlp_write` (down-projection columns): raw > gram_null. **z > 0, large.**
3. `attn_cross` (columns sampled across heads): raw > gram_null, because
   rotating across heads mixes heads and head boundaries are real.
4. `residual` (standard basis of the residual stream): raw > gram_null but
   smaller than mlp_write — RMSNorm's gamma is elementwise, so weak privilege.
5. `router` (MoE router rows): raw > gram_null.
6. Across all arms, `svd` sits with the nulls, not with `raw`.

## Kill criteria (decided now, not later)

- **K1.** If `attn_head` z is within 2x of `mlp_write` z, the instrument is
  measuring "single vector vs mixture of 64 vectors" (a central-limit
  artifact), not privilege. Report as a negative and stop.
- **K2.** If mean log token-frequency of the top-10 differs between `raw` and
  `gram_null` by more than 0.5 nats, NPMI is frequency-confounded; the
  comparison must be redone frequency-matched before it is reported.
- **K3.** If the OLMoE-0924 -> OLMoE-0125 replication (independent training
  run, identical architecture) flips any sign, the result is checkpoint noise.
- **K4.** If more than 30% of top-10 tokens are unseen in the corpus for any
  arm, that arm's NPMI is not trustworthy and is reported as N/A.

## What a positive result would license

A quantitative exchange rate for orthogonalisation: "running SVD on this
matrix and reading the top components costs you X% of the coherence that was
sitting in the rows." Plus a calibrated null (the attention arm) that says
how much of any such gap is measurement artifact.


---

## Addendum, written 2026-09-08, after Experiments A and B were complete

Experiment C (does the model's own basis *act* better, not just read better?)
was added. Its predictions and kill criteria, again written before the numbers:

**Predictions**
- C1. The attention-head arm reads zero causally too. Rotating a head's basis
  is exactly function-preserving, so raw and orbit are two names for the same
  weights; this must hold at every dose.
- C2. If the MLP arm's readout advantage reflects something real about the
  directions, it should survive as a causal advantage.

**Kill criteria**
- **KC1 (dose).** If the mean pairwise cosine of the induced logit change
  across the 64 directions ("common-mode") exceeds 0.05, the intervention is
  not direction-specific and no basis comparison can discriminate. That dose is
  void.
- **KC2 (magnitude).** If the induced top-10 change is under 0.01 nats the
  intervention is below numerical noise and the dose is void.
- **KC3 (power).** Report the mean within-group z with a bootstrap CI. If the
  CI is wide enough to contain both zero and Experiment B's effect size for the
  same arm, report "underpowered", not "no effect".

**KC1 fired.** The first Experiment C run used alpha in {0.25, 0.5, 1.0}, the
doses this repo has used before. Common-mode at alpha=1.0 reached 0.46 pooled
and 0.97 in the MLP arm at layer 1. That run produced a flat null in every arm
and was discarded rather than reported. Experiment C was re-run at alpha=0.05,
where common-mode is 0.005 and the result is dose-independent from 0.003 to 0.1.
