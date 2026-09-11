# What we have

Plain summary of the whole project. Written 2026-09-07.

**Scale:** 168 analysis scripts, 74 commits, 22 result documents, 17 result pages,
5 model families, ~22GB of computed data.

---

## The question we started with

**J-Lens** is a way of looking inside a language model. You take a matrix J that
describes how a nudge at one layer travels to the output, break it into its main
directions, and read what each direction "means" by pushing it through the
model's word-prediction matrix.

We set out to characterise it: does it work, and when?

---

## Four things we investigated, and what each gave us

### 1. Language models — the original arm

Took 2160 directions from J across four models, tried to label each with a
concept, then tried to steer with them.

**Result: 63 of 2160 pass a statistical readout test. 3 change generated text.
All 3 are in one 135M model.**

Also: a re-check found the directions further down the list are no better than
random, so it is not that we looked in the wrong place.

**Verdict: a clean negative.** Reading individual directions of J does not
recover features.

### 2. Chess — why the negative happens

Switched to a chess-playing model because chess has an oracle: `python-chess`
says exactly which moves are legal, so nothing depends on human labelling.

**Result: the top of J's spectrum encodes a tokenisation artifact, not chess.**
The strongest thing those directions carry is *which half of a move is being
emitted* (R² 0.66) -- a consequence of how moves are tokenised. Actual chess
properties (captures 0.16, checks 0.07) need the whole spectrum.

Underneath: J is an average over prompts, and averaging over different
computations gives you the *mixture*, not any computation.
`cos(J_origin, J_destination) = 0.21` -- nearly orthogonal.

**Verdict: a mechanism for the negative.** Not just "it fails" but "here is why".

### 3. Mixture-of-Experts — the newest arm

OLMoE has 16 layers x 64 experts. Each expert has a "router direction" -- a
vector in the model's internal space saying what that expert looks for. Those
come free with the checkpoint.

**Result A (the best positive):** the model's own, crowded, non-orthogonal
directions are more readable than an orthogonal basis of the same subspace.
Holds under four different measurements, including one needing no concept list.
Ratio between 1.9x and 3.7x. Mechanism verified: each orthogonal direction is a
blend of ~17 experts, and a blend of 17 concepts reads as nothing.

**Result B:** the Jacobian loses to doing nothing. Predicting which experts fire
from an earlier layer: the raw residual gets 7.2 of 8 correct; the Jacobian
gets 1.3, where chance is 1.0. Zero wins in 15 cells, and centring it properly
does not help (+0.08 of 8).

**Result C:** routing tracks surface form, not algorithm. Five experts fired on
100% of "LCM of X and Y" problems and 0% of addition with identical digits --
then collapsed to 0-35% when the wording changed to "least common multiple".
A generic arithmetic expert survived at 100%.

**Result D (one clean demo):** push the medicine expert's direction and the
model goes from "the type of material you want to use" to naming a real
chemotherapy drug, staying coherent.

### 4. Steering and localisation

**Position dominates everything.** Same direction, same layer, same dose:
+12.3 standard deviations at the subject token, -1.3 at the final token.
Confirms ROME's premise directly, and invalidated three of our own earlier
measurements that had used the wrong position.

**Dose determines the conclusion.** The correlation between how well a direction
steers and how specific it is: -0.47 at moderate strength, +0.24 at saturation.
Same directions, opposite conclusion.

---

## The failure catalogue

Every one of these killed a result of ours. Each is a mistake that is easy to
make and hard to see.

1. **A metric that is mostly its own baseline.** Chess legality probes at AUC
   0.999, of which 88-98% was the per-square marginal -- e2 is *usually* a legal
   pawn origin. Four measurements were wrong before this surfaced.
2. **A concept category that acts as a magnet.** "quantifier" absorbed 60 of 87
   hits, because its words are high-frequency function words whose average sits
   near the generic direction. Dropping it promoted "comparison" to the same role.
3. **A chance line derived instead of measured.** `k/min(m,n)` for a tall matrix
   where the truth is `k/m`: off by 5.4x, and it manufactured an entire
   component-level finding.
4. **A single dose.** See above: the sign flips.
5. **A single random draw.** One random direction beat the real one by 18x on a
   rung. Thirty draws fixed it.
6. **Comparing unlike with unlike.** A "cross-model replication" compared a
   post-audit set against a pre-audit one, and looked like a discovery.
7. **The final token.** Every prompt set ended in a different token, and routing
   is computed *at* that token -- so the whole enrichment result was "the router
   can tell ' is' from ' '".
8. **An outlier that owns the variance.** A finding passed a 300-draw null and
   then reversed sign at every layer once one massive-activation direction was
   projected out.

---

## What it adds up to

**One positive:** non-orthogonal directions read better than orthogonal ones,
with a measured mechanism and two alternative explanations excluded.

**One clean negative:** the Jacobian transport is not worth its cost --
eight measurements, three model families, two readout spaces, with a positive
control proving the instrument could detect signal when it was there.

**One mechanism:** prompt-averaged Jacobians are averages over different
computations, so their dominant directions encode the mixture rather than any
computation.

**One method contribution:** the failure catalogue above, each item with a
worked example and a number.

**What we do not have:** a discovery. Nothing here says something new about what
models believe or do. It says things about what a class of measurement can and
cannot see.
