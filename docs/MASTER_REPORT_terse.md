# What is a direction, really?

**Everything this project did, in order.** Twelve arms, most negative, one that
finally got an answer that couldn't be argued with. Numbers here were recomputed
from the models on disk while writing; `verify.py` reproduces the headline ones.

---

# PART A — The question

Nearly all interpretability works the same way. You find a **direction** inside a
model — a list of numbers pointing somewhere in its internal space — and you say
what it means. *"This direction means medical." "This neuron fires on Python."
"Direction 12 of head 7 promotes surnames."*

Every arm below is a different attack on one question: **are those directions
real?**

## The answer, in six sentences

1. **Directions work as levers.** Push a model along a gradient direction and its
   answer moves, reliably, better than the obvious alternatives, on 8 models.
2. **But they aren't surgical.** Pushing "France → Rome" moves Germany and Spain
   almost as much, and most of the variation is explained by how unlikely the
   answer was to start with, not by meaning.
3. **What a direction looks like it means is often not what it means.** In a chess
   model the strongest direction mostly tracks a bookkeeping fact about the text
   format, not chess.
4. **Sometimes the direction isn't even well-defined.** For attention heads there
   is an edit that leaves behaviour *exactly* unchanged and replaces **99%** of
   what you'd read out of that head.
5. **That has teeth.** A monitor trained on head features drops to chance
   (0.91 → 0.55) on a model that generates identical text. Two real sparse
   autoencoders score **100%** on the same test.
6. **Applying (4) to our own earlier result overturned it** — and revealed a real
   effect the mess had been hiding.

---

# PART B — The twelve arms

## 1. The instrument: a Jacobian lens

Everything early rests on one object. Take a middle layer, and ask: if I nudge the
model's state here, how does the final output move? That's a matrix, `J`.

Two disciplines that mattered, both learned the hard way:

- **Type discipline.** `J = UΣVᵀ`. The `U` directions live in output space and can
  be read as tokens. The `V` directions live in the layer's space and can be
  injected. Mixing them up produces confident nonsense, and we did it before we
  noticed.
- **Cost.** One backward pass gives `Jᵀw` for *every* layer at once. So the
  expensive-looking thing is cheap. (Early on we thought the backward pass was
  broken on newer architectures — it wasn't; the failure was specific to one
  numeric format on one accelerator. Corrected in `BIGMODELS_RESULT.md`.)

## 2. Steering with the pullback — **the biggest positive**

**Question.** If a direction means "Rome," pushing along it should make the model
say Rome.

**Method.** `Jᵀw`, where `w` is the difference of the two output embeddings. By
Cauchy–Schwarz this is the direction at that layer which maximally raises "Rome"
over "Paris" — the optimal linear lever, not a heuristic.

**Result.**

| direction | ΔlogP("Rome"), Llama-3.2-1B |
|---|---|
| pullback `Jᵀw` | **+5.38** |
| difference of output embeddings | +3.69 |
| random | +0.28 |

Same ordering at every layer of SmolLM2 (+4.96 / +3.48 / +0.62 at layer 12).
Direct effects replicate in **46 of 48** model × scenario combinations across 8
models. New topics steer *more easily* than capital cities: food, tech and
currency reach +8 to +11 nats.

**Things we tried that did NOT beat it:**
- hard top-k filtering of `J`'s components (worse than raw at k=8, ties at k=64) —
  the signal lives spread across individually-mediocre components
- generalised eigenvectors / whitening (steer +1.72 vs pullback +3.09)
- pushing three layers at once instead of one (identical, at every dose tested)

**Limits, honestly.**
- **Not surgical.** At a matched, non-saturating dose, with 30 random draws per
  cell: target France **+9.31** (z = +11.6), but Spain **+6.18** (z = +7.5),
  Japan **+5.33**, Germany **+4.56** — all far above their own nulls.
- **Effect size tracks headroom, not meaning.** The *paraphrase* of the target
  moved least (+1.20) because it had the least room. Any target-vs-other
  comparison must regress on base log-probability first.
- **Narrow dose window**, roughly 0.006–0.02 in our units. At 0.06 everything
  degenerates; below 0.006 nothing moves.
- **Scale matters and we found the reason.** Steerability tracks how concentrated
  the layer's activations are: sink share 0.998 (SmolLM2, unsteerable) → 0.995
  (Qwen, partial) → 0.991 (Llama, clean). *Correction:* the widely-quoted "99.8%
  of variance in one direction" is the **uncentred** figure and is mostly a
  constant offset. Centred it is **8.6%** at layer 12 and 71.5% at layer 4.

*Check: `verify.py 6`*

## 3. Supervised subspaces — **four attempts, four negatives**

If the pullback isn't surgical, maybe a *learned* subspace is.

| attempt | result |
|---|---|
| **DAS, 1 dimension** | loss flat, effect +0.00. Diagnosed: the patch moved **less than 1%** of the state, so no gradient existed. A result about scale, not about the method. |
| **DAS, 8 dimensions** | fits all 3 training phrasings, transfers to **0** held-out ones (−0.00, +0.09). Memorised template geometry. |
| **De-sinked DAS** | one phrasing moves 6×, the others flat. A paraphrase failure. |
| **Generalised eigenvectors** | gains explode (1200×) while occupancy goes to **0.00×** random — the objective maximised a ratio by sending the denominator to zero. |

We also compared **ΔJ** (the wiring difference between Paris-prompts and
Rome-prompts) against `J` itself: `‖ΔJ‖/‖J‖ = 0.45`, and its components read at
gaps of +11.3 where `J`'s own read at +5 to +10. So the fact-specific wiring is
strong and concentrated — but the one component we steer-tested was the wrong
side of the contrast, at the wrong position, at a saturated dose. **Recorded as
untested, not as a result.**

## 4. Does the Jacobian predict where to edit? — **a clean negative**

Hase et al. found that causal-tracing effect doesn't predict edit success. We
asked whether `J`-amplification does better.

Pooled correlations looked promising (+0.42). Then the control: **within a fixed
layer**, across facts, the correlation is **−0.064**. A regression on layer alone
explains R² = 0.344; adding `J` adds **+0.000**; adding tracing adds +0.002.
Argmax agreement 1/6. Same structure as Hase, at lower absolute R².

Two real bugs were found and fixed along the way, and the conclusion survived
both.

## 5. Position and dose — **the method finding that invalidated our own work**

Same direction, same layer, same dose. Only *where* it is injected changes:

| model | at the subject word | at the last word |
|---|---|---|
| SmolLM2-135M | +0.58 | **−3.24** (sign flips) |
| Llama-3.2-1B | +5.38 | +2.35 |

And dose reverses the relationship between efficacy and precision: correlation
−0.43 at α=0.3, −0.47 at α=0.5, **+0.24** at α=1.0.

This invalidated several of our own earlier numbers. The protocol that came out
of it — inject at the last subject token and say so, sweep dose, ≥30 random draws
per cell, regress on base log-probability, gate coherence against the *unedited*
model — is in `docs/POSITION_IS_THE_VARIABLE.md`.

*Check: `verify.py 8`*

## 6. Chess, where the answers are objective — **negative, with a mechanism**

**Why.** Human labels are unreliable. A chess model lets a rulebook decide.
Legality checking was verified at **100%** against `python-chess`.

**What the strongest directions actually track:**

| | R² (chance = 0.003) |
|---|---|
| **which half of the move is being typed** (from-square vs to-square) | **0.66** |
| can the queen move? | 0.31 |
| can the rook move? | 0.22 |
| is a capture available? | 0.16 |

The dominant thing is a fact about the **text format**.

**The mechanism.** A prompt-averaged Jacobian averages over different
computations. The from-square and to-square computations point in nearly
unrelated directions (similarity **0.21**), so the average describes the
*mixture* rather than either one.

**Structure that is real.** The readouts do separate the player's own pieces from
the opponent's (+1.54 vs −0.58 in arbitrary units), and `J`'s fidelity to the
model rises smoothly with depth (cosine 0.20 at layer 0 → **0.95** at layer 6).

**A second negative, carefully.** We looked for a subspace per piece type:

| subspace removed | pawn accuracy lost | knight | queen |
|---|---|---|---|
| "pawn" | 1.93% | −0.03% | 1.33% |
| "king" | 1.60% | −0.02% | 1.12% |
| random | 0.01% | 0.00% | 0.00% |

Own-piece 0.752%, someone-else's 0.708%. **Every row is the same** — the six
subspaces are interchangeable. They *are* more important than random ones, so the
method finds something that matters; it just isn't what we labelled it.

*Check: `verify.py 7`*

## 7. Mixture-of-experts — **a negative, then a correction**

**Can we predict routing?** The Jacobian method got **1.3 of 8** experts right.
The dumbest baseline — assume the experts used at one layer are used at the next —
got **7.2 of 8**. Chance is 1.0. Also: routing tracks surface wording, not
meaning; task-specific-looking experts stopped firing on paraphrases.

**The router's directions** looked like they read ~2× better than a random
orthogonal set, and we concluded the effect was about *the shape of the region
they occupy, not which directions they are*.

**Then the correction.** The router has a free component: adding the same number
to every expert's score changes no routing decision (verified — same 8 experts,
weights differ by 3×10⁻⁸). **38.3% of the router's size lives there.** Removing
it — a free operation — inverts the conclusion:

| | raw router | free part removed |
|---|---|---|
| does the identity of these directions matter? | +0.0016 [−0.0127, +0.0131] | **+0.0294 [+0.0203, +0.0385]** |
| is it just the shape of the region? | +0.0236 [+0.0084, +0.0425] | +0.0079 [+0.0036, +0.0129] |

**Our earlier conclusion was the free component in disguise**, and cleaning it up
*uncovered* a real effect rather than removing one.

*Check: `verify.py 4`*

## 8. How directions form during training — **a shape, not a claim**

Using 11 OLMo training checkpoints, we tracked the top 64 directions at layer 20
and measured their similarity to the final ones (chance = 0.125):

| checkpoint | similarity to final |
|---|---|
| stage 1, step 0 | 0.126 — *chance* |
| stage 1, step 8,000 | 0.182 |
| stage 1, step 512,000 | 0.398 |
| stage 1, step 1,413,814 | 0.519 |
| **stage 2, step 8,000** | **0.487 — a step backwards** |
| stage 2, step 47,684 | 0.678 |
| stage 3, step 5,000 | 0.769 |

Directions begin at chance and form **gradually**, not suddenly. And the
**training-stage boundary knocks them back** — the only non-monotone point in the
curve. 34 of the 64 settle late, at stage 2. The measure is guaranteed to end at
1.0, so the content is the *shape*, especially that setback.

## 9. Emergent-misalignment organisms — **deflated by its own control**

Three published fine-tunes (bad medical advice / risky financial advice / extreme
sports) look like they converged on the same internal change.

**The trap.** They share a starting point. Their input-side matrices are cosine
**0.9994** identical *before any training*. So the input half of all three changes
is identical by construction, and any naive weight comparison finds enormous
"convergence" that is initialisation.

**The control.** We trained three of our own with seeds we controlled (sanity:
same seed → 0.9874, different seed → 0.0255):

| | shared-recruitment excess |
|---|---|
| two unrelated **benign** fine-tunes | 1.58× |
| the three misaligned fine-tunes | 2.17× |

Two *benign* fine-tunes already show most of the effect. **Convergence is mostly
generic to fine-tuning this base model.**

**Does it survive a different seed?** Write-side overlap (chance 0.036):
different-init/same-data **0.294** > same-init/different-data 0.139. Task
identity matters more than the seed.

## 10. Reward hacking and chain-of-thought monitoring

Substrate: Qwen3.5-9B trained with group-relative PPO on programming problems
against a deliberately breakable grader. 300 held-out runs released by lucabaroni,
upstream UK AI Safety Institute, judged blind. We did not train it. **Fidelity
gate first:** with the adapter attached we reproduce the released policy's own
token probabilities at r = **0.995** (base model: 0.942).

| | |
|---|---|
| confirmed hacks | 275 / 300 |
| hacks where the reasoning gives no sign | 201 |
| keyword monitor catches, on those | **18.4%** |
| hacks that still contain a real solution attempt | **70.5%** |
| where the exploit sits in the code | median **3.0%** through |
| dead code written after it | median **1,032 characters** |

The model reasons correctly, writes a two-line short-circuit at the top, then
writes the real solution underneath as code that can never run.

**The one clean result.** Only positions where the label isn't already in the
context count, and the baseline is a *text monitor*, not chance:

| question | position | internal | text monitor | null |
|---|---|---|---|---|
| will it hack? | before generating | 0.46 | 0.49 | 0.41 |
| **will a real solution follow?** | at the exploit, before that code exists | **0.811** | 0.449 | 0.39 |
| will the reasoning conceal it? | before any reasoning | 0.56 | 0.53 | 0.52 |

n = 183, with a proper depth curve (0.46 → 0.80 across layers), and it **transfers
to an unseen exploit type** (0.775).

**The causal test, with its prediction confirmed.** Keep the code fixed, destroy
the reasoning: the exploit's probability moves by **−0.011**; the real code drops
**−0.290**. Split by mechanism, as predicted in advance: the trivial exploit
**+0.007**, the involved one **−0.098**.

**Two deflations, both ours.**
1. A probe scoring 1.00 at "did the reasoning conceal it" was **circular** —
   counting words in that same reasoning scores 0.985.
2. **The system prompt spells out all three exploits step by step** and then
   forbids them. The model discovers nothing. This is closer to
   instruction-following violation than emergent reward hacking, and it makes
   "concealment" a weak construct.

What survives is the 0.811, because what it predicts is not in the prompt, not in
the text yet, and not implied by "RL rewarded hacking."

## 11. Is the model's own basis better than a random one? — **pre-registered**

604 comparisons, 4 models, **written down in advance** with kill criteria
(`docs/PRIVILEGE_PREREG.md`). Take 64 of the model's directions; compare against
random sets spanning the *same subspace*. Coherence judged in a **different
model's** word space.

| | better than random by |
|---|---|
| **MLP neurons** | **+0.0209 [+0.0121, +0.0297]** — a 1.31× advantage |
| **attention head columns** | **+0.0011 [−0.0023, +0.0047]** — *exactly zero* |
| OV-circuit directions | +0.0065 [+0.0033, +0.0097] |
| mixing columns across heads | +0.0339 [+0.0244, +0.0439] |
| residual-stream basis | −0.0023 — **prediction refuted** |

The **zero** is load-bearing: it's the one arm whose answer was known in advance
from the maths, and the instrument wasn't tuned to produce it. Two written-down
predictions came out wrong and are reported as wrong.

## 12. Symmetries — **the answer**

A symmetry is a weight edit that leaves behaviour **exactly** unchanged. If a
method gives a different answer afterwards, it was describing coordinates.

### The six, all verified

| symmetry | the edit | verified |
|---|---|---|
| **transform an attention head's value-output basis** | `W_O → W_O·M`, `W_V → M⁻¹·W_V`, **any invertible M** | \|Δlogit\| 3.4–5.9×10⁻⁵, identical text |
| **rotate a head's query-key basis inside its rotary pairs** | `W_Q, W_K → R·W_Q, R·W_K` with R block-diagonal | \|Δlogit\| 4.6×10⁻⁵ (a *general* R breaks it: 0.56) |
| **rescale an MLP neuron** | output 3× bigger, outgoing weight 3× smaller | \|Δlogit\| 3.9×10⁻⁵ |
| **permute neurons** | shuffle hidden units and matching weights | \|Δlogit\| 3.4×10⁻⁵ |
| **shift the MoE router** | add the same number to every expert's score | same 8 experts, weights differ by 0.00 |
| **absorb the normalisation scale** | fold it into the unembedding (untied models only) | logits differ by 3.8×10⁻⁶ |

### What each method scores (1.00 = unchanged)

| method | head rotation | rescale 9× | rescale 2× | permutation |
|---|---|---|---|---|
| **head-column readout** | **0.00–0.01** | 1.00 | 1.00 | 1.00 |
| OV singular vectors, naive | 0.44–0.50 | 1.00 | 1.00 | 1.00 |
| **OV singular vectors, sign-fixed** | **1.00** | 1.00 | 1.00 | 1.00 |
| **OV subspace** | **1.00** | 1.00 | 1.00 | 1.00 |
| MLP logit lens | 1.00 | 1.00 | 1.00 | **0.00** |
| max-activating examples | 1.00 | 1.00 | 1.00 | **0.05** |
| cross-unit activation ranking | 1.00 | **0.87** | 0.98 | **~0.00** |
| activation × gradient | 1.00 | **1.00** | 1.00 | ~0.00 |
| **residual-stream direction** | **1.00** | **1.00** | **1.00** | **1.00** |

Stable across two extra seeds and a different layer; head-column readout is
0.00–0.01 every time.

### Five things this says

1. **Head-column readouts describe coordinates**, 3/3 models.
2. **The OV circuit is the fix — with a wrinkle.** Its subspace is invariant to
   3.6×10⁻⁷ and per-vector |cos| is 1.0000, but the *signed* cosines are −1, −1,
   −1, +1, −1, −1, +1, −1: the SVD carries its own arbitrary sign, and top-k of
   `−v` is the bottom-k of `v`. Fix the sign by convention and invariance returns.
   (Also: 55 of 63 singular values are nearly tied, so those directions are weakly
   determined anyway.)
3. **Activation × gradient is invariant where raw activation isn't** — activation
   scales by *c*, its gradient by 1/*c*, and the product cancels.
4. **Weight decay silently fixes the neuron-scale freedom.** Its optimum is
   balanced weights and real models sit there: **1.015** (Olmo-3-7B), **1.032**
   (SmolLM2), **1.036** (Qwen2.5-0.5B). Pinned by the regulariser, not by meaning
   — so magnitudes aren't comparable across differently-regularised models.
5. **Forgetting to fold the normalisation scale costs 18–43%** of your readout.


### How big the freedom actually is — *a correction to our own counting*

We had been quoting the size of the attention-head freedom as the **rotation
group**, `d_head(d_head−1)/2` = 2016 parameters for a 64-dimensional head. That
was too small, and asking "is my own group the right one?" showed why.

A head's output depends on `W_O·W_V` only through that **product**. So
`W_O → W_O·M`, `W_V → M⁻¹·W_V` works for **any invertible M**, not just
rotations. Verified: a matrix with condition number **115** leaves the logits
unchanged to 4.6×10⁻⁵.

So the value-output freedom is the **general linear group** — `d_head²` = **4096**
parameters per head, not 2016. The head's internal space has no preferred axes
*and no preferred notion of length or angle either*.

Our "read the OV circuit instead" recommendation survives this, and is
strengthened by it: `W_O·W_V` is exactly what the larger group preserves.

### The other half of the head — and a symmetry that architecture accidentally closed

Attention scores are `(W_Q x)·(W_K y)`, which depends on `W_Q^T W_K`. So the
query-key basis has its own freedom, independent of the value-output one. But
**rotary position embeddings** rotate coordinate *pairs* by position-dependent
angles before that dot product, and a general rotation does not commute with
that.

The framework makes a precise prediction: **only rotations acting inside each
rotary pair should survive.** Tested:

| what we did to the query-key basis | max \|Δlogit\| | |
|---|---|---|
| a general rotation | **0.56** | breaks the model |
| a rotation inside each rotary pair | **4.6×10⁻⁵** | free |

Exactly as predicted. Which means **rotary embeddings remove 98.4% of the
query-key freedom by accident** — from 2016 parameters down to 32 — while the
value-output half has no such protection and keeps all 4096.

Practical consequence: **query-key head dimensions are far better determined
than value-output ones.** If you are going to read something out of a head's
internal basis, that asymmetry tells you which half is safer.

### The corrected totals

Free parameters that no behavioural experiment can resolve:

| model | layers | kv heads | d_head | OV per head | QK per head | **total** | % of attention weights |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 3 | 64 | 4,096 | 32 | 371,520 | 1.40% |
| Qwen2.5-0.5B | 24 | 2 | 64 | 4,096 | 32 | 198,144 | 0.45% |
| Llama-3.2-1B | 16 | 8 | 64 | 4,096 | 32 | 528,384 | 0.31% |
| OLMoE-1B-7B | 16 | 16 | 128 | 16,384 | 64 | 4,210,688 | 1.57% |
| **Olmo-3-7B** | 32 | 32 | 128 | 16,384 | 64 | **16,842,752** | 0.78% |

*Check: `verify.py 11`*

### The consequence: a monitor that breaks

A classifier ("is this code?") trained on the attention layer's internal state,
then run on a model that behaves identically:

| model | monitor on head features | monitor on residual features | model changed by |
|---|---|---|---|
| Llama-3.2-1B | 0.91 → **0.55** | 0.95 → 0.95 | 4.5×10⁻⁵ |
| Qwen2.5-0.5B | 0.83 → **0.42** | 0.97 → 0.97 | 3.3×10⁻⁵ |
| SmolLM2-135M | 0.84 → **0.61** | 0.95 → 0.95 | 3.1×10⁻⁵ |

Same task, same data, same number of features. **The head-based monitor falls to
chance; the residual-based one doesn't move.**

*The first Qwen run showed the model "changing" by 0.93 — Qwen has attention
biases and our rotation hadn't rotated them. The script now prints the model's own
change on every row and flags the row if it isn't tiny.*

### The method that passes: sparse autoencoders

Two **independently trained** SAEs for Llama-3.2-1B — EleutherAI's (131k features,
MLP output) and huypn16's (65k, residual stream):

**100.0% of features kept, all three symmetries, both autoencoders.** Largest
activation change 1.4×10⁻⁶.

And their own freedom turns out to be closed. Scale a feature's detector up and
its output direction down and nothing changes — *if* the activation is a plain
ReLU. Both obey the unit-norm convention exactly (mean 1.0000, spread 0.0000).
Drop it and the two architectures come apart:

- **plain autoencoder:** reconstruction identical (6.7×10⁻⁶), but "which features
  are strongest" keeps only **46.9%**. The freedom is real.
- **top-*k* autoencoder** (what both are): the reconstruction genuinely changes
  (0.21), because keeping the strongest 32 features means comparing features
  against each other. **Top-*k* pins the scale by construction.**

SAEs come out of the audit best: immune to every symmetry of the model, and in
top-*k* form, immune to their own.

*Check: `verify.py 1 2 3 5 9 10`*

---

# PART C — Nine traps, each one we fell into

1. **The label was already in the text.** A probe scored 1.00; counting words in
   the same text scored **0.985**.
2. **Chance is the wrong thing to beat.** The bar is what someone reading the
   transcript gets.
3. **An average can hide the thing you care about.** A confidence gap vanished
   (effect size −0.01) once we looked at the *first* token instead of the mean.
4. **Position.** Same direction, two words apart, sign flips.
5. **One dose is not a measurement.** Efficacy-vs-precision correlation reverses
   (−0.47 → +0.24).
6. **Shared starting points look like agreement.** Similarity **0.9994** before
   any training.
7. **"Shared" and "biggest" can be the same thing.** 0.941 vs 0.937 — not
   distinguishable with three models.
8. **Beating a random *orthogonal* set is easy.** Almost any non-orthogonal set
   does; the real test is a random set with the **same geometry**. That distinction
   overturned the router result.
9. **One outlier token can own your variance.** "99.8% of variance in one
   direction" at layer 12 becomes **8.5%** when you drop a single token — the
   attention sink at position 0, whose norm is 140× the median. We got this
   wrong twice: once in the original claim, once in the correction that blamed
   centring. Always report whether the sink token is in your sample.

Plus two of our own retractions worth naming: a claimed replication that
disappeared under the audit rule that produced it (157 → 0), and a "specificity
fails" table that turned out to be measured at a saturated dose *and* at the wrong
token position.

---

# PART D — What I'd write

**Lead with the symmetry result.** Only finding with ground truth known in
advance; replicates on three model families and across seeds; one-sentence
headline: *rotate an attention head and the model's output moves by four
hundred-thousandths of a logit while 99% of what you'd read out of that head
changes.*

**Then the monitor and the SAE audit together** — the "so what". A classifier on
head features collapses to chance on an identical model; two real sparse
autoencoders score 100%. That turns the finding from a curiosity into a reason to
prefer one method over another.

**Then the router**, as evidence of judgement: it overturned one of our own
conclusions and revealed an effect the artefact was hiding.

**Then the pre-registered study**, as the calibration story.

**Then the trap list as the body, not an appendix.**

**Do not claim** a discovery about what models believe or do. Do not sell the
reward-hacking arm as emergent misalignment — the system prompt hands the model
the recipes.

---

# PART E — Limits

- **Passing the test doesn't make a method right.** It rules out; it doesn't certify.
- **Five symmetries, not all of them.**
- **Two SAEs, not a survey** — both top-*k*, one model family.
- **Small models** for most of it (135M–1B); 7B and 9B for two arms. The
  symmetries are architectural so they hold at any size; the *magnitudes* were
  measured here.
- **The steering specificity question is open**, not answered. It needs the dose
  sweep and the headroom control together.
- **The training-dynamics curve** is a similarity-to-final measure, so it must end
  at 1.0. The content is the shape, not the endpoints.

---

# PART F — How to check any of it

```
PYTHONPATH=. .venv/bin/python verify.py        # all eleven, ~10 minutes
PYTHONPATH=. .venv/bin/python verify.py 3      # just one
```

| | prints |
|---|---|
| 1 | the rotation is free — identical text, logits move 4×10⁻⁵ |
| 2 | and it replaces 99%+ of the head's readout, two models |
| 3 | the OV circuit survives exactly — and its signs flip |
| 4 | 38.3% of the MoE router does nothing |
| 5 | weight decay already fixed the neuron-scale freedom |
| 6 | gradient steering beats embedding-difference beats random |
| 7 | the chess piece subspaces are interchangeable |
| 8 | position flips the sign of a steering effect |
| 9 | a head-based monitor falls to chance; a residual one doesn't |
| 10 | two real sparse autoencoders survive every symmetry, 100% |
| 11 | the freedom is the full invertible group; rotary closes 98.4% of the other half |

Nothing reads from a saved result. Everything recomputes from the models on disk.

**Where things live:** `gauge/` (symmetries), `jlens/` (steering, chess, MoE,
training dynamics, the pre-registered study), `rh/` (reward hacking), `em/`
(misalignment organisms), `docs/PRIVILEGE_PREREG.md` (written before the numbers
existed), `out/gauge/readouts.json` (raw before/after token lists),
`out/rh/index.jsonl` (300 trajectories: prompt, reasoning, code, labels).

**Models used:** SmolLM2-135M, Qwen2.5-0.5B, Llama-3.2-1B, Qwen3.5-4B,
OLMoE-1B-7B, Olmo-3-7B, Qwen3.5-9B, ChessGPT, plus 11 OLMo training checkpoints.
183 analysis scripts, 74 commits. Everything runs on a 34 GB laptop; the central
result needs no GPU.
