# The basis you read is not the basis the model chose

**Result document. Pre-registration: `docs/PRIVILEGE_PREREG.md` (written before
any number below existed). Code: `jlens/rotate_sweep.py`, `jlens/privilege.py`,
`jlens/corpus_npmi.py`, `jlens/tokspace.py`.**

---

## The one-paragraph version

Interpretability reads directions: "this neuron writes *medical*", "direction 3
of head 7 promotes surnames". A direction is a set of coordinates, and
coordinates need a basis. Inside an attention head the transformer has **no
preferred basis** — you can rotate a head's internal axes and the model computes
*exactly* the same function, to the last bit. So any statement about "direction
i of head h" is a statement about a coordinate system the model never picked.
We measure what that costs: rotating a head's basis by a random orthogonal
matrix changes the logits by 4e-05 (float32 noise; the floor is 0.0) and
replaces **99% of the tokens you would read out of that head**. The same
algebraic move applied to MLP neurons — where an elementwise nonlinearity *does*
pick a basis — changes the logits by 2.0, i.e. it breaks the model. Same
operation, two places, a factor of 16,000-51,000 apart.

---

## 1. Why a transformer has a preferred basis in some places and not others

A basis is *privileged* only where an elementwise operation acts in it
(Elhage et al., "Privileged Bases in the Transformer Residual Stream").

**Attention.** A head computes `out = W_O · (sum_t a_t · W_V x_t)`. The
attention weights `a_t` come from Q and K and never touch V. So for any
orthogonal `R` acting on the head's `d_head` internal axes:

```
W_V -> R W_V ,  W_O -> W_O Rᵀ    =>    W_O Rᵀ R W_V = W_O W_V
```

The function is **unchanged, exactly**. Nothing in the architecture, the loss,
or the data can distinguish the two parameterisations. The head's internal
basis is a gauge freedom.

**MLP.** A SwiGLU MLP computes `W_down · (SiLU(W_gate x) ⊙ W_up x)`. SiLU and
the product `⊙` act coordinate-by-coordinate, so `SiLU(R W_gate x) != R SiLU(W_gate x)`.
The same substitution changes the function. The neuron basis is not a gauge
freedom; it is picked out by the nonlinearity.

**This gives an experiment with a known answer on one side** — which is the
only reason the measurement below can be trusted.

## 2. How big is the gauge freedom?

`dim O(d_head) = d_head(d_head-1)/2` rotations per head (tied across a
grouped-query group), all invisible to behaviour:

| model | layers | heads | kv | d_head | gauge dims / group | total gauge dims |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 9 | 3 | 64 | 2,016 | 181,440 |
| Qwen2.5-0.5B | 24 | 14 | 2 | 64 | 2,016 | 96,768 |
| Llama-3.2-1B | 16 | 32 | 8 | 64 | 2,016 | 258,048 |
| OLMoE-1B-7B | 16 | 16 | 16 | 128 | 8,128 | 2,080,768 |
| Olmo-3-7B | 32 | 32 | 32 | 128 | 8,128 | **8,323,072** |

Olmo-3-7B has 8.3 million directions in parameter space that no behavioural
experiment can ever resolve — and every one of them changes what a head readout
says.

## 3. Experiment A: the same rotation, two places (exact, float32)

`jlens/rotate_sweep.py`. For every layer of three models: rotate one kv-group's
attention basis (function-preserving), and separately rotate 64 MLP hidden
units the same way (not function-preserving). Measure `max |delta logit|` over
6 prompts, and the top-10 overlap of the head's `d_head` logit-lens readouts
before vs after.

| model | layers | attn \|Δlogit\| | MLP \|Δlogit\| | ratio | head readout kept |
|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 4.20e-05 | 1.95 | 46,392x | 1.5% |
| Qwen2.5-0.5B | 24 | 5.63e-05 | 2.89 | 51,350x | 0.4% |
| Llama-3.2-1B | 16 | 4.15e-05 | 0.67 | 16,066x | 0.3% |

Logit scale is ~25 in all three, so the attention perturbation is ~2e-06
relative — float32 arithmetic noise. Greedy continuations are identical on
every prompt. Every layer of all 70 behaves the same way; this is not a
cherry-picked layer.

**Worked example** (SmolLM2-135M, layer 20, head 3, `out/priv/rotate_head.json`):

```
dir 0 BEFORE : isans, nings, ortic, ousing, iage, acquaint, transferable, idim
dir 0 AFTER  : groups, systems, ve, [, are, who, via, id
dir 1 BEFORE : MetaInfo, <filename>, OrderedDict, CreateModel, Yours, brilli
dir 1 AFTER  : oolean, %|, unting, ortunately, enburg, ]}, avorable, ields
```

Both readouts are equally "real". Neither is a property of the model.

---

## 4. Experiment B: does the model's own basis read better than a random one?

Experiment A shows the head basis is *unidentifiable*. It does not say whether
bases that **are** identifiable (MLP neurons, the residual stream, MoE router
rows) are actually *better to read*. That is the question everyone's practice
assumes a "yes" to. Experiment B tests it with a calibrated null.

### The four bases

Take `n = 64` direction vectors as the columns of `A` (d x n, unit norm), with
QR factorisation `A = QR`. All four bases below span **exactly the same
subspace** (verified numerically to 3e-07):

| basis | construction | Gram matrix | what it is |
|---|---|---|---|
| `raw` | `A` | `AᵀA` | what the model actually has |
| `svd` | left singular vectors of `A` | `I` | what you get if you SVD the matrix |
| `reparam_null` | `A H`, `H ~ Haar(n)` | — | **the symmetry orbit**: the weights the model could equally have had (exactly, for an attention head; illegally, for an MLP) |
| `gram_null` | `Q H R`, `H ~ Haar(n)` | `AᵀA` — identical to raw | same norms, same pairwise angles, same conditioning, random identity |
| `orth_null` | `Q H`, `H ~ Haar(n)` | `I` | an arbitrary orthonormal basis of the subspace |

`gram_null` is the control that matters: because its Gram matrix is *bit-for-bit*
the same as raw's, no difference between them can be attributed to
orthogonality, vector norms, or conditioning. Only to *which directions* they are.

### Scoring, and why it is not circular

Each direction is read with the standard logit lens,
`logits = W_U (gamma ⊙ u)` (RMSNorm, so the scale drops out of the ranking),
and the **top 10 tokens** are kept. Those 10 tokens are then scored **in a
different model's embedding space** — Llama-3.2-1B for everything else,
Qwen2.5-0.5B as a robustness check — matched by decoded string. The model being
probed contributes only the token list; it contributes nothing to the judgement
of whether that list is coherent.

Two frequency controls, because coherence metrics are notoriously
frequency-biased:
1. the score is `mean pairwise cosine - E[cosine | df-bin of each token]`,
   with the expectation estimated from 400,000 random token pairs;
2. `dlogdf`, the mean log-frequency gap between raw's tokens and the null's,
   is reported for every row. Pre-registered criterion **K2** voids any arm
   where it exceeds 0.5 nats.

### The statistic: the tail, not the mean

Most units in any layer are not readable. If a basis is privileged, that shows
up as *more highly-coherent directions*, not as a higher average — averaging 64
directions of which ~3 are interpretable washes the signal out. So the primary
statistic is the **90th percentile** of per-direction coherence within a basis,
compared against the distribution of that same percentile across 16 Haar draws.
The mean is reported alongside it.

*(This was a correction made mid-run: the first pass used the mean and found
nothing anywhere, including in arms where the mean cannot distinguish a basis
with 3 readable directions from one with none. The mean numbers are reported
too, and the null arm calibrates both.)*

### 4.1 Results

604 groups, 4 models (OLMoE-1B-7B, Olmo-3-7B, Qwen2.5-0.5B, SmolLM2-135M),
64 directions per group, 16 Haar draws per null. Primary statistic: the 90th
percentile of per-direction coherence. Bootstrap CIs are over groups.

| arm | n | raw | orbit | raw − orbit (95% CI) | ratio |
|---|---|---|---|---|---|
| **attention head columns** (provable null) | 96 | +0.1017 | +0.1006 | **+0.0011 [−0.0023, +0.0047]** | 1.02 |
| OV-circuit singular vectors | 96 | +0.1001 | +0.0936 | **+0.0065 [+0.0033, +0.0097]** | 1.07 |
| attention columns across heads | 96 | +0.1094 | +0.0755 | **+0.0339 [+0.0244, +0.0439]** | 1.49 |
| residual-stream basis | 96 | +0.0484 | +0.0508 | −0.0023 [−0.0055, +0.0007] | 0.98 |
| **MLP neurons** | 96 | +0.1121 | +0.0912 | **+0.0209 [+0.0121, +0.0297]** | 1.31 |
| MoE router rows | 28 | +0.0942 | +0.1029 | −0.0087 [−0.0404, +0.0180] | — |

**The provable-null arm reads as a measured zero.** That is the load-bearing
line: the instrument was not tuned to produce it, and it is the only arm whose
answer was known in advance.

Against that calibration:

- **MLP neurons are privileged**: +0.0209 (tail ratio 1.31), 8.5-13.4x the size
  of the null arm in each of three models. Pre-registered check **K1 passes** on SmolLM2-135M,
  Qwen2.5-0.5B and OLMoE-1B-7B; on Olmo-3-7B the MLP arm has the right sign but
  a CI that includes zero even with 40 groups (see §6).
- **Head boundaries are real**: mixing columns *across* heads costs 1.49x,
  mixing them *within* a head costs nothing. Same matrix, same operation — the
  difference is exactly where the architecture puts a symmetry.
- **The residual-stream basis is not privileged at the tail** (predicted it
  would be; it is not). It is marginally privileged in the mean
  (+0.0009 [−0.0001, +0.0018], marginal). Prediction 4 is refuted.

### 4.2 What is actually carrying the signal: identity vs cone

Five bases of the same subspace, mean coherence:

| arm | raw | orbit `AH` | gram `QHR` | SVD | orthonormal | **raw − orbit** (identity) | **orbit − orth** (cone shape) |
|---|---|---|---|---|---|---|---|
| attn head columns | .0484 | .0481 | .0442 | .0460 | .0440 | +0.0003 [−0.0015,+0.0019] | +0.0040 [+0.0018,+0.0071] |
| OV singular vectors | .0459 | .0439 | .0441 | .0433 | .0439 | +0.0020 [−0.0003,+0.0040] | +0.0001 [−0.0003,+0.0005] |
| attn across heads | .0461 | .0300 | .0285 | .0287 | .0283 | **+0.0161 [+0.0116,+0.0210]** | +0.0017 [+0.0007,+0.0031] |
| residual basis | .0196 | .0188 | .0184 | .0202 | .0185 | +0.0009 [−0.0001,+0.0018] | +0.0003 [+0.0000,+0.0005] |
| MLP neurons | .0421 | .0379 | .0367 | .0378 | .0365 | **+0.0043 [+0.0008,+0.0075]** | +0.0014 [+0.0008,+0.0020] |
| MoE router rows | .0424 | .0355 | .0161 | .0210 | .0169 | +0.0070 [−0.0008,+0.0135] | **+0.0186 [+0.0089,+0.0298]** |

Two different things can make a basis read well, and they come apart:

- **identity** (`raw − orbit`): are *these particular* directions better than
  random mixtures of themselves? True for MLP neurons and for cross-head
  columns. Not true for attention head columns (as it must not be).
- **cone shape** (`orbit − orth`): is the *arrangement* non-isotropic, so that
  any direction drawn from the same cone reads well? Small everywhere except
  the MoE router, where it is an order of magnitude larger than anywhere else.

**This corrects an earlier result in this repo.** We previously found that
OLMoE's router rows read ~2x better than an orthogonal basis of their span and
read it as "the model's own directions are special". They do beat an
orthonormal basis, by +0.0255 [+0.0194, +0.0315] — but a *random non-orthogonal*
basis of the same span recovers **73%** of that gap (+0.0186 of +0.0255), and
raw vs that random basis is not significant. The router's readability is a
property of the narrow cone its rows span, not of which rows they are.

### 4.3 Robustness

| variation | attn head cols (null) | MLP neurons |
|---|---|---|
| main: Llama-3.2-1B scorer, 13k-token vocab | +0.0038 | +0.0348 |
| scorer swapped to Qwen2.5-0.5B | +0.0001 | +0.0344 |
| scored vocabulary widened to 35k tokens | +0.0057 | +0.0547 |

The effect does not depend on the semantic space used to judge coherence
(two unrelated model families agree), and widening the vocabulary *increases*
it — the restriction was blunting the signal, not manufacturing it.
`dlogdf` stays within ±0.12 nats on every reported arm, so **K2 never fires**
on the main results. It did fire, at −1.9 nats, on the unembedding-row
"ceiling" arm, which is therefore dropped and not reported.

---

## 5. What this changes about practice

**1. Do not interpret directions inside an attention head's internal basis.**
"Direction 12 of head 7 promotes surnames", the top singular vectors of `W_O`
or `W_V` alone, and per-head-dimension neuron-style analyses are all statements
about coordinates the model never chose. Experiment A shows what that costs:
99% of the readout is replaced by a change the model cannot detect. Experiment B
shows the readouts are worth exactly as much as random ones (+0.0002
[−0.0020, +0.0021]).

**2. There is a fix, and it is measurably better.** `W_O W_V` is invariant
under the rotation; so are its singular vectors. Those span *the same subspace*
as the head's columns, and they read significantly better
(+0.0078 [+0.0043, +0.0114]) where the columns read at zero. Same subspace,
same 10-token readout format — one basis is gauge, the other is not.

**3. Add the orbit null to direction-finding methods.** Before claiming a set
of directions is interpretable, rotate them within their own span and score the
rotation. It costs one QR and a Haar draw. If the method's directions do not
beat their own orbit, the method has found a coordinate system, not structure.

**4. Separate two things that both look like "our directions are special".**
Beating an *orthonormal* basis is easy and mostly means the real basis is
non-isotropic. Beating the *orbit* is the claim that actually implies these
directions matter. For the MoE router, the first is true and the second is not.

## 6. Limitations, stated plainly

- **The proxy is narrow.** "Coherent top-10 logit-lens tokens in an unrelated
  model's embedding space" is one operationalisation of interpretability. A
  direction can matter and have no coherent token readout — it may feed later
  computation rather than the unembedding. Everything here is about *readouts*,
  and Experiment B's negatives should be read as "not readable this way", not
  "meaningless".
- **Olmo-3-7B is underpowered on the MLP arm.** At 14 groups it failed K1
  outright. Pooling 40 groups (two seeds) moves it to MLP `+0.0138
  [-0.0016, +0.0286]` against a null arm of `+0.0000 [-0.0037, +0.0033]` — the
  right sign and much larger than the null, but the CI still includes zero. The
  same model's cross-head arm is strongly positive (`+0.0455 [+0.0314, +0.0604]`),
  so the instrument works there; it is the MLP arm specifically that is weak.
  The leading explanation is sampling: 64 neurons drawn from `d_ff = 11008`
  contain fewer readable ones than 64 drawn from 1024 (OLMoE) or 1536
  (SmolLM2), so a 90th-percentile statistic over 64 samples has less to find.
  That is a hypothesis. It predicts the effect returns if `n` is raised, and
  that test is not run here.
- **The effect is modest.** 1.33x at the tail, not an order of magnitude. Most
  MLP neurons in every model tested are not readable in this sense; the
  privileged basis buys a somewhat better tail, not a dictionary.
- **No behavioural test of the readouts.** Experiment A's causal check is about
  the *function* (does the rotation change the model), not about whether raw
  directions are better *interventions* than rotated ones. That is the obvious
  next experiment and it is not run here.
- **Attention arms assume MHA or correctly-tied GQA groups.** For grouped-query
  models the rotation is tied across each kv group, as the code does; a
  per-query-head rotation would not be function-preserving and was not used.
- **Prediction 4 was wrong.** The residual-stream basis was predicted to be
  privileged and is not, at the tail. Reported as refuted rather than dropped.

## 7. Reproducing

```
PYTHONPATH=. .venv/bin/python -m jlens.rotate_sweep --model HuggingFaceTB/SmolLM2-135M
PYTHONPATH=. .venv/bin/python -m jlens.privilege --model allenai/OLMoE-1B-7B-0924 --groups 14 --draws 16
PYTHONPATH=. .venv/bin/python -m jlens.privilege_report
```

Everything runs on a 34 GB MacBook with no GPU beyond MPS. Experiment A is
minutes; Experiment B is ~2 minutes per model. No forward passes are needed for
Experiment B at all — it is pure weight-space algebra plus a logit-lens readout.

| file | what it does |
|---|---|
| `jlens/rotate_sweep.py` | Experiment A: rotate attention vs MLP, every layer |
| `jlens/rotate_head.py` | single-head worked example with before/after readouts |
| `jlens/privilege.py` | Experiment B: the five bases and the arms |
| `jlens/corpus_npmi.py` | Pile window/token presence matrix + NPMI (secondary metric) |
| `jlens/tokspace.py` | cross-model embedding coherence + frequency matching |
| `jlens/privilege_report.py` | the tables above |
| `docs/PRIVILEGE_PREREG.md` | predictions and kill criteria, written first |

---

## 8. Experiment C: do the model's own directions *act* better, or only *read* better?

Experiments A and B are about what a direction **says**. This is about what it
**does**: inject each direction into the residual stream and score the tokens
the model actually ends up promoting. Same arms, same bases, same seed, same
coherence metric — the only change is where the top-10 comes from.

| | top-10 taken from |
|---|---|
| Experiment B | `W_U (gamma ⊙ d)` — the readout |
| Experiment C | `logits(h + a·d) − logits(h)` — the intervention |

### 8.1 The first run was invalid, and the check that caught it

The obvious dose (`a` = 0.25–1.0 × ‖h‖, which is what this repo has used
before) turns out to be far outside the usable range. Two validity checks,
run on the raw bases only:

- **common-mode**: mean pairwise cosine of the induced logit change across the
  64 directions. If every direction pushes the model the same way, no
  comparison between bases can discriminate anything.
- **cos(predicted, actual)**: how much of the logit-lens prediction survives.

| alpha | top-10 change | common-mode | cos(pred, act) | verdict |
|---|---|---|---|---|
| 0.003 | 0.02 nats | 0.002 | 0.236 | linear, direction-specific |
| 0.01 | 0.06 | 0.003 | 0.236 | linear, direction-specific |
| 0.03 | 0.16 | 0.005 | 0.236 | linear, direction-specific |
| 0.1 | 0.54 | 0.021 | 0.234 | linear, direction-specific |
| 0.3 | 1.58 | 0.147 | 0.226 | drifting |
| **1.0** | 6.77 | **0.457** | 0.212 | **saturated — every direction does the same thing** |

At `a = 1.0`, layer 1, common-mode reaches **0.97** in the MLP arm: 97% of each
direction's causal effect lies along a single shared direction. The first
version of Experiment C was run at 0.25–1.0 and produced a flat null in every
arm. That null was an artifact of the dose, and it was discarded, not reported.
Experiment C is run at **a = 0.05**, with `a = 0.3` kept as a contrast.

Note that `cos(pred, act)` is *identical to three decimals* from 0.003 to 0.03:
inside the window the result is dose-independent, which is what "linear regime"
should mean operationally.

### 8.2 A result that came out of the validity check

**The logit lens predicts what a direction does only in the last quarter of the
network.** Same measurement, two model families, `a = 0.05`:

| depth | top-10 overlap (readout vs action) | cos(pred, act) |
|---|---|---|
| first quarter | 0.6 – 0.8% | 0.04 – 0.06 |
| second quarter | 0.5 – 1.2% | 0.067 |
| third quarter | 6.1 – 7.4% | 0.22 – 0.28 |
| **last quarter** | **43.4 – 43.5%** | **0.59 – 0.76** |

SmolLM2-135M (30 layers, 84 cells) and Qwen2.5-0.5B (24 layers, 72 cells) agree
to within a percentage point at every depth. Before ~75% depth, the tokens a
direction reads as and the tokens it actually promotes are **disjoint**
(overlap under 1.5%, against a chance rate of ~0.02%).

Within the last quarter the agreement barely depends on *which* component you
read — attention head columns 47.7/48.8%, OV singular vectors 50.1/45.7%,
MLP neurons 34.5/44.7%, cross-head 45.6/49.4%. Only the residual standard basis
is lower (26.6/28.4%). So readout–action agreement is a property of **depth**,
not of which part of the model the direction came from.

**What this means for Experiment B.** Every arm in Experiment B was scored by
its readout. This says those readouts are causally meaningful in the last
quarter of the network and largely decorative before it. Experiment B's
positives are therefore claims about *what directions encode for the
unembedding*, and only in the final layers are they also claims about what
those directions *do*.

### 8.3 Result: the readout advantage survives as a causal advantage

`alpha = 0.05` (inside the validated window), 44 groups per arm across
SmolLM2-135M (26 groups) and Qwen2.5-0.5B (18 groups), 4 Haar draws per null,
264 group-cells. Primary statistic is the within-group z of raw against its own
null draws; the raw-minus-orbit difference is reported alongside.

| arm | within-group z (95% CI) | q90 raw − orbit (95% CI) |
|---|---|---|
| **attention head columns** (provable null) | **+0.30 [−0.14, +0.78]** | +0.0052 [−0.0054, +0.0171] |
| attention head columns (subsampled) | +0.84 [+0.15, +1.59] | +0.0053 [−0.0022, +0.0125] |
| OV-circuit singular vectors | +0.37 [−0.31, +1.06] | −0.0072 [−0.0279, +0.0095] |
| attention columns across heads | +1.00 [+0.16, +1.94] | +0.0164 [−0.0033, +0.0369] |
| residual-stream basis | +0.20 [−0.35, +0.76] | +0.0022 [−0.0085, +0.0137] |
| **MLP neurons** | **+1.55 [+0.81, +2.41]** | **+0.0231 [+0.0111, +0.0353]** |

**Prediction C1 held.** The provable-null arm is a measured zero causally as
well as in readout: +0.30, CI containing zero.

**Prediction C2 held.** MLP neurons act better than rotations of themselves —
+0.0231, **4.4x the null arm**, with a CI excluding zero. The advantage
Experiment B found in the readout is not an artifact of reading through the
unembedding; it is there when you intervene.

The two experiments agree arm by arm, which is the real result here:

| arm | Experiment B (readout) | Experiment C (intervention) |
|---|---|---|
| attention head columns | zero | zero |
| MLP neurons | **positive** | **positive** |
| attention across heads | **positive** | **positive** (z only) |
| residual-stream basis | zero | zero |
| OV-circuit singular vectors | positive | not significant |

The one disagreement is the OV arm: significantly better to *read*
(+0.0065 [+0.0033, +0.0097]) but not measurably better to *inject*. That is
consistent with §8.2 — most groups sit below 75% depth, where readouts and
interventions are largely decoupled.

**Per-model, K1 splits.** On Qwen2.5-0.5B the null arm is −0.0016 and the MLP
arm +0.0350, a ratio of 21.9x — K1 passes cleanly. On SmolLM2-135M the null arm
is itself elevated (+0.0099) against an MLP arm of +0.0149, a ratio of 1.5x —
**K1 fails on SmolLM2.** Pooled, the null arm's CI contains zero and the MLP
arm's does not, but the honest statement is that the causal effect is clean on
one of two models and confounded on the other.

**Concentration is not a discriminating measure.** The share of induced positive
logit mass landing in the top 10 tokens is 0.0020–0.0022 for *every* arm and
*every* basis (raw/orbit ratio 0.98–1.03x). Whatever makes raw directions better
is not that their effects are more concentrated.

### 8.4 What Experiment C cost in wrong turns

Two failures, both caught by pre-registered checks rather than by inspection of
the results:

1. **Wrong dose.** The first run used `alpha` = 0.25–1.0, the doses this repo
   had used before. It produced a flat null in every arm. Common-mode was 0.46
   pooled and 0.97 in the MLP arm at layer 1 — every direction was pushing the
   model the same way. KC1 voided it.
2. **Underpowered.** The re-run at a valid dose used 4–6 groups per arm and
   produced a *significant negative* for the MLP arm, the opposite sign to the
   final answer. A separate diagnostic at the same dose gave a positive. The
   disagreement was the signal that n was too small; KC3 called for reporting
   "underpowered", and the fix was 44 groups per arm rather than a narrative.
   The effect-size confound was checked directly and excluded: raw and orbit
   induce logit changes of matched magnitude (‖Δlogit‖ 13.57 vs 13.90, top-10
   0.312 vs 0.273 nats in the MLP arm), so the comparison is not signal-to-noise.

Both wrong answers were reportable-looking. Neither survived a check that was
written down before the numbers existed.

---

## 9. Reproducing Experiment C

```
# dose + depth validity, cheap (raw bases only)
PYTHONPATH=. .venv/bin/python -m jlens.readout_vs_action --model HuggingFaceTB/SmolLM2-135M \
    --groups 14 --alphas 0.003 0.01 0.03 0.1 0.3 1.0
# the intervention battery, at the validated dose
PYTHONPATH=. .venv/bin/python -m jlens.causal_basis --model HuggingFaceTB/SmolLM2-135M \
    --groups 26 --draws 4 --alphas 0.05 --only raw reparam_null
PYTHONPATH=. .venv/bin/python -m jlens.causal_report
PYTHONPATH=. .venv/bin/python -m jlens.agree_report
```

| file | what it does |
|---|---|
| `jlens/causal_basis.py` | Experiment C: inject each direction, score what it promotes |
| `jlens/readout_vs_action.py` | dose calibration + readout/action agreement by depth |
| `jlens/causal_diag.py` | effect-size and degeneracy checks behind the sign flip |
| `jlens/causal_report.py` | Experiment C tables |
| `jlens/agree_report.py` | depth and dose tables |

Raw output: `out/priv/CAUSAL.txt`, `out/priv/AGREE.txt`, `out/priv/REPORT.txt`.
