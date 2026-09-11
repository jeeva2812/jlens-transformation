# Superseded symmetry-led draft

This draft is retained for its detailed symmetry derivations, but it is no
longer the recommended application guide. It overweights head-internal gauge
freedom even though the main J-Lens experiments operate on the residual stream.
Use `docs/APPLICATION_RESEARCH_GUIDE.md` for the repository-wide assessment and
application strategy. Treat the material below as an optional appendix.

# Understanding `jlens-transformation`

## A first-principles study guide for building an honest, technically defensible MATS application

This is a learning document, not application prose. Its purpose is to help you
understand the project well enough that you can write the application yourself,
answer skeptical questions without bluffing, and distinguish results supported
by the code from attractive claims that are still uncertain.

The most important rule is simple:

> Do not submit a sentence merely because it appears in this repository. Submit
> it only after you can explain the mechanism, the measurement, the control, and
> the main alternative explanation in your own words.

## Strategic correction: the head-only result is not broad enough

The first version of this guide recommended leading with **gauge freedom inside
attention heads**. That recommendation needs an important correction.

Most current interpretability signals are read from the residual stream, MLP
outputs, or learned features such as SAE decoder directions—not from individual
value coordinates inside an attention head. A head value/output rotation leaves
the residual activation unchanged by construction. Therefore:

- the residual monitor's stability under that transformation is expected;
- the SAE stability experiment, when the SAE reads an unchanged downstream
  activation, is primarily a positive control;
- a fixed classifier failing after its input coordinates are rotated without
  transforming the classifier is an illustrative coordinate-mismatch example,
  not evidence that common residual-stream monitors are broadly unsafe; and
- the result directly rules out intrinsic interpretations of individual
  head-internal value/output axes, but does not directly rule out residual probes,
  residual SAEs, steering vectors, or MLP-output features.

This makes the head symmetry valuable as a **calibration arm**, but too narrow to
carry the practical motivation by itself.

The strongest revised application question is:

> When an interpretability method finds directions in the residual stream, are
> the individual directions special, or is only their shared subspace carrying
> the signal?

The existing orbit-null machinery is unusually well suited to this question. It
can compare real residual-derived direction families—SAE decoder features,
J-Lens directions, supervised probe subspaces, PCA directions, gradient
pullbacks, and MLP write directions—with random rotations spanning the exact
same subspace. The attention-head arm remains useful because it supplies a
known-null calibration, not because head-coordinate monitoring is the dominant
interpretability paradigm.

Until those residual-derived arms are run, there are two honest submission
options:

1. present the gauge work narrowly as a methodological note about
   non-identifiable head-coordinate claims and orbit-null calibration; or
2. lead with the broader J-Lens red-team, where the measurements and
   interventions are already residual-stream based, and treat gauge freedom as
   one lesson that motivated a better control.

The best potential application is a third option: extend the calibrated orbit
null to the residual feature families researchers actually use. That is now the
recommended research direction in this guide.

The project began as a broad investigation of J-Lens. The gauge-freedom work
still contributes several valuable ingredients:

1. it has a ground-truth answer known before running the experiment;
2. it identifies a concrete failure mode rather than only a vague limitation;
3. it demonstrates an operational consequence for monitoring;
4. it proposes a cheap control other researchers can actually use; and
5. it includes failed predictions and corrections, which demonstrate research
   judgment rather than merely engineering output.

The rest of the repository remains useful. It explains how the question was
found, supplies supporting experiments, and demonstrates a record of correcting
mistakes. It should not compete with the main story in the first pages of the
application.

---

## 1. The entire project in one page

Mechanistic interpretability often begins with a vector inside a neural network.
A researcher obtains a direction, projects it through the model's output matrix,
sees tokens such as `hospital`, `doctor`, and `patient`, and calls the direction
"medical." This kind of evidence feels concrete, but there is a hidden question:
did the model itself choose that direction, or did the parameterization merely
choose an arbitrary coordinate system?

The project creates a case where the answer is known mathematically. Inside the
value/output half of an attention head, we can replace the internal coordinates
with any invertible coordinate system and compensate in the output matrix. The
two changes cancel. In exact arithmetic, every input produces exactly the same
output. In the implementation, the largest observed logit difference is around
`4e-5` on logits of magnitude roughly `25`, consistent with floating-point
roundoff, and greedy generations remain unchanged.

The model is therefore functionally unchanged. Yet a conventional readout of an
attention head's individual columns loses about 99% of its top-ten tokens after
the coordinate change. Both the original and transformed token lists are equally
compatible with the exact same model behavior. The token interpretation was
therefore a property of coordinates, not a uniquely determined property of the
model.

This leads to three deeper results.

First, the correct invariant object is not an output column in isolation. It is
the combined value-output transformation, often called the **OV circuit**. Its
subspace and singular values survive the coordinate change. Individual singular
vectors still have a sign ambiguity, and nearly degenerate singular values can
make an individual basis unstable, so the subspace is the most defensible object.

Second, the issue matters operationally. A linear monitor trained to distinguish
code from English using head-internal features performs well in the original
coordinates and then falls toward chance after a function-preserving rotation.
A monitor trained on the residual stream does not move. A new 32-seed robustness
sweep added for this study guide makes the result harder to dismiss as a lucky
rotation: on SmolLM2-135M, held-out token accuracy falls from `0.840` to a mean of
`0.495` across rotations, while the residual control remains `0.951`. The worst
logit change across all 32 transformed models is only `4.20e-5`.

Third, the project introduces an **orbit null**. Given a set of candidate
directions spanning a subspace, randomly rotate those directions within the same
subspace and score them again. If the original directions do not outperform
their own rotations, the evidence supports the subspace but not the identity of
the individual directions. In the current results:

- attention-head columns return the mathematically required null result;
- MLP neurons outperform their rotated alternatives;
- mixing columns across different attention heads loses information, suggesting
  that head boundaries are architecturally meaningful even though coordinates
  inside a head are not;
- residual-stream coordinate axes do not show the predicted advantage at the
  tail of the score distribution; and
- centering an MoE router by removing a functionally irrelevant shared component
  overturns an earlier explanation and exposes a different effect.

The responsible conclusion is not "everything invariant is interpretable." A
method can survive all tested symmetries and still measure nothing useful. The
conclusion is one-directional:

> Failing a valid symmetry test is evidence that the interpretation depends on
> arbitrary coordinates. Passing the test only means the method survives that
> particular falsification attempt.

---

## 2. Repository map: what is central and what is a side branch

The repository has grown through exploration. There are hundreds of scripts,
many saved results, and several generations of reports. It is better understood
as a laboratory notebook than as a conventional software package.

### 2.1 The application spine

| File or directory | Role |
|---|---|
| `gauge/matrix.py` | Implements the main function-preserving transformations and asks which interpretability outputs survive. |
| `jlens/rotate_sweep.py` | Compares the same rotation inside attention, where it is a symmetry, with an MLP rotation, where it breaks the model. |
| `jlens/privilege.py` | Constructs raw, orbit, Gram-matched, SVD, and orthonormal bases of the same subspace and scores their token coherence. |
| `docs/PRIVILEGE_PREREG.md` | Records hypotheses, kill criteria, and the later causal addendum. |
| `docs/GAUGE_RESULT.md` | Detailed result report for the gauge and privileged-basis experiments. |
| `gauge/monitor_flip.py` | Trains the code-versus-English monitor and tests it after one valid coordinate change. |
| `gauge/monitor_seed_sweep.py` | New robustness check repeating the monitor experiment over 32 or more coordinate choices. |
| `gauge/sae_audit.py` | Tests two trained sparse autoencoders under model symmetries and investigates an SAE's own scale freedom. |
| `gauge/router_centre.py` | Removes a functionally irrelevant common router component and re-runs the basis comparison. |
| `jlens/causal_basis.py` | Repeats the privileged-basis comparison using interventions rather than token readouts. |
| `verify.py` | A reviewer-oriented collection of eleven compact checks. It is useful, but its current claim to recompute every result is too broad; this is discussed later. |
| `jlens/app_figs.py` | Generates the figures used by the draft application. |
| `docs/APPLICATION_GOOGLEDOC.md` | Current long-form application draft. Treat it as a factual outline, not text to paste unchanged. |
| `docs/MATS12_FORM_ANSWERS.md` | Draft responses corresponding to form fields. Several require personal information or personal judgment. |

### 2.2 The original J-Lens spine

| File or directory | Role |
|---|---|
| `jlens/lens.py` | Computes partial or full Jacobians between residual-stream locations. |
| `tests/test_lens_math.py` | Verifies the vector-Jacobian-product shortcut against an explicit sum over destination positions. |
| `tests/test_identity_at_target.py` | Checks that a Jacobian from a layer to itself is the identity. |
| `tests/test_full_vs_partial.py` | Checks that a materialized Jacobian and the cheaper partial computation agree. |
| `tests/test_multilayer.py` | Checks the all-layers optimization against separate single-layer calculations. |
| `jlens/verify.py` | Matches a published J-Lens artifact and resolves layer-index conventions. This is different from the top-level `verify.py`. |
| `jlens/eigen.py`, `jlens/eigen_vs_svd.py`, `jlens/assay_uv.py` | Study eigenvectors versus singular vectors for reading and steering. |
| `jlens/birth.py`, `jlens/eigen_training.py`, `jlens/ode_view.py` | Study how Jacobian structure changes over training and relate depth to a state-transition system. |

### 2.3 Supporting or failed research branches

| Area | Representative files | Status in an application |
|---|---|---|
| Steering and localization | `pullback_steer.py`, `position_sweep.py`, `tradeoff.py` | Useful supporting lesson: position and dose are hidden variables. Do not make this the main claim. |
| Chess | `chess_j.py`, `chess_concepts.py`, `piece_ablate.py` | Useful negative result with an oracle: dominant directions track tokenization more than chess concepts. |
| Mixture of experts | `moe_*`, `router_*` | Useful application of the orbit-null idea and a strong example of correcting an earlier result. |
| Emergent misalignment | `em/`, `jlens/em_*` | Failed to replicate robustly. Mention only as evidence of killing a hypothesis. |
| Reward hacking | `rh/` | Interesting but weakly connected and methodologically caveated. Keep out of the executive summary. |
| Reporting and visualization | `book/`, `jlens/*report*`, `out/*.html` | Useful for learning and browsing; not independent evidence. |

The practical lesson is that file count is not result count. Many scripts are
successive attempts, audits, visualizations, or corrections of the same claim.

---

## 3. Foundations: the minimum transformer vocabulary

### 3.1 Tokens

A language model does not directly process words. A tokenizer converts text into
integer token IDs. A token may be a word, a word fragment, punctuation, or a
special marker. The model predicts a probability distribution over the next
token.

This matters because a supposedly semantic result can actually be about
tokenization. The chess branch is an example: the leading Jacobian directions
strongly tracked which half of a tokenized move was being produced, not the
underlying chess concept.

### 3.2 Vectors and dimensions

At each text position, the model stores a vector with `d_model` numbers. For
SmolLM2-135M, `d_model = 576`; for Llama-3.2-1B it is larger. A direction is
another vector of the same length. Projecting a state `h` onto a unit direction
`d` means taking a dot product:

```text
activation along d = h · d
```

Coordinates are the individual numbers of the vector. A basis is the collection
of axes used to describe the vector. The familiar x/y axes are one basis for a
two-dimensional plane, but rotating those axes does not rotate the physical
point. It only changes its coordinates.

### 3.3 The residual stream

A transformer repeatedly updates a shared vector space called the residual
stream. A schematic layer is

```text
h_(l+1) = h_l + Attention_l(h_l) + MLP_l(h_l + Attention_l(h_l))
```

The actual order depends on the architecture, normalization, and implementation,
but the important idea is that information is written into and read from a
common `d_model`-dimensional space.

Residual-stream directions are not the same as coordinates inside an individual
attention head. Head-internal coordinates can be changed while compensating in
the head's output projection, leaving the residual-stream result unchanged.

### 3.4 Logits and the unembedding

At the end of the network, the residual vector is normalized and multiplied by
an output matrix `W_U`, often called the unembedding:

```text
logits = W_U · norm(h_final)
probabilities = softmax(logits)
```

There is one logit per vocabulary token. Larger logits correspond to greater
next-token preference before the softmax conversion to probabilities.

If we have a residual direction `d`, a common readout computes approximately

```text
token scores = W_U · (gamma ⊙ d)
```

where `gamma` is the final RMSNorm scale and `⊙` means coordinatewise
multiplication. Taking the tokens with the largest scores gives a list of words
the direction would directly promote near the model output. This is a **logit
lens** readout.

A coherent list is suggestive, not conclusive. It can be frequency-biased, it
may reflect an arbitrary sign, and it may not predict what the direction does at
an early layer after later nonlinear computation.

### 3.5 Attention heads

Ignoring batching and multiple positions, the value/output part of one head can
be written as

```text
z = sum_t attention_weight_t · W_V h_t
head_output = W_O z
```

`z` lives in the head's internal `d_head`-dimensional space. `W_V` maps the
residual stream into that space. `W_O` maps it back to the residual stream.

The attention weights are computed using queries and keys. The value vector is
mixed across positions using those weights and then sent through `W_O`.

### 3.6 MLPs

Many models in this repository use a gated MLP such as SwiGLU:

```text
g = SiLU(W_gate h)
u = W_up h
mlp_output = W_down (g ⊙ u)
```

The intermediate coordinates are commonly called neurons. The elementwise
nonlinearity and elementwise product distinguish individual neuron coordinates.
A general rotation mixes coordinates before these operations and does not
commute with them. This is why the neuron basis can be architecturally
privileged while an attention head's value basis is not.

"Privileged" means the architecture distinguishes the basis. It does **not**
mean every neuron has a simple human concept.

### 3.7 Mixture-of-experts routing

An MoE layer has many expert MLPs and a router assigning scores to experts. If
the router scores are `s_1, ..., s_E`, the model typically chooses the top few
and normalizes their weights. Adding the same constant `c` to all scores changes
neither their ordering nor their softmax probabilities:

```text
softmax(s + c·1) = softmax(s)
```

Therefore a component shared identically by every router row is functionally
irrelevant to routing. Removing it is a valid gauge choice, not a model edit in
the behavioral sense.

---

## 4. J-Lens from first principles

### 4.1 What a Jacobian is

A Jacobian describes how a small change in one vector changes another vector.
If the later residual state `h_T` depends on an earlier residual state `h_l`,
then

```text
J_l = ∂h_T / ∂h_l
```

is a `d_model × d_model` matrix. For a small perturbation `δh_l`, local linearity
gives

```text
δh_T ≈ J_l δh_l.
```

This is only a local approximation around the prompts used to compute it. Large
interventions can leave the linear regime.

The project averages Jacobians across prompts and valid token positions. That
makes the result cheaper to reuse, but it can mix unrelated computations. The
chess work suggests this averaging can elevate common tokenization structure
over task-specific mechanisms.

### 4.2 Why a full Jacobian is expensive

A full `d × d` Jacobian can require work proportional to `d` backward passes.
For a model with thousands of residual dimensions, this is expensive.

Suppose we only care how the earlier state affects token `k`. Let `w_k` be that
token's unembedding vector. Instead of materializing all of `J`, automatic
differentiation can compute

```text
J_l^T w_k
```

with a vector-Jacobian product. For `K` selected tokens, this costs roughly `K`
backward passes rather than `d_model` passes.

The core implementation is in `jlens/lens.py`. It takes care with three details:

1. a layer index refers to the residual **leaving** a block;
2. causal masking makes one seeded backward pass sum over all permitted later
   positions; and
3. early attention-sink positions and the last position are excluded from the
   averaging convention.

The first implementation matched the published lens at cosine roughly `0.94`,
which looked reassuring but was still wrong. Fixing the layer convention and
position reduction raised the match to `0.9984`. This is an important general
lesson: a high similarity is not automatically validation when the expected
answer is much closer to one.

### 4.3 The tests that establish chain of custody

The repository does not use a conventional `pytest` suite; the tests are
executable scripts with assertions.

- `tests/test_lens_math.py` compares the optimized VJP against a slow explicit
  sum. In the current environment it passes with maximum absolute difference
  `5.364e-7` at signal scale `2.052`, and zero causal leakage.
- `tests/test_identity_at_target.py` checks that a Jacobian from a residual point
  to itself is the identity.
- `tests/test_full_vs_partial.py` checks the partial token-seeded result against a
  materialized full Jacobian.
- `tests/test_multilayer.py` checks the optimization that obtains multiple source
  layers from one graph.
- `jlens/verify.py` compares against a published external lens.

The intended chain is:

```text
published artifact
      ↓
partial VJP path
      ↓
full Jacobian path
      ↓
all-layers path
```

Each optimization is compared to a simpler path that was already checked.

### 4.4 SVD versus eigenvectors

For a square Jacobian, singular vectors and eigenvectors answer different
questions.

The singular value decomposition is

```text
J = U Σ V^T
J v_i = σ_i u_i.
```

The right singular vector `v_i` is an input direction. The left singular vector
`u_i` is the resulting output direction. If you want to inject the direction
that produces maximum linear output change at fixed input norm, you inject
`v_i`, not `u_i`. An early project version injected `u_i`; `assay_uv.py`
documents the correction.

An eigenvector satisfies

```text
J q_i = λ_i q_i.
```

Because the source and destination are both residual-stream spaces, an
eigenvector is a direction that maps back onto itself, scaled or sign-flipped.
That makes eigenvectors conceptually appealing for reading persistent axes.

The original headline was "read with eigenvectors; steer with singular
vectors." It is interesting, but it is no longer the best application headline.
The decomposition itself can depend on coordinate conventions, individual
vectors can be unstable under near-degeneracy, and the later gauge analysis
provides a more fundamental question with an exact null.

### 4.5 Continuous-depth interpretation

If a residual network is approximated as a continuous dynamical system

```text
dh/dt = F(t, h),
```

then the derivative of a later state with respect to an earlier one is the
state-transition matrix `Φ(T,t)`. In this analogy, J-Lens estimates

```text
J_l ≈ Φ(T,l).
```

This predicts a semigroup relation:

```text
Φ(T,a) ≈ Φ(T,b) Φ(b,a).
```

The repository reports that the composition approximation has relative error
`0.256`, compared with `0.744` for an identity control. This is suggestive rather
than exact: transformer blocks are discrete and prompt averaging does not
generally commute with matrix multiplication.

---

## 5. Gauge freedom: the central idea

### 5.1 What “gauge” means here

In this project, a gauge freedom is a change in parameters or internal
coordinates that leaves the model's computed function unchanged. Multiple
parameter settings describe the same input-output behavior.

The word comes from physics, but no physics background is required. A mundane
analogy is describing the same location using meters or feet. The numbers
change; the place does not. A statement that depends on the unit choice is not
an intrinsic property of the place.

For interpretability, the test is:

```text
1. Transform the model using a proven function-preserving symmetry.
2. Verify numerically that outputs are unchanged to floating-point tolerance.
3. Re-run the interpretation method.
4. If the interpretation changes, it depends on the representation, not only
   on the model's function.
```

This gives a known-zero control. The model changed by zero in the relevant
functional sense, so a function-level interpretation should also change by zero.

### 5.2 Attention value-output freedom

Write one head's value/output computation as

```text
y = W_O W_V x
```

where attention-weighted mixing across positions is omitted because it is linear
in the values and does not act on the value coordinates.

Choose any invertible `d_head × d_head` matrix `M`. Define

```text
W_V' = M W_V
W_O' = W_O M^{-1}.
```

Then

```text
W_O' W_V'
= W_O M^{-1} M W_V
= W_O W_V.
```

Nothing downstream can tell which internal coordinate system was used.

For an orthogonal rotation `R`, the inverse is the transpose, so the code often
uses

```text
W_V' = R W_V
W_O' = W_O R^T.
```

The orthogonal group has `d_head(d_head-1)/2` continuous degrees of freedom. But
the actual value-output freedom is larger: all invertible matrices are allowed,
so the full general linear group has `d_head^2` dimensions. At `d_head = 64`,
that is `4096` continuous parameters per independently transformable group.

Grouped-query attention requires care. Several query heads may share one value
head. The value rotation must then be compensated in every corresponding output
head. `gauge/matrix.py` and `jlens/rotate_sweep.py` do this by rotating an entire
KV group, not an arbitrary isolated query head.

### 5.3 Why the head-column interpretation is not identified

Each column of `W_O` can be read through the unembedding and assigned tokens.
But the transformation `W_O → W_O R^T` replaces each column with a mixture of
the old columns. Since every `R` describes the same model, no behavioral
observation can select the original columns as uniquely real.

This does not imply that the **subspace** spanned by those columns is arbitrary.
Right multiplication by an invertible matrix changes the basis but preserves the
column space. The head's write subspace can therefore be invariant even when its
individual axes are not.

### 5.4 Query-key freedom and rotary embeddings

Without positional complications, matching transformations of queries and keys
can preserve dot products. Rotary positional embeddings apply structured
position-dependent rotations to coordinate pairs. A general change of basis no
longer commutes with those rotations.

The project tests two cases:

- a general query/key rotation changes logits substantially, around `0.56` in a
  demonstrated Llama case;
- rotations inside each rotary pair remain free to floating-point precision,
  around `4.6e-5`.

For a 64-dimensional head, this reduces the exhibited continuous freedom from
`2016` orthogonal rotation dimensions to `32` pairwise rotations. The language
"RoPE gauge-fixes 98.4%" is a helpful shorthand, but it should be presented as a
statement about the tested rotational family, not a complete theorem classifying
every possible symmetry.

### 5.5 Why an MLP rotation is different

For a gated MLP,

```text
y = W_down [SiLU(W_gate x) ⊙ (W_up x)].
```

If all hidden units are rotated by a dense matrix `R`, generally

```text
SiLU(Ra) != R SiLU(a)
```

and

```text
(Ra) ⊙ (Rb) != R(a ⊙ b).
```

The rotation therefore changes the function. In `jlens/rotate_sweep.py`, the
same style of algebraic rotation that is harmless inside attention produces
logit changes of roughly `0.67` to `2.89` when applied to MLP units, versus
roughly `4e-5` for attention.

Some narrower MLP transformations remain free:

- permuting neuron identities while applying the same permutation to every
  connected matrix is a relabeling;
- scaling the linear `W_up` branch by `c` and the corresponding `W_down` column
  by `1/c` cancels, because the scale does not pass through the SiLU branch;
- arbitrary dense rotations are not free.

This is why the neuron basis is more identified than the attention value basis,
without implying that neuron activations have an absolute, convention-free
scale in every architecture.

### 5.6 Router shift freedom

If router logits are `R x`, replace every router row by

```text
R' = R + 1 v^T.
```

Every expert score then receives the same additional scalar `v^T x`. Top-k
selection is unchanged, and softmax is unchanged. Subtracting the mean router
row is one convenient gauge choice:

```text
R_centered = R - mean_over_experts(R).
```

The project measures approximately `38.3%` of the router matrix norm in this
shared, functionally irrelevant direction for the tested OLMoE model. Because a
large common component also makes the rows form a narrow cone, failing to remove
it can create an apparently meaningful geometric effect.

---

## 6. Experiment A: the same rotation in attention and an MLP

### 6.1 Question

Does the mathematical symmetry appear numerically in real transformer
implementations, and how much does a head-column readout change under it?

### 6.2 Design

For every layer of three model families:

1. record logits on six prompts;
2. read the top ten tokens of the chosen head's output columns;
3. rotate a valid attention KV group and compensate in its output blocks;
4. recompute logits and readouts;
5. restore the original weights;
6. separately apply a matched-size dense rotation to selected MLP hidden units;
7. recompute logits and restore again.

The attention transformation is predicted to preserve the function. The MLP
transformation is predicted to break it. This comparison guards against an
uninformative explanation such as "the rotation was too small to matter."

### 6.3 Results

The detailed report gives the following layer-aggregated figures:

| Model | Attention max logit change | MLP max logit change | Ratio | Head readout retained |
|---|---:|---:|---:|---:|
| SmolLM2-135M | `4.20e-5` | `1.95` | `46,392×` | `1.5%` |
| Qwen2.5-0.5B | `5.63e-5` | `2.89` | `51,350×` | `0.4%` |
| Llama-3.2-1B | `4.15e-5` | `0.67` | `16,066×` | `0.3%` |

The exact values can vary slightly with software and hardware. The qualitative
separation is enormous: the attention change is at the numerical floor while
the MLP rotation visibly changes the model.

![The same rotation is harmless in attention but destructive in an MLP; the head-column readout is replaced](../out/figs_app/fig1_symmetry.png)

### 6.4 What the result establishes

- The code implements an actual numerical symmetry rather than only an algebraic
  claim on paper.
- Individual head-column token readouts are not invariant under that symmetry.
- The failure replicates across the tested architectures and layers.

### 6.5 What it does not establish

- It does not show that the head's subspace is meaningless.
- It does not show that every published attention interpretation reads individual
  value/output coordinates in this way.
- It does not show that every invariant object is interpretable.
- It does not establish a behavioral danger by itself; the monitor experiment
  supplies an operational example.

---

## 7. The invariance matrix: auditing several interpretation methods

`gauge/matrix.py` applies several symmetries and records how much each method's
output is retained. Different output types need different comparison metrics:

- top-k token lists use fractional overlap;
- rankings use Spearman correlation;
- subspaces use projection error;
- scalar residual probes use relative difference.

The main methods are:

1. MLP write-direction logit lens;
2. maximum-activating examples;
3. unit ranking by activation magnitude;
4. activation times gradient;
5. a residual-stream probe;
6. OV singular-vector readout;
7. OV subspace;
8. attention-head output columns.

The qualitative result is more important than memorizing every table cell:

- head-column token readouts collapse under a valid head rotation;
- the OV subspace survives;
- raw OV singular vectors can appear to change because singular-vector signs are
  arbitrary;
- fixing signs by a deterministic convention removes that superficial failure;
- residual-stream values survive transformations that only reparameterize
  internal components upstream;
- MLP feature identities change under a neuron permutation, as expected, unless
  results are compared modulo that permutation;
- activation-times-gradient cancels some scale freedoms that raw activation
  magnitude does not.

A method's output format matters. If a transformation merely relabels neurons,
then comparing "neuron 17 before" with "neuron 17 after" is the wrong matching.
The correct invariant object might be the set of features modulo permutation.
The project sometimes uses deliberate naive comparisons to demonstrate how a
typical claim fails, but your explanation should always specify what equivalence
relation is being used.

---

## 8. The OV circuit and the sign problem

### 8.1 Why `W_O W_V` is invariant

Under the value/output transformation,

```text
W_O' W_V' = W_O M^{-1} M W_V = W_O W_V.
```

Therefore the combined linear map is independent of the internal value basis.
For grouped-query attention, the relevant combined map and grouping require
care, but the same cancellation applies when transformations are tied correctly.

### 8.2 Why singular vectors can still appear unstable

An SVD is not unique. If `u` and `v` form a singular pair, then `-u` and `-v`
form the same pair. A numerical solver is free to return either sign.

This is harmless for reconstructing the matrix:

```text
σ u v^T = σ (-u) (-v)^T.
```

It is not harmless for a one-sided top-token readout. The top tokens of `u` can
become the bottom tokens of `-u`. The project fixes the sign by forcing the
largest-magnitude coordinate of each vector to be positive. This is a convention,
but a deterministic convention is enough to compare equivalent matrices.

Near-equal singular values create a deeper issue. If two singular values are
equal or nearly equal, their vectors can rotate within the degenerate subspace
without meaningfully changing the matrix. In that case, individual singular
vectors are weakly identified even after fixing signs. The span of the block is
the more stable object.

### 8.3 Application-safe conclusion

Say:

> Head output columns are basis-dependent. The combined OV operator is invariant,
> and its subspace is a better-defined object. Individual decomposed directions
> additionally require sign and degeneracy handling.

Do not compress this to "SVD fixes attention interpretation." That is broader
than the evidence.

---

## 9. The monitor experiment: turning coordinate dependence into a consequence

### 9.1 Why this experiment matters

Changing a list of top tokens may sound cosmetic. A skeptic could agree that the
labels are coordinate-dependent while asking whether anything operationally
important fails. `gauge/monitor_flip.py` answers with a deliberately simple
monitor.

The task is binary classification: is a piece of text code or ordinary English?
The dataset contains 16 short code examples and 16 short prose examples. The
model processes each text. At a selected layer, the script captures:

- the concatenated internal value/output input across all attention heads;
- the residual-stream vector leaving the layer.

Both representations have the same total dimensionality. The script trains a
ridge-regularized linear classifier on the first half of the code texts and the
first half of the prose texts. It splits by entire text rather than by token, so
tokens from one text cannot leak into both training and test sets.

It then rotates every attention-head value basis in the layer, with the required
compensation in `W_O`, and applies the **same already-trained classifier** to the
transformed features. No retraining is allowed: the question is whether a monitor
deployed on a functionally equivalent model remains valid.

### 9.2 Original cross-model result

| Model | Head monitor before → after | Residual monitor before → after | Max model change |
|---|---:|---:|---:|
| SmolLM2-135M | `0.84 → 0.61` | `0.95 → 0.95` | about `3e-5` |
| Qwen2.5-0.5B | `0.83 → 0.42` | `0.97 → 0.97` | about `3e-5` |
| Llama-3.2-1B | `0.91 → 0.55` | `0.95 → 0.95` | `4.48e-5` |

![A head-internal classifier degrades while a residual-stream control remains stable](../out/figs_app/fig3_monitor.png)

Calling `0.55` "chance" is approximately correct for a balanced binary task,
but the evaluation is token-weighted, so the exact majority baseline should be
reported when precision matters. `0.42` is below chance on this finite sample;
it should be described as a collapse rather than as evidence of systematically
inverted knowledge.

### 9.3 New multi-seed robustness experiment

The original result uses one set of random rotations per model. That leaves an
obvious question: was the chosen rotation unusually destructive?

`gauge/monitor_seed_sweep.py` keeps the following fixed:

- SmolLM2-135M;
- layer 15;
- the 32 source texts;
- the train/test split;
- the fitted head and residual classifiers; and
- all model parameters outside the temporary gauge transformation.

It samples 32 coordinate systems. For each seed it rotates every valid KV group,
checks the model logits, records monitor performance, and restores the original
weights in a `finally` block.

The result is:

| Measurement | Before | After 32 rotations |
|---|---:|---:|
| Head monitor, token accuracy | `0.840` | mean `0.495`, median `0.460`, range `0.368–0.620` |
| Residual monitor, token accuracy | `0.951` | `0.951` for every seed |
| Head monitor, equal-weight text accuracy | `0.875` | mean `0.490` |
| Residual monitor, equal-weight text accuracy | `1.000` | `1.000` for every seed |
| Worst max absolute logit change | — | `4.20e-5` |

![The monitor failure repeats across 32 random coordinate systems](../out/figs_app/fig10_monitor_seeds.png)

This is stronger evidence that the single-seed failure is typical. The plot also
shows why reporting a distribution is better than reporting only the mean: some
rotations leave accuracy near `0.62`, while others push it below `0.40`.

### 9.4 Limits of the monitor result

This is a demonstration, not a realistic safety-monitor benchmark.

- There are only 32 hand-written texts and 16 held-out texts.
- The classifier is linear and the task is easy.
- Tokens from the same text are correlated, so 163 test tokens are not 163
  independent examples. The equal-weight text metric helps, but 16 texts remain
  a small sample.
- The result shows that a coordinate-specific monitor can break after a valid
  reparameterization. It does not show that deployed monitors routinely undergo
  such rotations during training or fine-tuning.
- The rotation seeds are robustness conditions, not independent draws from a
  population of models. Quantiles across seeds should not be presented as a
  confidence interval for generalization to all models.
- A monitor could be retrained after a reparameterization. The concern is that
  its feature identity is not intrinsic and cannot be expected to transfer
  without alignment.

The defensible statement is:

> A fixed linear monitor built on head-internal coordinates is not invariant to
> function-preserving changes of those coordinates; in this controlled example,
> performance falls from strong to approximately chance across 32 random valid
> rotations, while a residual-stream control remains unchanged.

---

## 10. Sparse autoencoders: what the audit does and does not show

### 10.1 What an SAE is

A sparse autoencoder maps an activation vector `x` to a much larger feature
vector `f`, keeps only a small number of active features, and reconstructs the
activation:

```text
preactivations = W_enc (x - b_dec) + b_enc
features = sparsify(preactivations)
reconstruction = W_dec features + b_dec.
```

Researchers inspect the decoder directions or the examples that activate each
feature. An SAE is intended to replace a dense, potentially superposed
representation with a sparse feature dictionary.

### 10.2 External symmetry audit

`gauge/sae_audit.py` uses two independently released SAEs on Llama-3.2-1B:

- an EleutherAI SAE with roughly 131k features on an MLP output;
- a huypn16 SAE with roughly 65k features on the residual stream.

It runs model-level symmetries such as head rotation, MLP rescaling, and MLP
permutation, then compares the SAE features. The saved result reports 100% of
firing features retained, with maximum activation changes around `1e-6` to
`2e-5`.

This is expected for an SAE reading a representation that the model-level
transformation leaves numerically unchanged. It is still useful as a positive
control: the audit does not mechanically declare every method unstable.

### 10.3 The SAE's own scale freedom

A plain ReLU SAE has its own rescaling symmetry. For positive `c_i`, multiply
feature `i`'s detector and bias by `c_i` and divide its decoder vector by `c_i`.
Because ReLU is positively homogeneous,

```text
ReLU(c_i z_i) = c_i ReLU(z_i),
```

the reconstruction can remain unchanged even though feature activation
magnitudes change. In the repository's test, reconstruction changes by only
`6.68e-6`, while the identity of the strongest features retains only `46.9%`.

Both released SAEs use unit-norm decoder directions, which fixes this scale by
convention. They also use top-k selection. Top-k compares feature magnitudes
against one another, so arbitrary independent rescaling changes which features
are selected and changes the reconstruction; the test reports reconstruction
difference about `0.208` after the rescaling.

### 10.4 Correct interpretation

The evidence supports:

- these two SAE implementations are stable under the tested upstream model
  symmetries;
- their decoder normalization and top-k architecture constrain their internal
  scale freedom; and
- a plain ReLU SAE without a normalization convention would not support absolute
  comparisons of feature magnitude.

It does not support "SAEs are safe" or "SAE features are correct." Only two SAEs
from one model family were tested, and symmetry invariance cannot certify semantic
validity.

---

## 11. The orbit null and privileged-basis experiment

### 11.1 The question

Experiment A shows that directions inside one attention head are unidentifiable.
It does not answer whether bases that the architecture *does* distinguish are
more interpretable than arbitrary alternatives.

For example, an MLP's elementwise nonlinearity distinguishes individual neuron
coordinates. Are those coordinates actually more readable than random mixtures
of the same neuron subspace?

### 11.2 Holding the subspace fixed

Let `A` be a `d × n` matrix whose columns are `n` candidate directions. The
columns span a subspace `S`. We want to compare bases without changing `S`.

Write the QR decomposition

```text
A = Q R,
```

where the columns of `Q` are orthonormal and span `S`. Let `H` be a random Haar
orthogonal matrix—informally, a uniformly random rotation in `n` dimensions.

The code constructs five versions:

#### Raw basis

```text
A
```

These are the original model directions.

#### Symmetry-orbit or reparameterization null

```text
A H
```

This randomly mixes the original columns while preserving their span. For the
columns inside an attention head, it corresponds to weights reachable by an
exact function-preserving rotation. For MLP neurons it is a counterfactual basis,
not a function-preserving reparameterization.

#### Gram-matched null

```text
Q H R
```

Its Gram matrix is

```text
(QHR)^T(QHR)
= R^T H^T Q^T Q H R
= R^T H^T H R
= R^T R
= A^T A.
```

Therefore it has exactly the same pairwise dot products, norms, and conditioning
as the raw basis, but randomizes which directions carry that geometry. This is a
strong control against the explanation "the raw directions only win because
they are non-orthogonal or have unusual norms."

#### Orthonormal null

```text
Q H
```

This is a random orthonormal basis of the same subspace. Comparing only against
this null can be misleading because it changes the geometry as well as the
identity of the directions.

#### SVD basis

```text
U from A = U Σ V^T
```

The left singular vectors give a deterministic orthonormal basis of the same
column space, subject to sign and degeneracy ambiguities.

### 11.3 Which direction families are tested

`jlens/privilege.py` samples groups from several arms:

- columns from inside one attention head—the known-null calibration;
- OV-circuit singular vectors;
- attention columns mixed across different heads;
- MLP down-projection columns;
- standard residual-stream coordinate axes; and
- MoE router rows.

Each group generally contains 64 directions. The executed report aggregates 604
groups across four model families, with multiple random basis draws per group.

### 11.4 Turning a direction into a score

For each direction, the model-under-test supplies its top-ten logit-lens tokens.
The current headline metric judges whether those tokens are semantically coherent
using an unrelated model's embedding space. Tokens are matched by decoded text
across tokenizers.

For a token list, the metric computes mean pairwise embedding cosine and subtracts
the expected cosine for tokens in matched frequency bins. This reduces a common
confound: frequent or rare tokens may cluster differently even without semantic
coherence.

The main group statistic is the 90th percentile across directions, not the mean.
The motivation is that only a small tail of units may be readable. Averaging 64
directions could make a basis containing three coherent directions look similar
to one containing none.

This choice is defensible, but it is also a protocol change. The initial
pre-registration describes an NPMI topic-coherence measure, and the repository
states that the first pass used a mean statistic before switching to the
90th-percentile cross-model embedding score. The application must not describe
the entire final analysis as fully pre-registered. A precise description is:

> The hypotheses and kill criteria were written in advance. Diagnostics led to a
> revised coherence metric and tail statistic; the known-null attention arm and
> frequency checks were retained as calibration, and the change is disclosed.

### 11.5 Results

The primary reported comparison is raw basis minus its random orbit on the
90th-percentile coherence statistic:

| Arm | Raw − orbit, 95% interval | Interpretation |
|---|---:|---|
| Attention-head columns | `+0.0011 [−0.0023, +0.0047]` | Required null; the instrument returns approximately zero. |
| MLP neurons | `+0.0209 [+0.0121, +0.0297]` | The model's neuron basis has a measurable readable tail. |
| OV-circuit directions | `+0.0065 [+0.0033, +0.0097]` | Small positive result for the invariant construction. |
| Columns mixed across heads | `+0.0339 [+0.0244, +0.0439]` | Head boundaries appear meaningful. |
| Residual-stream axes | `−0.0023 [−0.0055, +0.0007]` | Pre-registered positive prediction is not supported at the tail. |
| MoE router rows | `−0.0087 [−0.0404, +0.0180]` | No clear identity advantage in the original pooled table. |

![The known-null arm returns zero while MLP neurons and cross-head structure show advantages](../out/figs_app/fig6_privilege.png)

The attention row is load-bearing. Without it, a positive MLP difference might
reflect a generic advantage of single vectors over mixtures, a metric artifact,
or the chosen tail statistic. The attention arm undergoes the same measurement
but has an answer known from the model symmetry. It returns approximately zero.

### 11.6 Identity versus cone geometry

Two effects can make raw directions outperform an orthonormal basis:

1. **identity:** these particular directions are better than random mixtures of
   themselves, measured by `raw − orbit`;
2. **geometry:** any basis preserving the original cone or non-orthogonality is
   better than an orthonormal basis, measured by `orbit − orthonormal`.

The MoE router originally looked readable relative to an orthonormal basis. Much
of that gap was recovered by a random non-orthogonal basis, suggesting that its
narrow cone rather than the identity of its rows drove the result. The later
router-centering experiment refined this conclusion again.

### 11.7 Statistical cautions

- Confidence intervals are bootstrapped over groups, but groups within the same
  model and nearby layers may be correlated. Four model families are not 604
  independent model replications.
- The 90th percentile was adopted after the mean appeared insensitive. This is a
  reasonable measurement correction, but it increases researcher degrees of
  freedom and should be disclosed.
- Multiple arms, metrics, model splits, and robustness variants exist. The known
  null and stated kill criteria help, but they do not eliminate multiple-testing
  concerns.
- The coherence proxy measures readable token lists, not all possible kinds of
  semantic or causal structure.
- Olmo-3-7B's MLP result has the expected sign but a confidence interval including
  zero. Do not imply uniform replication across all models.

---

## 12. The causal basis experiment

### 12.1 Why reading is not acting

A logit-lens readout asks what tokens a direction would directly promote if it
were near the output. An early-layer direction then passes through many nonlinear
blocks. Its readout may have little relationship with its actual intervention
effect.

`jlens/causal_basis.py` replaces the readout token list with tokens actually
promoted after injecting the direction into the residual stream and completing
the forward pass.

### 12.2 Dose validation

Large interventions can make many different directions produce a shared generic
effect. The project measures **common-mode**, the mean similarity among the
induced logit changes from different directions. If common-mode is high, a basis
comparison cannot distinguish directions meaningfully.

An initial dose around `alpha = 1.0` produced common-mode as high as `0.97` in an
MLP arm. The pre-registered causal kill criterion fired, and the run was discarded.
The experiment was repeated at much lower doses, where common-mode is around
`0.005` and the effect is direction-specific.

This is an excellent example to understand because it shows a control changing
the experiment rather than being mentioned decoratively after the result.

### 12.3 Result and scope

The detailed report gives:

- attention-head known-null mean within-group z: `+0.30` with interval spanning
  zero;
- MLP neurons: `+1.55 [+0.81, +2.41]`;
- MLP raw-minus-orbit effect: `+0.0231 [+0.0111, +0.0353]`, about `4.4×` the
  null-arm effect size.

The ordering agrees with the readout experiment: attention columns behave like a
null and MLP neurons show an advantage over random rotations. This is stronger
than a readout-only result.

However:

- interventions are made at the final token in a small prompt set;
- the repository's separate depth experiment shows readout/action agreement is
  poor before roughly the final quarter of the network;
- one model split reportedly fails a kill criterion or is confounded; and
- a low common-mode verifies directional diversity, not semantic specificity.

The correct conclusion is that the basis advantage has a causal analogue under
the tested low-dose protocol, not that individual MLP neurons implement clean,
portable concepts.

---

## 13. Applying the method to the MoE router

The router branch is valuable because it shows the method correcting the
project's own earlier conclusion.

The initial comparison found router rows more readable than an orthonormal basis
and attributed much of the advantage to their shared cone geometry. The gauge
analysis then noticed that adding a common component to every router row changes
nothing about routing. In the tested OLMoE model, approximately `38.3%` of the
router matrix norm lies in that free shared direction.

`gauge/router_centre.py` subtracts the mean row—a function-preserving operation—
and repeats the basis comparison. The application draft reports:

| Contrast | Original router | Centered router |
|---|---:|---:|
| Identity, raw − orbit | `+0.0016` with interval spanning zero | `+0.0294 [+0.0203, +0.0385]` |
| Cone, orbit − orthonormal | `+0.0236 [+0.0084, +0.0425]` | `+0.0079 [+0.0036, +0.0129]` |

The earlier "it is mostly the cone" conclusion was partly a gauge artifact. Once
the irrelevant common direction is removed, a clearer identity effect emerges.

This is a persuasive application example because the method does something
productive: it does not merely invalidate a result; it separates an artifact
from a remaining signal.

It also needs careful language. The centering is free for the router's softmax
and top-k decisions, but downstream implementations should be checked for any
additional use of raw router logits. The current code verifies top-k identity and
probability differences near numerical zero for the architecture tested.

---

## 14. Supporting lessons from the rest of the repository

### 14.1 Steering: the right direction, position, and dose

For `J v = σu`, injecting `v` is the type-correct singular-vector intervention;
injecting `u` confuses output and input spaces. Several early steering results
were invalidated by this mistake and re-run.

Even with the correct direction, position dominates. The top-level verification
shows that a gradient pullback for changing Paris to Rome behaves differently
when injected at the `France` token versus the last token. In SmolLM2-135M, the
effect changes sign. Across a broader analysis, the same direction and dose were
reported as roughly `+12.3` standard deviations at the subject position and
`−1.3` at the final position.

Dose also changes conclusions. At saturation, many directions bulldoze the same
output behavior, and the relationship between efficacy and specificity can even
reverse sign. A credible steering result should therefore report:

- exact layer;
- exact token position;
- direction construction;
- normalization and dose;
- dose sweep;
- random-direction distribution, not one random draw;
- target effect and non-target leakage;
- base log-probability or headroom; and
- qualitative generations at randomly selected examples.

### 14.2 Chess: an oracle-backed negative result

Chess provides ground truth through legal-move computation. The repository finds
that leading Jacobian directions track a tokenization variable—roughly which half
of a move is being emitted—with reported `R² = 0.66`, while capture status has
`R² = 0.16` and check status less.

The proposed mechanism is averaging: prompt- or position-specific Jacobians for
different stages of move generation have low similarity, around `0.21` in one
comparison. Their average emphasizes the mixture structure shared across
positions rather than any one chess computation.

A causal ablation result finds piece-labeled subspaces largely interchangeable:
removing the "pawn" subspace does not selectively hurt pawn decisions much more
than removing another piece's subspace. This is a useful negative result, but
`verify.py` currently reads its values from a saved JSON rather than recomputing
the experiment. Do not describe it as independently regenerated by that script.

### 14.3 J-Lens for MoE routing

When predicting which experts will be selected at a later layer, the repository
reports:

- carry forward the existing residual/router information: about `7.2/8` experts;
- transport with the averaged Jacobian: about `1.3/8`;
- chance: about `1.0/8`.

This is evidence that an expensive Jacobian transformation adds little for this
particular routing-prediction task. It is not a universal proof that Jacobians
are useless. The no-change baseline is strong because router states may persist
locally across layers.

### 14.4 Training dynamics

The project's special advantage is that J-Lens needs no trained probe, so the
same construction can be applied at many training checkpoints. Reported patterns
include:

- readable directions appearing gradually rather than existing at initialization;
- a dip across one training-stage transition;
- a collapse in large-magnitude eigenvalue channels during training; and
- early layers having greater leverage per unit weight change because their
  effects have more depth through which to compound.

These results are interesting background. They rely on particular models,
checkpoints, and averaging conventions, and they are less directly connected to
the clean gauge-freedom application story.

### 14.5 Reward hacking and emergent misalignment

The reward-hacking branch reports a probe that predicts whether a real solution
will follow an exploit, with AUC around `0.811` at a position before the solution
appears. It also records two major deflations: one concealment probe was nearly
duplicated by bag-of-words text features, and the system prompt explicitly
described the exploits.

The emergent-misalignment branch did not find a robust shared Jacobian-change
direction across architectures. These branches demonstrate breadth and the
ability to reject weak findings, but including them in the executive summary
would make the application look unfocused.

---

## 15. The failure catalogue and what each failure teaches

The repository's mistakes are not an embarrassment to hide. They are valuable
only if you understand exactly why they happened and how the corrected protocol
prevents recurrence.

### Failure 1: confusing a strong marginal baseline with learned structure

A chess legality probe can appear nearly perfect because certain squares are
usually legal origins or destinations. The correct baseline is not random chance;
it is a predictor using the square's marginal frequency. A high AUC above 0.5 can
still add almost nothing beyond that baseline.

**Lesson:** compare against the strongest trivial predictor that uses information
available at the same point.

### Failure 2: semantic categories acting as magnets

A broad category such as "quantifier" may collect many hits because frequent
function words cluster near a generic direction. Replacing one category with
another can move the apparent discovery.

**Lesson:** inspect category base rates, frequency, and nearest generic baselines;
do not treat a labeler's chosen noun as ground truth.

### Failure 3: deriving the chance line incorrectly

The expected overlap for a tall matrix used the wrong denominator in an early
component analysis, inflating the chance baseline error by a factor reported as
`5.4×` and manufacturing a result.

**Lesson:** measure null distributions empirically when practical and verify
analytical baselines with simulation.

### Failure 4: one dose

Large doses can saturate nonlinearities or impose a shared generic output. A
single dose answers neither whether the effect is linear nor whether it is
specific.

**Lesson:** use a dose-response curve and predefine a validity region.

### Failure 5: one random direction

One random direction once beat a targeted direction by a large factor. A single
draw is an anecdote, not a null distribution.

**Lesson:** report the targeted result relative to many matched random draws.

### Failure 6: mismatched audit populations

A cross-model comparison mixed pre-audit and post-audit direction sets and looked
like replication.

**Lesson:** freeze inclusion rules before comparing datasets and retain a table
showing which rule generated each row.

### Failure 7: hidden token position

Routing or steering measured at the final token can classify prompt formatting
rather than the intended concept. Position changes the computation and can flip
an intervention's sign.

**Lesson:** position is part of the intervention definition, not an implementation
detail.

### Failure 8: a single outlier owns the variance

An attention-sink direction carried up to `99.8%` of uncentered variance in one
analysis. A correlation that survived a random-direction null reversed from
`+0.229` to `−0.856` when the outlier direction was removed.

**Lesson:** random nulls do not protect against structured outliers. Plot the
distribution, center appropriately, remove leading components as a sensitivity
analysis, and report the sign stability.

### Failure 9: a result already present in the text

A probe can predict a label because the label's evidence is already present in
the input text. A reported concealment probe at AUC `1.00` was matched by a simple
text baseline around `0.985`.

**Lesson:** evaluate before the relevant evidence appears, and compare against a
strong text-only monitor.

### Failure 10: arbitrary sign and degeneracy

SVD and eigenvector directions can flip signs or rotate inside near-degenerate
subspaces. Token labels then change even when the represented operator does not.

**Lesson:** define sign conventions, inspect spectral gaps, and prefer subspace
comparisons when directions are not uniquely identified.

---

## 16. Evidence ledger: what you can safely claim

Use this section as a fact-checking checklist. "Confidence" is an assessment of
the evidence in this repository, not a formal probability.

| Claim | Evidence | Confidence | Required caveat |
|---|---|---|---|
| A valid attention value/output basis change preserves the model | Algebra plus `verify.py` check 1 and layer sweeps; logit changes around `4e-5` | Very high | Say "mathematically identical, numerically equal up to float roundoff," not literally bit-identical. |
| The full invertible value/output group is free | Algebra plus one condition-number-115 test in `verify.py` check 11 | High | Numerical test covers examples, while the general conclusion comes from algebra. |
| Head-column top-token readouts depend on basis | 70-layer, three-family rotation sweep; roughly 0.3–1.5% retained | High for this readout | Do not generalize to every form of attention-head interpretation. |
| The OV product/subspace is invariant | Algebra and `verify.py` check 3 | Very high | Individual SVD vectors still have sign and degeneracy issues. |
| A head-coordinate monitor can break on an equivalent model | Three one-seed model demos plus the new 32-seed SmolLM sweep | High as a demonstration | Tiny, hand-written binary task; not a realistic deployed-monitor benchmark. |
| These two SAEs retain features under tested model symmetries | `gauge/sae_audit.py` | High for these exact artifacts | Two top-k SAEs, one model family; invariance does not prove semantic correctness. |
| MLP neuron bases have a readable-tail advantage | 604-group basis battery, known-null calibration, cross-model scorer | Moderate to high | Metric/statistic evolved after preregistration; Olmo-7B is underpowered; models are not independent replicates. |
| Head boundaries matter | Cross-head mixing has a positive gap while within-head mixing is null | Moderate to high | This is evidence under the chosen token-coherence score, not a complete account of head modularity. |
| Residual coordinate axes are privileged | Not supported at the 90th-percentile statistic | Low / refuted prediction | Report the null result rather than rescuing the hypothesis with secondary metrics. |
| Router centering exposes an identity effect | Function-preserving centering and repeated basis analysis | Moderate | One MoE architecture; bootstrap units are layers, which are correlated. |
| MLP basis advantage is causal | Low-dose intervention battery | Moderate | Small prompts, final-position intervention, uneven model splits. |
| J-Lens directions are generally meaningful features | Evidence argues against this broad claim | Low | The project explicitly concludes J-Lens is a sensitivity map, not a feature dictionary. |
| Steering pullbacks can move target logits | Direct interventions, e.g. Paris→Rome | High for demonstrated prompts | Effects leak to other targets and depend sharply on dose and position. |
| The chess subspaces are piece-specific | Causal result argues against specificity | Low / negative result | Current top-level verifier loads the saved result rather than recomputing it. |
| Reward-hacking representations reveal hidden intent | Not established | Low | Prompt leakage and simple text baselines explain important earlier results. |

---

## 17. A skeptical code and reproducibility audit

The application becomes stronger if you fix these issues or describe them
honestly.

### 17.1 The root README tells the older story

`README.md` leads with eigen-versus-SVD and depth leverage. The application draft
leads with gauge freedom. A reviewer following the repository link may wonder
which project is being submitted.

**Recommended fix:** rewrite the top of the README around the gauge question and
place the older J-Lens work under "How this question emerged" or create a clean,
immutable application tag with a short reviewer guide.

### 17.2 The central application files are untracked

At the time of this guide, `gauge/`, top-level `verify.py`, the current
application drafts, and many supporting scripts are not tracked by Git. They
will not appear in a remote repository merely because they exist locally.

**Recommended fix:** decide what belongs in the submission, review secrets and
large files, then commit a coherent subset. Do not blindly commit every scratch
artifact.

### 17.3 Installation instructions are broken

The README says:

```bash
uv venv && uv pip install -r requirements.txt
```

but there is no `requirements.txt`. `pyproject.toml` lists only Torch,
Transformers, Accelerate, NumPy, and Matplotlib, while branches also import
packages such as Datasets, PEFT, SciPy, PyArrow, Safetensors, python-chess,
Hugging Face Hub, and Modal.

The current virtual environment also lacks `pytest`, although the test files are
written as standalone scripts rather than pytest functions.

**Recommended fix:** create a minimal core dependency set and optional extras for
chess, fine-tuning, and remote compute. Test the documented installation in a
fresh environment.

### 17.4 `verify.py` overstates what it verifies

Its header and application prose say nothing is loaded from saved results and
every headline is recomputed. At least one check contradicts this:
`check7()` loads `out/chess/piece_ablate.json`. `check6()` recomputes a single
Paris→Rome example, not the broader 46-of-48 result sometimes associated with it.

**Recommended fix:** either make each check recompute its exact advertised claim,
or rename it to `evidence_demo.py` and state which checks recompute versus summarize
saved experiment output. A small machine-readable manifest could record claim,
source file, generation command, runtime, and whether the check is live.

### 17.5 The pre-registration language is too strong

The hypotheses and kill criteria were written early, but the executed metric and
primary statistic changed. The detailed report admits this. Calling the final
battery simply "pre-registered" risks sounding evasive.

**Recommended fix:** distinguish:

- hypotheses and failure criteria specified in advance;
- metric changes made after observing diagnostics;
- confirmatory reruns after the change; and
- exploratory analyses.

If timestamps or commits establish this sequence, link them.

### 17.6 Some prose moves from falsification to certification

Examples of claims to soften include:

- "invariant means the method measures the model";
- "residual-stream features are the safe substrate";
- "most published readings use the wrong object";
- "the only benchmark that cannot be gamed"; and
- "bit-identical" when the implementation produces floating-point differences.

Better language:

- "survives this necessary invariance test";
- "stable under the transformations tested here";
- "this class of head-column readout is basis-dependent";
- "a benchmark with an analytically known null"; and
- "function-equivalent in exact arithmetic and numerically unchanged to about
  `4e-5` logits in float32."

### 17.7 The application scope looks larger than the time limit

The repository spans many days, dozens of commits, hundreds of scripts, multiple
large models, and several research branches. The MATS instructions allow relevant
prior research, but ask for an honest hour estimate and individual contribution,
and warn that prior work may be judged more harshly.

Do not label the entire repository a 16- or 20-hour project unless that is true.
Possible honest framings are:

1. **Prior-research route:** give the real total time and explain the gauge audit
   as the central contribution.
2. **Bounded application project:** identify an exact 20-hour window with a time
   log and submit only experiments genuinely performed in that window. Earlier
   general preparation may be excluded only according to the admissions rules,
   not according to convenience.

The choice is factual, not rhetorical. Use the route matching what happened.

### 17.8 LLM usage must be concrete

The drafts contain extensive agent-written prose and the repository includes
agent prompts. The application instructions explicitly discourage submitting raw
LLM prose.

A strong disclosure answers:

- which tools wrote code;
- which tools proposed hypotheses or controls;
- which prose began as generated text;
- what you personally designed;
- what you personally inspected;
- examples where the agent was wrong;
- how every important number was checked; and
- what percentage or sections you rewrote yourself, if you can estimate honestly.

The defense is not hiding tool use. It is demonstrating ownership of the
reasoning and evidence.

---

## 18. The best application story

This section describes the strongest story supported by the **current** results,
not the strongest possible version of the project. The current evidence supports
a narrow methodological result about head coordinates plus an orbit-null study
of several architecturally supplied bases. It does not yet answer the most
important residual-feature question.

If there is time for another focused research block, prioritize this experiment:

1. sample groups of actual residual-derived directions from trained SAE decoder
   dictionaries, J-Lens, PCA, and supervised probe subspaces;
2. construct raw, orbit, Gram-matched, SVD, and orthonormal bases of each exact
   span;
3. compare held-out activation selectivity, cross-model token coherence, and
   low-dose causal specificity;
4. retain the attention-head arm as the known-zero calibration and MLP neurons
   as an architectural positive comparison; and
5. pre-specify the statistic, dose gate, position, frequency control, and model
   split before observing the new results.

The question then becomes directly relevant to mainstream residual-stream
interpretability: does a method find individually meaningful features, or merely
a useful subspace with an arbitrary basis?

### 18.1 One-sentence research question

For the current narrow submission, write your own version of:

> When interpretability assigns meaning to a direction inside a transformer, is
> that direction a property of the model or an arbitrary coordinate choice, and
> can exact model symmetries provide a ground-truth test?

For the recommended residual extension, the stronger question is:

> Do residual-stream feature methods recover individually special directions, or
> only subspaces whose apparent feature basis is replaceable by a matched random
> rotation?

### 18.2 One-sentence answer

The current evidence supports only the narrower answer. Write your own version
of:

> Attention-head column readouts and monitors can change almost completely under
> function-preserving basis transformations, while invariant OV objects survive;
> comparing candidate directions with random rotations of their own subspace
> provides a practical null test, under which MLP neurons—but not head columns or
> residual coordinate axes—show a measurable basis advantage in the current
> experiments.

Do not imply that this answer covers SAE features, learned residual probes, or
other mainstream residual-derived directions. Their orbit comparison has not yet
been run.

These are study prompts. Do not paste them unchanged if they do not sound like
you.

### 18.3 Recommended executive-summary structure

The admissions document asks for roughly one page, at most three pages and 600
words, with graphs. A good summary can use five short blocks.

#### Block 1: problem and why it matters

Explain that interpretability labels directions without ground truth. Introduce
function-preserving coordinate changes as a case with a known answer.

#### Block 2: decisive experiment

Give the attention equation and one result:

```text
W_V → R W_V, W_O → W_O R^T
max |Δlogit| ≈ 4e-5, identical greedy text,
but about 99% of head-column top tokens are replaced.
```

Use Figure 1.

#### Block 3: operational consequence

Describe the monitor experiment. Prefer the new seed-sweep statistic because it
answers a natural robustness objection:

```text
SmolLM2 head monitor: 0.840 → 0.495 mean across 32 rotations
residual control:      0.951 → 0.951
worst max |Δlogit|:    4.20e-5
```

State the small-dataset limitation immediately. Use either Figure 3 for
cross-model breadth or Figure 10 for across-seed robustness; do not crowd the
summary with both unless page layout permits.

#### Block 4: method contribution

Explain the orbit null in one sentence and give the calibrated result:

```text
attention columns: +0.0011 [−0.0023, +0.0047]
MLP neurons:       +0.0209 [+0.0121, +0.0297]
```

Use Figure 6. Mention that the metric/statistic evolved after the initial
pre-registration.

#### Block 5: conclusion and limitations

State the necessary-not-sufficient logic. Name two limitations: small models and
a narrow coherence proxy. Mention one refuted prediction, probably the residual
basis null. End with the concrete practice change: compare a direction set to
random rotations within its own span.

### 18.4 What belongs below the executive summary

The main report should then contain:

1. the attention symmetry derivation;
2. numerical verification and raw examples;
3. the invariance matrix and OV sign issue;
4. monitor experiment and multi-seed robustness;
5. orbit-null construction and frequency controls;
6. basis results, including refuted predictions;
7. causal low-dose result;
8. centered-router self-correction;
9. limitations and protocol deviations;
10. reproducibility instructions; and
11. a brief chronological research log.

Move steering, chess, reward hacking, emergent misalignment, and most training
dynamics to an appendix titled "Other investigations and negative results." They
show judgment but dilute the first-pass story.

### 18.5 Which figures to prioritize

If limited to three figures in the executive summary:

1. `fig1_symmetry.png`: proves the separation and shows readout destruction;
2. `fig10_monitor_seeds.png` or `fig3_monitor.png`: shows operational relevance;
3. `fig6_privilege.png`: shows the calibrated method result.

Move the router figure below the executive summary. Check the exported PDF rather
than trusting Markdown length: 572 words plus four wide figures may exceed three
pages.

Figure 1 currently has a slightly crowded/overlapping annotation near its left
title. Fix that before publication.

---

## 19. Mapping the story to likely form questions

### “What question did you try to answer?”

Include:

- interpretability assigns meanings to internal directions;
- coordinates need not be model-intrinsic;
- attention-head basis changes give a known-null experiment;
- the project asks which readouts survive and whether identified bases perform
  better than their own rotations.

Avoid listing every branch or model.

### “Why is this interesting?”

Include:

- most interpretability evaluations lack ground truth;
- a function-preserving transformation supplies one;
- basis dependence threatens portability of readouts and monitors;
- the orbit null is cheap enough to become routine methodology.

Avoid claiming an immediate deployed catastrophe.

### “What conclusions did you reach?”

Use three or four conclusions:

1. head-column readouts fail the symmetry test;
2. the OV product/subspace is invariant, with sign/degeneracy caveats;
3. a fixed head-coordinate monitor fails across transformations while the
   residual control is stable;
4. the orbit-null battery finds MLP privilege but returns the required attention
   null and refutes the residual-axis prediction.

The router correction is a good optional fifth point.

### “What is the strongest evidence against your hypothesis?”

Use results that genuinely changed beliefs:

- residual-stream axes did not show the predicted tail advantage;
- Olmo-3-7B's MLP arm is underpowered;
- the original router explanation changed after centering;
- large-dose causal runs failed the common-mode kill criterion; and
- earlier J-Lens steering results used the wrong singular-vector side or position.

Do not present only small limitations. The question asks for evidence against
you.

### “What are the biggest limitations?”

Name:

- a necessary invariance test is not a semantic certificate;
- mostly small models for the central magnitude estimates;
- one token-coherence operationalization;
- metric/statistic changes after the initial preregistration;
- correlated groups and few independent model families;
- small monitor dataset;
- limited SAE sample; and
- causal interventions concentrated at particular positions and low doses.

Then distinguish limitations addressable with more compute from conceptual ones.

### “How did you use LLMs?”

Write this from personal memory. Include actual examples. Good evidence from the
repository includes:

- the first J-Lens implementation reached cosine `0.94` but still used the wrong
  convention;
- steering originally injected `u` instead of `v`;
- the Qwen head transformation initially missed attention biases and changed the
  model substantially;
- generated labels created overly broad semantic categories;
- figures retained stale annotations; and
- the current verification script itself overstates live recomputation.

Explain which of these you personally caught, how, and what checks you added.
Never claim personal judgment you did not exercise.

### “Why this stream?”

Connect to Neel's interest in pragmatic mechanistic interpretability, J-Lens
red-teaming, model diffing, and falsifiable controls. The compelling personal
reason is not that the keywords match. It is that the project repeatedly forced
you to discard appealing results, and the stream's research culture can help you
develop faster judgment about when a mechanistic claim is real.

---

## 20. Questions a skeptical interviewer may ask

You should be able to answer these without opening the repository.

### 1. Why exactly does the attention transformation preserve the function?

Because the internal coordinate map and its inverse are adjacent in the
value-output computation: `W_O M^{-1} M W_V = W_O W_V`. Attention weights mix
values linearly and do not apply an elementwise nonlinearity in the value basis.

### 2. Why is the numerical logit difference nonzero if the symmetry is exact?

Matrix multiplication order changes floating-point rounding. The difference is
around `1e-5` on logits around `25`, and generated text is unchanged. The algebra
establishes exact equality in real arithmetic; the experiment checks that the
implementation behaves at the expected numerical floor.

### 3. Why does the same rotation fail in an MLP?

Because SiLU and the gating product act coordinatewise. Dense rotation does not
commute with either operation, so the inverse output rotation cannot cancel the
change.

### 4. Does basis dependence mean the attention head has no meaning?

No. It means individual axes in its value/output coordinate system are not
uniquely identified. The head's function, write subspace, OV product, singular
values, and other invariant quantities may still be meaningful.

### 5. Why not just align features after rotation?

Alignment can recover correspondence if the transformation is known or inferred.
But the need for alignment proves the raw coordinate label was not intrinsic.
For portability, a method should either operate on invariant objects or explicitly
solve and report the alignment problem.

### 6. Why is `AH` the orbit null?

Right multiplication mixes the columns of `A` without changing their span. For
attention output columns, such mixing is exactly what a valid internal basis
change produces, with a compensating inverse applied to the value projection.

### 7. Why introduce `QHR` if `AH` already exists?

`AH` changes the column Gram matrix. `QHR` preserves `A^T A` exactly, so it rules
out norms, pairwise angles, and conditioning as explanations for a raw-basis
advantage.

### 8. Why use a different model to score coherence?

If the probed model both produces and judges its token list, the metric can be
circular. An unrelated embedding space reduces that risk. It does not eliminate
all semantic or tokenizer biases, so frequency matching and scorer swaps are
still needed.

### 9. Why use the 90th percentile rather than the mean?

The hypothesis concerns a sparse tail of readable units. A mean over 64 mostly
unreadable directions has low power. But because this statistic was selected
after diagnostics, it must be disclosed and ideally confirmed on fresh held-out
models or groups.

### 10. What does the attention null actually calibrate?

It checks whether the whole basis-generation and coherence-scoring pipeline
manufactures a raw-over-rotation advantage when the architecture says none can
exist. Returning zero makes positive effects in other arms more credible.

### 11. Why is passing a symmetry test not sufficient?

A constant method that always says "nothing" is perfectly invariant and useless.
Invariance rules out certain false interpretations; it cannot prove that a
surviving interpretation captures a real human concept or causal mechanism.

### 12. Is the monitor experiment realistic?

No. It is a controlled counterexample showing that a coordinate-level monitor can
fail on a functionally equivalent model. The dataset and classifier are small,
so the result should motivate invariant monitor design rather than estimate a
real-world failure rate.

### 13. What was the most important negative result?

A strong answer is the residual-basis prediction: it was expected to be weakly
privileged, but the primary tail statistic was essentially zero. Another is the
router conclusion changing after a symmetry-justified centering operation.

### 14. What would you do next?

The cleanest next steps are:

1. confirm the final tail statistic on held-out model families chosen before
   examining results;
2. test published head-level direction claims with the orbit null;
3. build or evaluate monitors using explicitly invariant head features;
4. expand the monitor dataset and separate model, prompt, layer, and rotation
   uncertainty;
5. test more SAE architectures, especially plain ReLU versus top-k designs; and
6. close the steering-specificity question with position, dose, and headroom
   controlled jointly.

---

## 21. A concrete study plan

### Session 1: understand the decisive symmetry, 60–90 minutes

1. Read Sections 3 and 5 of this guide.
2. On paper, derive `W_O M^{-1} M W_V = W_O W_V`.
3. Run:

   ```bash
   PYTHONPATH=. .venv/bin/python verify.py 1
   PYTHONPATH=. .venv/bin/python verify.py 2
   PYTHONPATH=. .venv/bin/python verify.py 3
   ```

4. Read `gauge/matrix.py` lines around `apply_sym`.
5. Explain aloud why head columns change but the OV subspace does not.

Stop if you cannot explain the GQA compensation or SVD sign ambiguity.

### Session 2: understand measurement, 90 minutes

1. Read `jlens/privilege.py` through `make_bases` and `arm_matrices`.
2. Derive why `QHR` preserves the Gram matrix.
3. Read `jlens/tokspace.py` and explain the cross-model coherence score.
4. Read the pre-registration and list every deviation in a notebook.
5. Recreate the main six-row result table from source JSON or report scripts.

### Session 3: understand operational consequences, 60 minutes

1. Read `gauge/monitor_flip.py` line by line.
2. Run the new robustness sweep with fewer seeds if needed:

   ```bash
   PYTHONPATH=. .venv/bin/python -m gauge.monitor_seed_sweep --seeds 8
   ```

3. Explain why splitting by text matters.
4. List at least four reasons the demo is not a realistic monitor benchmark.
5. Read `gauge/sae_audit.py` and derive the plain-ReLU scale symmetry.

### Session 4: understand corrections and negative results, 60–90 minutes

1. Read `gauge/router_centre.py`.
2. Explain why subtracting the mean router row is free.
3. Read the causal dose kill criteria and the discarded saturated run.
4. Review the occupancy retraction and singular-vector steering correction.
5. Write a one-page "things I got wrong" memo without copying repository prose.

### Session 5: write from memory, 90 minutes

Close every draft. Write answers to:

1. What was the question?
2. What did I do?
3. What is the strongest number?
4. Why should anyone care?
5. What result went against me?
6. What is still uncertain?

Then reopen the evidence ledger and correct numbers, not style. Sentences that
survive should be yours.

### Session 6: build the actual submission, 2 hours

1. Produce a maximum-600-word executive summary.
2. Use no more than three summary figures.
3. Export to PDF and verify that the summary occupies at most three pages.
4. Put random raw examples immediately after the summary if interpretation
   quality rests on them.
5. Add the honest time log, contribution statement, and LLM-use description.
6. Run the reproduction commands from a clean environment or soften the claims.
7. Verify that the Google Doc is viewable by anyone with the link.

---

## 22. Reproduction commands and current status

These commands assume the existing local virtual environment and model cache.
They are not yet a complete clean-machine reproduction procedure.

### Fast mathematical check

```bash
PYTHONPATH=. .venv/bin/python tests/test_lens_math.py
```

Observed during preparation of this guide:

```text
max abs diff 5.364e-07 (signal scale 2.052)
causality leak 0.000e+00
PASS
```

### Main attention symmetry

```bash
PYTHONPATH=. .venv/bin/python verify.py 1
```

Observed:

```text
largest change in any logit: 3.58e-05
logit scale: 27.3
same text generated: True
```

### Monitor robustness sweep

```bash
PYTHONPATH=. .venv/bin/python -m gauge.monitor_seed_sweep \
  --model HuggingFaceTB/SmolLM2-135M \
  --layer 15 \
  --seeds 32
```

Outputs:

- `out/gauge/monitor_seed_sweep.json`
- `out/figs_app/fig10_monitor_seeds.png`

### Full headline demo collection

```bash
PYTHONPATH=. .venv/bin/python verify.py
```

Treat this as a mixed live-check/demo script until check 7 and the broad claims
described in Section 17 are corrected.

### Basis experiment

```bash
PYTHONPATH=. .venv/bin/python -m jlens.privilege \
  --model allenai/OLMoE-1B-7B-0924 \
  --groups 14 \
  --draws 16
```

This can depend on large locally cached model weights and precomputed frequency
data. Record exact runtime, hardware, model revision, package versions, and random
seed for a clean submission.

---

## 23. Glossary

**Ablation:** Removing or suppressing a component to test whether behavior
depends on it.

**Activation:** The numerical state produced by a model component on a particular
input.

**Attention head:** A component that computes attention weights, mixes value
vectors across positions, and writes a result back to the residual stream.

**Basis:** A set of axes used to represent vectors in a space.

**Causal intervention:** Deliberately changing an activation or weight and
measuring the resulting output change.

**Confidence interval:** A procedure-derived range expressing sampling
uncertainty under specified assumptions. It does not include every source of
modeling or protocol uncertainty.

**Coordinate:** One number in a vector relative to a chosen basis.

**Eigenvector:** A direction mapped to a scalar multiple of itself by a square
linear operator.

**Gauge freedom:** Multiple internal parameterizations or coordinate descriptions
that compute the same function.

**Gram matrix:** `A^T A`; records pairwise dot products among the columns of `A`.

**Haar random rotation:** A rotation sampled uniformly according to the natural
probability measure on the orthogonal group.

**Headroom:** How much a target probability or logit can increase before
saturation; low initial probability often permits a larger numerical change.

**Jacobian:** Matrix of derivatives describing the local linear effect of one
vector on another.

**Logit:** An unnormalized score for a token before softmax.

**Logit lens:** A technique that maps an intermediate residual vector through the
unembedding to obtain token scores.

**Matched null:** A control distribution preserving important nuisance properties
such as norm, subspace, or pairwise geometry.

**Mechanistic interpretability:** Study of internal algorithms and representations
used by neural networks.

**MoE:** Mixture of experts; a model architecture routing each token through a
subset of expert networks.

**Non-normal matrix:** A matrix that does not commute with its transpose/conjugate
transpose. Its eigenvectors need not be orthogonal, and transient singular-value
amplification can exceed eigenvalue magnitudes.

**Orbit:** The set of objects reachable by applying a family of symmetry
transformations.

**OV circuit:** The composition of an attention head's value and output maps,
abstracting away its arbitrary internal value coordinates.

**Probe:** A classifier or regression model trained on activations to predict a
label.

**Privileged basis:** A basis distinguished by the architecture, typically
because an elementwise operation acts in those coordinates.

**QR decomposition:** Factorization `A = QR`, with `Q` having orthonormal columns
and `R` upper triangular.

**Residual stream:** The shared vector space updated across transformer layers.

**SAE:** Sparse autoencoder, which learns a larger sparse feature representation
that reconstructs model activations.

**Singular value decomposition:** `A = UΣV^T`; describes input directions, output
directions, and associated gains for a linear map.

**Softmax:** Function converting logits into positive probabilities summing to
one; invariant to adding the same constant to every input logit.

**Steering:** Adding a direction to model activations to influence behavior.

**Subspace:** A set of vectors closed under addition and scalar multiplication;
many different bases can span the same subspace.

**VJP:** Vector-Jacobian product, an efficient reverse-mode automatic
differentiation computation of `v^T J` or equivalently `J^T v` depending on
vector convention.

---

## 24. Final checklist before writing the application

You are ready to write only when all answers are "yes."

- Can I derive the attention value/output symmetry on paper?
- Can I explain why GQA requires rotating a whole KV group?
- Can I distinguish a basis from a subspace?
- Can I explain why head columns fail while the OV product survives?
- Can I explain sign ambiguity and near-degenerate singular values?
- Can I derive why `QHR` preserves the Gram matrix?
- Can I explain the difference between orbit and orthonormal nulls?
- Can I describe the coherence metric and its frequency control?
- Can I state which parts were and were not pre-registered?
- Can I explain why the 90th percentile was used and why that choice is a risk?
- Can I state the attention-null and MLP-effect numbers without exaggeration?
- Can I explain the monitor data split and its small-sample limitations?
- Can I explain why 32 rotation seeds are robustness conditions rather than 32
  model replications?
- Can I state why SAE invariance is necessary but not proof of interpretability?
- Can I explain the router centering operation and how it changed the conclusion?
- Can I name three results the project rejected and why?
- Can I state my true active research hours and personal contribution?
- Can I describe LLM usage with concrete examples and no concealment?
- Can a new user reproduce the advertised checks from the instructions I link?
- Does the executive summary fit within 600 words and three rendered pages?

If any answer is no, return to the relevant section before editing the application
draft. Understanding is the deliverable; the application is only its compressed
expression.
