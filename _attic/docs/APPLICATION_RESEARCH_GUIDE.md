# What the averaged Jacobian actually gives you

## A complete guide to the repository and a defensible MATS application

This document is the recommended way to understand and present this project.
If your mental model is still hazy or time is short, begin with the new
[plain-English experiment-to-application guide](SIMPLE_EXPERIMENT_TO_APPLICATION_GUIDE.md).
It follows every main result through experiment, inference, follow-up question,
and larger-model replication.
**Latest reading update:** start with
[the six-question follow-up](FOLLOWUP_QUESTIONS_AND_REWARD_DATA.md), which clarifies
Experiment 5, includes an actual prompt experiment and a new reward-data audit,
and corrects the novelty framing. Jacobian transport itself is established
J-Lens methodology, not our contribution. Chess (section 8) is optional
supplementary material and can be skipped in the main reading path.
For worked equations answering the initial questions about w, ridge, the
paper's spider/ant swap, and dose, see [Steering questions](STEERING_QUESTIONS.md).
For the exact `+Rome -Paris` objective, all 36 prompt/model examples, and its
spectral decomposition, see the
[contrastive-logit experiment](CONTRASTIVE_LOGIT_EXPERIMENT.md).
It assumes no prior knowledge of transformers, Jacobians, J-Lens, steering, or
mechanistic interpretability. It also separates three things that earlier drafts
blurred together: what the original J-Lens paper claims, what this repository
actually tests, and what the evidence supports.

The final, evidence-calibrated conclusion is:

> On three open models, an averaged Jacobian supplies effective early-layer
> steering directions for pre-specified output concepts. The effect is carried
> by many singular components and the estimated direction changes with the
> averaging corpus. These experiments validate a use of J-Lens while warning
> against treating the singular vectors of the averaged matrix as an automatic
> inventory of context-independent features.

That is the application story. The exact-symmetry work is a useful side result
about head-internal coordinates. It is not the main result because the central
experiments in this repository, and most practical interpretability methods,
operate on the residual stream.

The best-fit application stream is **Improved Interpretability Methods**, and in
particular the prompt asking what J-Lens is scientifically doing, how it compares
with simpler methods, and what its failure modes are.

Do not copy prose from this guide into the application form. The application
instructions explicitly say that raw LLM-written summaries are a negative signal.
Use this document to understand the work, close remaining evidential gaps, and
then write the final summary in your own voice.

---

## 1. Start here

If you only have one hour, do the following in order.

1. Read sections 2 through 5 of this document.
2. Open the main application figures:
   - `out/figs_core/fig6_large_model_replication.png`
   - `out/figs_core/fig2_truncated_pullback.png`
   - `out/figs_core/fig5_corpus_conditioning.png`
   - `out/figs_core/fig7_prompt_and_corpus.png`
   - `out/figs_core/fig9_bidirectional_corpus.png`
   - `out/figs_core/fig14_corpus_permutation.png`
   - `out/figs_core/fig3_averaging_and_conditioning.png`
   - `out/figs_core/fig13_smollm2_prompt_gallery.png`
   - `out/figs_core/fig13_qwen3.5_prompt_gallery.png`
   - `out/figs_core/fig13_olmo_prompt_gallery.png`
3. Read `jlens/lens.py`, especially the definition and averaging convention for
   the Jacobian.
4. Read `jlens/pullback_robustness.py`,
   `jlens/pullback_large_replication.py`, `jlens/pullback_corpus.py`,
   and `jlens/pullback_rank.py`. Chess is optional supplementary reading.
5. Explain the project aloud without looking at the text:
   - What is averaged?
   - What is a J-Lens vector in the paper?
   - What is a singular vector of `J`?
   - Why is `J.T @ w` a pullback?
   - Why can a pooled Jacobian mix two computations?

If you cannot answer all five questions precisely, do not start writing the
application yet.

### The one-sentence research question

> How well do averaged-Jacobian directions transport a pre-specified concept
> across layers and contexts, and what is lost when we interpret the matrix's
> leading singular directions as features?

### The one-sentence answer

> It gives useful steering directions across three open models, especially in
> early layers, but the effect is distributed and corpus-dependent. The present
> evidence supports operator utility more strongly than universal feature identity.

### The four pieces of main evidence

1. **Robust, replicated utility:** a pullback constructed with `J.T` beats naive
   residual steering on held-out concept words across four doses and a
   30-direction null in SmolLM2-135M, Qwen3.5-4B, and OLMo-3-7B.
2. **Distributed structure:** 32 to 64 singular components work better than one
   component and slightly better than the full 576-dimensional pullback.
3. **Estimator dependence:** two disjoint, genre-matched prose estimates
   agree closely, while a code-derived estimate rotates and transfers less well
   on both held-out neutral prose and code-like prompts. This shows corpus
   dependence, but not that domain matching is the cause.
4. **Concrete behavior and limits:** on “On the table there was a,” prose-derived
   food steering changes the top next token from “large” to “plate” and raises
   held-out food words. This is a next-token effect, not proof of coherent
   generations, semantic specificity, or improved task performance.

Chess provides an optional controlled example of regime mixing. Reward-hacking
activations provide the strongest proposed application extension, but their
current cross-mechanism result is exploratory and not a headline finding.

### The recommended application identity

Apply to **Improved Interpretability Methods**. The project is best described as
an open-model replication and stress test of J-Lens. Do not claim to have invented
Jacobian transport, J-Lens steering, or activation addition. The original J-Lens
paper already introduces the averaged Jacobian, token-indexed lens vectors,
steering, ablation, and coordinate swaps. Our contribution is the evaluation:
held-out concept words, an explicit no-J comparator, four intervention doses,
30 random directions per cell, replication from 135M to 7B, spectral truncation,
and controlled changes to the estimation corpus.

The future-facing research proposal is sharper than “do more J-Lens”: test how
much behaviorally predictive information survives different constrained
readouts under distribution shift. The reward-hacking trajectories make that
question safety-relevant, while the completed steering work demonstrates that
the applicant can build and audit the underlying method.

---

## 2. The minimum transformer background

### 2.1 Tokens and logits

A language model does not read text directly. A tokenizer converts text into
integer token IDs. The model assigns every possible next token a number called a
logit. Applying softmax to the logits produces probabilities.

If `z_t` is the logit for token `t`, then increasing `z_t` relative to the other
logits makes the token more likely. Many experiments in this repository report a
change in log probability, measured in natural-log units or “nats.” A change of
`+1` nat multiplies the odds by approximately `e`, or 2.72, before accounting for
changes to the competing tokens.

### 2.2 The residual stream

At each token position, a transformer carries a vector of length `d_model`
through the network. Attention blocks and MLP blocks read this vector and write
updates back into it. This shared vector is the residual stream.

For SmolLM2-135M in the main experiments, the residual width is 576. A residual
activation at one layer is therefore a point in a 576-dimensional space. A
residual direction is another 576-dimensional vector. Adding a scaled direction
to the activation is called activation steering.

The final residual stream is mapped to vocabulary logits by a matrix commonly
called the unembedding matrix, `W_U`. Ignoring normalization for a moment,

```text
logits = W_U h_final.
```

The row `W_U[token]` is therefore an output-space direction associated with that
token.

### 2.3 A Jacobian

Suppose `h_l` is the residual stream at layer `l`, and `h_T` is a later residual
state. The Jacobian

```text
J_l = d h_T / d h_l
```

is a matrix describing the first-order effect of a small change at layer `l` on
the later state. For a small perturbation `delta`,

```text
h_T(h_l + delta) approximately equals h_T(h_l) + J_l delta.
```

The Jacobian is local: in a nonlinear network, it depends on the prompt, token
position, attention pattern, and routing decisions present in the forward pass.

J-Lens replaces a local Jacobian with an average:

```text
J_l = E_prompt, source position, later position [d h_T / d h_l].
```

That averaging is not a harmless implementation detail. It determines which
distribution of computations the resulting map describes. This repository's
strongest negative result is about that fact.

### 2.4 Singular value decomposition

Any real matrix can be written as

```text
J = U Sigma V^T.
```

The columns of `V` are directions in the source residual space. The columns of
`U` are directions in the target residual space. A singular value `sigma_i`
tells us how strongly `J` maps the source direction `v_i` into the target
direction `u_i`:

```text
J v_i = sigma_i u_i.
```

This type distinction matters:

- inject `v_i` at the source layer;
- read `u_i` at the target layer;
- do not inject `u_i` merely because it has an interpretable token readout;
- do not call an arbitrary singular component a feature without a causal test.

Several early experiments in the repository mixed up `U` and `V`. The corrected
scripts explicitly preserve the distinction.

### 2.5 The pullback

Suppose `w` is a desired direction at the target, such as the average
unembedding direction for a set of food words. We want a source perturbation `x`
that increases the downstream score `w^T h_T`.

Under the linear approximation, the change in that score is

```text
w^T J x = (J^T w)^T x.
```

Among source perturbations with a fixed Euclidean norm, the one that maximizes
this linearized score is parallel to

```text
x = J^T w.
```

This is the pullback of `w` through the average transport map. The optimality
claim is narrow but exact: fixed norm, first-order objective, and the specific
average Jacobian used in the construction. It does not guarantee specificity,
natural text, or robust transfer to an unrelated prompt distribution.

#### What “seven vector–Jacobian products” means

For a fixed target vector `w`, automatic differentiation can compute `J.T @ w`
without storing every entry of `J`. It differentiates the scalar `w.T @ h_T`
with respect to the earlier activation `h_l`. In conventional row-vector
notation this is the vector–Jacobian product `w.T @ J`; transposing gives the
column vector `J.T @ w` used here.

There are seven concepts, hence seven target vectors and seven products **per
batch, per layer, per corpus**. This does not mean seven Jacobian matrices. In
the corpus experiment, 18 texts are processed as three batches of six, for three
corpora and four layers: `7 × 3 × 3 × 4 = 252` requested gradient products. The
model parameters remain frozen. The computation extracts derivatives; it does
not train the language model.

### 2.6 Why the pseudoinverse is different

It is tempting to solve `J x = w` using the pseudoinverse:

```text
x = J^+ w.
```

If `J` has singular values near zero, the pseudoinverse divides by them. Small,
noisy, or poorly estimated directions receive enormous weight. By contrast,
`J^T w` multiplies each component by its singular value and suppresses the weak
tail.

In the main steering experiment, the pseudoinverse produces essentially zero
held-out lift, while the transpose produces the strongest effect. This is a
concrete demonstration that “solve the inverse problem exactly” is the wrong
objective for a noisy, ill-conditioned average operator.

---

## 3. The vocabulary distinction the application must get right

The repository uses the phrase “J-Lens direction” loosely in some places. The
paper uses it more specifically.

### 3.1 The paper's J-Lens vector

The original paper defines the token-indexed J-Lens vector for token `t` as the
corresponding row of `W_U J`. Written as a source-space column vector, it is

```text
v_t = J^T W_U[t].
```

There is one such vector per vocabulary token. The paper treats the collection
as an overcomplete sparse frame and uses sparse nonnegative decompositions to
define a “J-space.” It does not identify the singular vectors of `J` with those
token-indexed vectors.

### 3.2 A singular direction of `J`

A singular direction is a component of the matrix factorization `U Sigma V^T`.
Its index is determined by the spectrum of `J`, not by a vocabulary token. One
can unembed `u_i` and obtain a list of tokens, but that list is an interpretation
of a matrix component, not the paper's token-indexed feature definition.

### 3.3 A concept pullback

The main positive experiment builds a target direction `w` from multiple concept
words and computes `J^T w`. This is a linear combination of token-related
pullbacks. In the singular basis,

```text
J^T w = sum_i sigma_i <u_i, w> v_i.
```

This identity explains the truncation result. The successful intervention is
not one discovered singular feature. It is a weighted combination of many
components chosen for a pre-specified downstream objective.

### 3.4 The application-safe interpretation

The results support all of the following:

- the averaged Jacobian contains useful information about downstream transport;
- a moderate-dimensional leading subspace can carry most of the useful causal
  effect for these concept interventions;
- the singular basis is useful computationally;
- human-readable token lists for individual singular components do not, by
  themselves, establish feature identity.

The results do **not** establish either of the following:

- that the original paper's token-indexed sparse frame is invalid;
- that residual-stream features in general are arbitrary.

This distinction is one reason the symmetry story should not lead the
application. The valid head rotation leaves the residual stream unchanged. It
audits head-internal coordinate claims, not the residual-stream object studied
by J-Lens.

### 3.5 Four steering and readout methods that must not be conflated

The literature already contains several ways to turn an internal representation
into a readout or intervention. They answer related but different questions.

| Method | How its direction or decoder is obtained | Training required? |
|---|---|---:|
| Logit lens | Apply the model's unembedding directly to an intermediate activation | No |
| Tuned lens | Learn a separate affine map from each layer to later predictions | Yes; the language model stays frozen, but the lens is trained |
| Contrastive Activation Addition | Average residual differences between positive and negative examples | Usually no gradient training for the mean-difference vector |
| J-Lens | Average downstream Jacobians; a token direction is `J.T @ W_U[token]` | No parameter fitting, but backward differentiation is required |

Our multi-word concept direction first averages centered, normalized unembedding
rows for a set of construction words to obtain `w`, then transports it with
`J.T @ w`. This is a direct extension of token J-Lens steering, not a new family
of activation intervention. A proper future comparison should include a tuned
CAA baseline built from contrastive prompts. Directly injecting `w` is an
important no-J control, but it is not the strongest known activation-steering
baseline in the literature.

### 3.6 Exactly how the original J-Lens paper steers

The paper uses two interventions. First, for vocabulary token `t`, form its
source-layer J-Lens vector

```text
v_t = J_l.T @ u_t,
```

where `u_t` is the token's row of the unembedding matrix, represented as a column
vector. It then adds a scaled copy to the residual activation:

```text
h_l' = h_l + alpha * v_t.
```

The paper applies this at selected layers and token positions. A negative scale,
or projecting out the component parallel to `v_t`, is used for ablation. `v_t`
is not “the activation at the final layer”; it is a fixed source-space direction
derived from an average derivative and a token output direction.

The ant/spider demonstration uses a more controlled coordinate swap. For source
concept vector `v_s` and target concept vector `v_t`, put them in the two-column
matrix

```text
V = [v_s, v_t].
```

Given the current activation `h`, compute its two least-squares coordinates

```text
c = V^+ h,
```

where `V^+` is the Moore–Penrose pseudoinverse. Here the pseudoinverse is only
solving for coordinates in a two-vector span; it is not inverting the full
downstream Jacobian. Let `swap(c)` exchange the two entries. The edited state is

```text
h' = h + V (swap(c) - c).
```

Thus the component inside the ant/spider span is changed while the component
orthogonal to that span is left unchanged. In the paper's example, the prompt
asks for the number of legs of the animal that spins webs. Swapping the inferred
spider coordinate for ant changes the answer from 8 to 6. This is stronger than
merely increasing an ant logit, because it redirects an intermediate concept
used by a later computation. It is also an already-published experiment; cite it
as motivation, not as our result.

---

## 4. What the repository contains

The repository is best understood as a sequence of attempts to answer “what is
a direction really?” It contains far more experiments than a single application
can carry. The job is to find the spine and use the other branches as evidence
of judgment.

### 4.1 Core Jacobian implementation

| File | Role |
|---|---|
| `jlens/lens.py` | Computes local and averaged residual-stream Jacobians, including multi-layer variants. |
| `tests/test_lens_math.py` | Checks the Jacobian construction against direct differentiation and checks causal masking. |
| `tests/test_identity_at_target.py` | Verifies that the target-to-target Jacobian is the identity. |
| `out/Jall_smollm2.pt` | Saved Jacobians for even layers of SmolLM2-135M, width 576. |
| `out/Jall_main.pt` | Saved dense OLMo Jacobians, width 4096. |
| `out/Jall_olmoe.pt` | Saved OLMoE Jacobians, including disjoint prompt halves. |

The local math test previously reported a maximum discrepancy of about
`5.4e-7` at a signal scale of about `2.05`, and zero causal leakage. Those are
implementation checks, not scientific validation.

### 4.2 The positive operator experiments

| File | Question |
|---|---|
| `jlens/pullback_steer.py` | Does `J.T @ w` beat the same target direction injected directly? |
| `jlens/pullback_robustness.py` | Does the result survive four doses and a 30-direction random null? |
| `jlens/pullback_large_replication.py` | Does the same controlled result replicate with a published Qwen3.5-4B lens and a locally computed OLMo-3-7B lens? |
| `jlens/pullback_corpus.py` | Are pullbacks stable across matched corpora, and how does code-to-prose distribution shift affect them? |
| `jlens/pullback_rank.py` | How many singular components of the pullback are useful? |
| `jlens/subspace_meaning.py` | Does the leading input subspace concentrate topic information better than activation PCA or random subspaces? |
| `jlens/selection_check.py` | Does per-direction topic separation survive a best-of-64 null and held-out prompts? |

### 4.3 The individual-direction audit

| File | Question |
|---|---|
| `jlens/steer_interp.py` | Does a more readable singular component steer its apparent concept more often? |
| `jlens/admissible.py` | Are failures explained by prompts that do not admit the concept? |
| `jlens/what_works.py` | Do singular value, rank, readability, or geometry predict success? |
| `jlens/leftover.py` | Is interpretable structure hiding outside the usual leading subspaces? |

These experiments concern SVD components of `J`. They should not be described as
a direct test of every token-indexed J-Lens vector in the paper.

### 4.4 The chess mechanism

| File | Question |
|---|---|
| `jlens/chess_j.py` | Build the residual Jacobian for a chess model. |
| `jlens/chess_concepts_big.py` | Which oracle-computable position properties correlate with component coefficients across 7,758 positions? |
| `jlens/chess_cond.py` | How different are origin-square and destination-square Jacobians? |
| `jlens/chess_q1q2.py` | Do pooled or matched conditional Jacobian subspaces align with an oracle-defined legality-gradient subspace? |
| `jlens/piece_shared.py` and `jlens/piece_ablate.py` | Are apparent piece subspaces specific to their named piece? |

Chess matters because the labels do not come from an LLM or a researcher's
judgment. `python-chess` supplies legal moves. The model assigns over 99.5% of its
square-token probability mass to legal squares in the sampled positions, so the
task is behaviorally real for the model.

### 4.5 Estimator stability and MoE routing

| File | Question |
|---|---|
| `jlens/moe_jall.py` | Do Jacobians estimated on disjoint prompt halves have the same leading subspace? |
| `jlens/moe_route.py` and `jlens/moe_ahead*.py` | Does the Jacobian predict later expert routing better than simple persistence? |
| `jlens/moe_condj.py` | Do routing-defined clusters correspond to stable local linear regimes? |

This branch shows that “the averaged Jacobian” is an estimator, not a Platonic
property of the model. Prompt choice and routing distribution affect it.

### 4.6 Methodological controls

`docs/POSITION_IS_THE_VARIABLE.md` records a critical lesson: steering results
depend strongly on the injection token and dose. In one comparison, the same
direction and dose changed the target by `+0.58` nats at the subject token but
`-3.24` at the final token in SmolLM2. Dose also reversed the measured
relationship between efficacy and specificity.

The recommended protocol that emerged is:

1. name the injection position precisely;
2. sweep multiple doses;
3. compare with at least 30 random directions per cell;
4. account for the target's baseline probability or headroom;
5. measure unrelated-text damage against the unedited model;
6. use held-out examples that were not used to construct or select the direction.

The first pullback experiment satisfies the held-out concept-word control and
compares against direct steering, ridge, pseudoinverse, a best singular
component, and random. The follow-up in `pullback_robustness.py` closes two
important gaps: it uses four doses and 30 random directions per layer. It still
uses a 12-prompt evaluation bank, one intervention site rule, and correlated
concept-level tests. The same controlled comparison has now been replicated on
Qwen3.5-4B and OLMo-3-7B; those replications remove the one-small-model objection
but retain the shared prompt/concept and intervention-design limitations.

### 4.7 Side branches

The repository also contains:

- supervised subspace learning attempts, including DAS and generalized
  eigenvectors;
- edit-site localization against causal tracing;
- training-checkpoint Jacobian dynamics;
- emergent-misalignment fine-tune comparisons;
- reward-hacking activation analysis;
- router-basis and exact-symmetry audits;
- dashboards, galleries, and report generators.

These show breadth and willingness to kill hypotheses. They should appear below
the executive summary as a compact “other attempted directions” section, not as
co-equal stories.

---

## 5. Experiment 1: the Jacobian works as a pullback operator

### 5.1 Question

If we specify an output concept first, can the average Jacobian find a better
residual intervention than simply adding the output direction at the source
layer?

### 5.2 Model and data

- Model: `HuggingFaceTB/SmolLM2-135M`
- Source layers: 4, 12, 20, and 26
- Residual width: 576
- Concepts: food, medicine, programming, law, sports, finance, and female
- Evaluation prompt bank: 12 neutral text prefixes
- Total concept-layer cells: 28
- Intervention dose: 0.15 times the estimated median residual norm at that layer

For each prompt, the script takes the median norm of valid residual vectors at
that layer, excluding the first token and padding, then averages those 12 prompt
medians to obtain the layer scale `s_l`. For a raw direction `d`, the exact edit is

```text
delta_l = alpha * s_l * d / ||d||
h_l'[position] = h_l[position] + delta_l
```

with `alpha = 0.15` in the reference experiment. The same `delta_l` is added to
every prompt position at the output of one selected transformer block. “0.15”
therefore means 15% of this residual-norm reference. It is not 15% of the
activation coordinates, a probability, or “15% more concept.”

Each concept begins as a list of roughly 20 words. The tokenizer keeps words with
a single-token representation. The token IDs are split by alternating index:
one half constructs `w`, and the other half scores the intervention. Therefore,
the scored words did not directly determine the direction.

More precisely, let `u_t` be the unembedding row for construction token `t`.
The implementation first subtracts the mean unembedding row across the whole
vocabulary, normalizes every resulting token row, averages the rows belonging to
the construction words, and normalizes again:

```text
u_t_centered = u_t - mean_vocabulary_unembedding
u_t_unit = u_t_centered / ||u_t_centered||
w = mean_{t in construction words}(u_t_unit)
w = w / ||w||
```

Thus `w` is a fixed target-space concept direction made from words, not an
activation collected from one prompt and not a learned classifier weight.

For every held-out concept token, the script selects a control token with a
similar baseline likelihood and low cosine similarity to `w`. The reported lift
is

```text
mean change on held-out concept tokens
minus
mean change on rank-matched control tokens.
```

This is substantially stronger than asking whether the exact token used to
construct the direction goes up. That test is nearly guaranteed by construction.

### 5.3 Methods compared

The experiment compares:

- `J.T @ w`, the pullback;
- `w` injected directly at the source layer, the essential no-J baseline;
- `J^+ @ w`, the pseudoinverse;
- ridge-regularized inverse solutions at three regularization strengths;
- the single singular component whose target-side direction is most aligned
  with `w`;
- a random source direction with the same intervention norm.

### 5.4 Results

![Pullback operator comparison](../out/figs_core/fig1_pullback_operator.png)

| Method | Mean held-out lift | Cells beating the cell's random control and 0.2 | Mean unrelated-text loss increase |
|---|---:|---:|---:|
| `J.T @ w` | **1.64** | **28/28** | +0.172 |
| Ridge, lambda 1 | 0.85 | 25/28 | +0.134 |
| Best single SVD component | 0.76 | 20/28 | +0.112 |
| `w` without `J` | 0.64 | 17/28 | +0.209 |
| Pseudoinverse | 0.00 | 3/28 | +0.022 |
| Random | 0.01 | 0/28 | +0.031 |

The mean pullback lift is about 2.6 times the direct residual baseline. The
advantage is depth-dependent:

| Source layer | Direct `w` | Pullback `J.T @ w` | Ratio |
|---:|---:|---:|---:|
| 4 | 0.11 | 0.56 | 5.2x |
| 12 | 0.16 | 1.21 | 7.7x |
| 20 | 0.68 | 2.29 | 3.4x |
| 26 | 1.60 | 2.51 | 1.6x |

The methods converge near the target, where residual connections make `J`
closer to identity. This was not used to define the metric and is a useful
internal consistency check.

#### Robustness follow-up: dose response and a multi-random null

`jlens/pullback_robustness.py` repeats the essential `J.T @ w` versus direct
`w` comparison at four intervention doses. At the preselected reference dose
of 0.15 residual norms, it also evaluates 30 isotropic random directions per
layer. Each random edited forward pass is reused to score all seven concepts,
so the concept-level nulls within a layer are correlated.

![Pullback robustness](../out/figs_core/fig4_pullback_robustness.png)

| Dose, as fraction of median residual norm | Pullback mean lift | Direct `w` mean lift | Positive pullback cells |
|---:|---:|---:|---:|
| 0.05 | 0.571 | 0.230 | 28/28 |
| 0.10 | 1.132 | 0.442 | 28/28 |
| 0.15 | **1.641** | 0.636 | 28/28 |
| 0.20 | 2.084 | 0.815 | 28/28 |

At dose 0.15, all 28 pullback cells exceed all 30 sampled random directions;
the median per-cell z-score against the random distribution is 8.31. Direct
`w` exceeds all sampled random directions in 16/28 cells and has median z-score
2.58. With only 30 draws, the smallest possible one-sided empirical p-value is
`1/31 = 0.032`; these are per-cell descriptive tests, not a family-wise-corrected
claim of 28 independent discoveries. The stronger and simpler statement is:

> Across the entire fixed grid, every pullback effect was positive at
> every dose, and at the reference dose every pullback exceeded every one of the
> 30 matched random-direction effects sampled for its layer and concept.

#### Largest-local-model replication

`jlens/pullback_large_replication.py` runs the identical causal protocol on two
larger architectures. Qwen3.5-4B uses the external published J-Lens artifact
from `camilablank/workspace-lenses`, estimated by the original project on 25
Pile documents. OLMo-3-1025-7B uses `out/Jall_main_fp16.pt`, the repository's
full 4096-dimensional Jacobian estimated on 25 Pile documents with the same
position convention. The causal evaluation uses the same seven concept word
lists, alternating construction/test split, 12 neutral prompts, rank-matched
controls, four doses, and 30 random directions per layer.

Layers were chosen at comparable fractions of the target depth:

- SmolLM2: 4, 12, 20, 26, with target 28;
- Qwen3.5-4B: 4, 13, 21, 28, with target 30;
- OLMo-3-7B: 4, 12, 22, 28, with target 30.

![Large-model replication](../out/figs_core/fig6_large_model_replication.png)

| Model | Parameters | Pullback lift at dose 0.15 | Direct `w` | Pullback/direct | Pullback cells exceeding all 30 random effects |
|---|---:|---:|---:|---:|---:|
| SmolLM2 | 135M | **1.641** | 0.636 | 2.58x | 28/28 |
| Qwen3.5 | 4B | **1.437** | 0.730 | 1.97x | 28/28 |
| OLMo-3 | 7B | **1.261** | 0.669 | 1.88x | 28/28 |

Every pullback cell is positive at every tested dose in every model. At the
reference dose, all 84 cells exceed every one of their 30 sampled random
effects. Median per-cell null z-scores are 8.31, 16.67, and 23.26 respectively.
The direct baseline clears the same empirical threshold in 16/28, 26/28, and
21/28 cells.

The depth crossover also replicates. In Qwen, the mean pullback/direct ratio
falls from 4.39x at layer 4 to 1.28x at layer 28. In OLMo, it falls from 15.79x
to 1.25x. Absolute early-layer direct effects are small, so the 15.79x ratio
should not be sold as an effect-size headline; the robust observation is that
all three methods converge toward a ratio near one as the source approaches the
target.

This is now the strongest application result. It uses two independently
estimated language-model Jacobians, one of them an external published artifact,
and spans a roughly 52-fold parameter range. It remains a replication of one
metric and one intervention protocol, not evidence that pullbacks are
semantically specific or harmless.

### 5.5 What this establishes

Across three models, the averaged Jacobian is not decorative. It improves a
held-out-word steering objective over directly adding the target direction,
survives a dose sweep and random null, and exhibits the expected depth
crossover. On SmolLM2, the transpose is also more useful than trying to invert
the operator.

### 5.6 What this does not establish

The pullback is the gradient direction of the average linearized objective. A
positive direct effect is therefore partly built into its construction. The
scientific content comes from the no-J baseline, the held-out concept words, the
depth pattern, the finite intervention, the inverse failure, and the truncation
experiment—not from the bare fact that moving along a gradient changes its
objective.

The experiments also do not show semantic specificity. Earlier capital-city
tests found that a France-to-Rome intervention also moved Spain, Germany, and
Japan. Target movement can be driven by headroom: low-probability tokens have
more room to rise. The follow-up adds multiple random directions and a dose
sweep, but a stronger version still needs base-probability regression,
paraphrases, position-specific interventions, correction for correlated tests,
and replication of semantic-specificity—not merely efficacy—across models.

### 5.7 Novelty relative to the paper

The paper already steers with token J-Lens vectors, and `J.T @ w` for a single
token is exactly such a vector. Do not claim that pullback steering itself is
new.

The application-safe description is “replication and evaluation,” not “a new
steering method.” The potentially original empirical contributions are:

- a head-to-head comparison with injecting the same target direction without
  `J`;
- the depth-dependent crossover as `J` approaches identity;
- the held-out multi-word concept protocol;
- the pseudoinverse failure and ridge sequence;
- the observation that truncating the operator can improve the intervention.

These comparisons should not be described as exhaustive state of the art. In
particular, the current experiments do not compare against Contrastive Activation
Addition or a trained probe/tuned lens. The direct-`w` comparator isolates the
value of Jacobian transport relative to a simple same-target direction.

### 5.8 A concrete prompt, with actual probabilities

The aggregate metric can feel abstract, so `jlens/application_prompt_example.py`
reruns one intelligible SmolLM2-135M case. It uses layer 20, dose 0.15, the food
direction, and the prompt:

> On the table there was a

The construction half includes tokens such as vegetables, cooking, meal, rice,
kitchen, dinner, baking, soup, restaurant, and pasta. The following scored words
were held out from the construction:

| Next-token probability | Unmodified | Direct `w` | Pooled-prose `J.T @ w` | Code-estimated `J.T @ w` |
|---|---:|---:|---:|---:|
| bread | 0.0118% | 0.0459% | 0.1115% | 0.0762% |
| cheese | 0.0093% | 0.0372% | 0.1374% | 0.0865% |
| sauce | 0.0143% | 0.0672% | 0.3682% | 0.1643% |
| salad | 0.0108% | 0.1002% | 1.4145% | 0.3649% |

The clean model's highest-probability next token is “large” at 4.25%. Under the
pooled-prose pullback, it becomes “plate” at 6.94%. Under the code-derived
pullback, “plate” is also first, at 7.97%, even though its aggregate held-out-food
score is lower. This illustrates why one top token and the predeclared multiword
metric should not be conflated.

![Prompt-level effects and actual probabilities](../out/figs_core/fig7_prompt_and_corpus.png)

The intervention is applied at every existing prompt position, and the script
then measures the next-token distribution once. It does not sample a completion.
The example was selected because it is easy to understand, so it should be
presented as an illustration rather than a random-case estimate.

---

## 6. Experiment 2: the useful effect is distributed across a subspace

### 6.1 Question

Is the pullback effect carried by one interpretable singular component, a small
subspace, or the full spectrum?

### 6.2 Construction

Using `J = U Sigma V.T`, write

```text
J.T @ w = sum_i sigma_i <u_i, w> v_i.
```

The experiment keeps only the first `k` components, for

```text
k = 1, 2, 4, 8, 16, 32, 64, 128, 256, 576.
```

Each truncated vector is normalized to the same intervention dose. This means
the comparison tests the direction of the truncated pullback, not the raw norm
captured by the prefix.

The sweep uses layers 12 and 20, seven concepts, and the same held-out-word
metric, for 14 observations at every `k`.

### 6.3 Result

![Truncated pullback](../out/figs_core/fig2_truncated_pullback.png)

| Components retained | Mean lift | Share of full pullback energy |
|---:|---:|---:|
| 1 | -0.13 | 1.6% |
| 8 | 1.11 | 16.9% |
| 16 | 1.55 | 32.5% |
| 32 | **1.83** | 49.5% |
| 64 | **1.89** | 69.3% |
| 128 | 1.84 | 84.8% |
| 256 | 1.78 | 95.8% |
| 576 | 1.75 | 100% |

One component is insufficient. The best result uses 64 components, and 32 to 64
slightly outperform the full vector. The smallest singular-value tail adds
energy but reduces the measured effect.

The new larger-model replication changes this conclusion. On Qwen3.5-4B, mean
lift rises from 0.162 at one approximate component to 1.529 at 256 and 1.641 for
the exact full pullback. On OLMo-3-7B it rises from 0.199 to 1.133 and 1.415.
Finite-`k` large-model directions use a seeded randomized low-rank SVD; the full
endpoint is exact. The robust conclusion is therefore “the effect is distributed,”
not “64 components are universally optimal.”

![Cross-model truncation](../out/figs_core/fig10_cross_model_rank.png)

### 6.4 Interpretation

This is the cleanest reason to say “useful subspace” rather than “single
feature.” The pullback is a coordinated combination of source directions. A
human-readable label on one component is not the same object as the effective
intervention.

The result also explains the pseudoinverse failure. The transpose suppresses
small singular values, while the pseudoinverse magnifies them. Truncation removes
the weakest tail completely.

### 6.5 A related decoding result

`jlens/subspace_meaning.py` asks whether topic information is concentrated in
the top input singular directions. It uses 72 prompts from six topics and a
leave-one-out nearest-centroid classifier. At eight dimensions:

| Layer | Top 8 of `J` | Top 8 activation PCs | Random 8D subspace | Shuffled labels |
|---:|---:|---:|---:|---:|
| 4 | **76%** | 47% | 34% | 15% |
| 12 | 79% | **90%** | 54% | 22% |
| 20 | 85% | **88%** | 51% | 19% |
| 28 | 69% | **97%** | 71% | 12% |

The early-layer result is interesting: `J` ranks directions by downstream
sensitivity, while PCA ranks directions by activation variance. At layer 4, the
first criterion concentrates more topic information in eight dimensions. At the
target layer, `J` approaches identity, its spectrum becomes degenerate, and its
singular directions lose a privileged ordering. The layer-28 result becomes no
better than a random eight-dimensional subspace, which is the expected null.

This experiment was rerun during this audit and saved as
`out/rare/subspace_meaning_full.json`.

The same test was then run on the larger models using seeded randomized rank-80
SVDs for the Jacobian subspaces. At layer 4 and eight dimensions, Qwen obtains
38.9% from `J` versus 98.6% from PCA and 48.9% from random; OLMo obtains 62.5%,
97.2%, and 37.0%. Thus the early SmolLM advantage does not replicate, although
OLMo's J-subspace remains above random.

![Cross-model topic concentration](../out/figs_core/fig11_cross_model_topic_subspace.png)

---

## 7. Experiment 3: readable singular components are not reliable causal features

### 7.1 Question

If the target-side singular direction `u_i` unembeds to a coherent cluster of
tokens, does injecting the corresponding source direction `v_i` reliably move
other tokens from the same concept?

### 7.2 Avoiding the circular test

If `u_i` has token `t` as its strongest readout, then pushing `v_i` often raises
`t` almost by arithmetic. That cannot validate the semantic label.

`jlens/steer_interp.py` instead forms a centroid from the top readout tokens,
excludes the top 200 tokens used in the readout, finds nearby held-out tokens,
matches controls by baseline probability, and tests whether the held-out tokens
rise more than the controls.

### 7.3 Result

For the leading singular components, directions were divided into thirds by a
cross-model readability score:

| Readability band | Mean readability | Directions passing the held-out steering test |
|---|---:|---:|
| High | 47.5 | 8/20, 40% |
| Middle | 11.3 | 8/20, 40% |
| Low | 3.4 | 7/20, 35% |

A fourteen-fold change in readability produced essentially no change in pass
rate. Across the readable directions re-tested on a 40-prompt bank, 19 of 37
passed the whole-bank criterion. Prompts that initially made the concept more
likely did not rescue the failures; the lower-baseline prompts showed a larger
mean lift, consistent with headroom rather than semantic admissibility.

The repository also reports split-half reliability of roughly `r = 0.94` and
81% agreement on the direction-level verdict. If that analysis is used in the
application, its exact producing artifact should be restored and rerun. The
qualitative point is that the mixed success is not obviously random run noise.

### 7.4 Interpretation

The safe conclusion is not “J-Lens features fail.” The experiment concerns
singular components of the average operator. The result says that readable
unembeddings of those components do not justify treating the components as
standalone semantic variables.

This fits the positive result rather than contradicting it. Individual
components are weak and unreliable, but the target-chosen combination
`J.T @ w` works.

---

## 8. Optional supplementary experiment: averaging can mix computations

### 8.1 Why chess can be a useful diagnostic

Natural-language feature labels are subjective. Chess provides objective
variables:

- whether the next square is a legal origin;
- whether it is a legal destination for the already-selected piece;
- whether a capture is available;
- whether a rook or queen can move;
- the gradient of total legal-square score versus illegal-square score.

The chess tokenizer emits a move as two square tokens. The model therefore
alternates between two known tasks:

1. choose the square a piece moves from;
2. choose the square that piece moves to.

This creates a clean test of whether one pooled Jacobian represents both
computations.

### 8.2 What the strongest coefficients track

`jlens/chess_concepts_big.py` uses 150 generated games and 7,758 positions. For
each of seven layers and the top 24 components, it correlates the
position-dependent coefficient with oracle labels after removing a linear trend
in ply number. Significance is calibrated with 400 permutations of the maximum
absolute correlation over the entire family of tests; the 95th-percentile
family-wise threshold is 0.048.

The strongest absolute correlation for each selected label is:

| Label | Best absolute correlation |
|---|---:|
| Origin versus destination slot | **0.550** |
| Queen can move | 0.392 |
| Capture available | 0.274 |
| Rook can move | 0.268 |

The strongest signal is the phase of the output format. It is not a chess
concept in the ordinary sense.

The repository contains an older summary reporting multivariate R-squared values
of 0.66 for output slot, 0.31 for queen mobility, 0.22 for rook mobility, and
0.16 for capture availability. The JSON exists, but no checked-in script
produces it. Do not put those R-squared values in the headline application until
the producer is restored and rerun. The fully reproducible correlation result is
enough.

### 8.3 Conditional Jacobians differ

`jlens/chess_cond.py` computes three Jacobians from the same sampled games:

- `J_origin`, using only positions that predict origin squares;
- `J_dest`, using only positions that predict destination squares;
- `J_pooled`, using both.

At layer 1, the flattened cosine similarity between `J_origin` and `J_dest` is
0.211, while their difference has norm 1.62 times the pooled matrix's norm. The
similarity rises later in the model and reaches 0.961 at layer 6. The strongest
mixture problem is therefore early, where the two tasks use very different local
transport maps.

### 8.4 Does conditioning help with an objective downstream subspace?

`jlens/chess_q1q2.py` defines a legality score at a position:

```text
logsumexp(logits on legal squares)
minus
logsumexp(logits on illegal squares).
```

It differentiates this score with respect to the residual stream, producing an
oracle-defined legality gradient for each position. It then compares the leading
subspace of those gradients with:

- the pooled Jacobian subspace;
- the parity-matched conditional Jacobian subspace;
- the activation PCA subspace;
- 30 random subspaces;
- a split-half stability ceiling for the gradient subspace itself.

The run contains 400 positions, 200 per regime, at layers 5 and 6. The model's
mean legal-square mass is 0.995 for origins and 0.997 for destinations.

![Averaging and conditioning](../out/figs_core/fig3_averaging_and_conditioning.png)

At destination positions in layer 5:

| Rank | Pooled `J` | Conditional `J_dest` | Activation PCA | Random | Gradient split-half ceiling |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.098 | **0.239** | 0.040 | 0.008 | 0.513 |
| 16 | 0.067 | **0.127** | 0.117 | 0.031 | 0.396 |
| 64 | 0.135 | 0.178 | **0.225** | 0.125 | 0.390 |

At rank 4, conditioning improves overlap by 2.4 times and beats both PCA and the
random-subspace baseline. At rank 64, PCA is stronger. Conditioning is therefore
a partial correction, not a solved problem.

For origin positions, the matched conditional Jacobian does not consistently
beat the pooled one. This asymmetry is important. A clean mechanism should make
different predictions for different regimes; the data do not support the claim
that simple parity conditioning universally fixes J-Lens.

### 8.5 The legality-gradient subspace is not tiny

The same experiment asks how many principal components explain 90% of the
normalized legality-gradient variance:

| Regime | Layer 5 | Layer 6 |
|---|---:|---:|
| Origin | 67 | 47 |
| Destination | 85 | 66 |

This reinforces the subspace result. The task-relevant causal geometry is
moderate-rank. Looking for one beautiful direction is the wrong inductive bias.

### 8.6 Interpretation

A pooled Jacobian is an average over local linear maps. If the prompt
distribution contains several distinct computational regimes, its leading
directions can emphasize variation between regimes. In chess, output parity is
both easy to identify and causally associated with different local maps. The
pooled decomposition therefore highlights format before much of the domain
content.

The correction is not “never average.” The pullback experiment shows that the
global average can still be useful. The correction is:

> Treat the averaging distribution as a model choice. Measure estimator
> stability, inspect known regimes, and compare a pooled lens with conditional
> lenses on held-out causal tasks.

---

## 9. Experiment 5: the averaged Jacobian changes across sampled distributions

The original split-half result below confounded sampling noise with corpus
shift. `jlens/pullback_corpus.py` now tests the claim more directly without
materialising a full Jacobian. It computes seven vector-Jacobian products per
layer, one for each concept direction, on three 18-prompt corpora:

- `prose_a` and `prose_b` are disjoint but paired by broad genre;
- `code` is an intentional domain shift;
- all causal effects are evaluated on the separate 12-prompt neutral bank used
  in Experiment 1.

The two prose halves contain different sentences about matched topics such as
weather, history, biology, travel, law, finance, medicine, and engineering.
The code bank contains Python, JavaScript, SQL, Rust, and Java-like snippets.
No evaluation prompt appears in an estimation corpus.

![Corpus-conditioned pullbacks](../out/figs_core/fig5_corpus_conditioning.png)

### 9.1 Geometry

| Pullback comparison | Mean cosine across 28 concept-layer cells |
|---|---:|
| Disjoint matched prose A versus B | **0.953** |
| Prose A versus code | 0.687 |
| Prose B versus code | 0.689 |
| New pooled prose versus saved global `J` | 0.878 |

The distribution effect is depth-dependent. Prose-versus-code cosine rises from
0.498 at layer 4 to 0.941 at layer 26, as every source-to-target operator becomes
closer to the same near-identity map. The matched prose A/B cosine rises from
0.871 to 0.999 over the same layers.

### 9.1.1 Prompt-level permutation test

Cosine differences alone do not establish that the effect exceeds finite-corpus
sampling noise. `jlens/corpus_conditioning_permutation.py` therefore saves the
individual `J(x)^T w` pullback for each of 54 estimation prompts and performs a
5,000-shuffle label-permutation test. Its global statistic is the sum, across
the 28 concept-layer cells, of the squared difference between corpus means
divided by within-corpus squared variation.

| Comparison | Mean cosine | Normalized mean shift | Global permutation p | FDR-significant cells |
|---|---:|---:|---:|---:|
| Prose A versus prose B | 0.953 | 0.058 | 1.000 | 0/28 |
| Prose A versus code | 0.687 | 0.819 | 1/5001 | 28/28 |
| Prose B versus code | 0.689 | 0.829 | 1/5001 | 28/28 |

![Prompt-level corpus permutation test](../out/figs_core/fig14_corpus_permutation.png)

This is strong statistical evidence that these prose and code samples produce
different mean pullbacks, beyond the finite-sample variation seen between the
matched prose banks. It is not strong proof that a single latent variable named
“domain” caused the difference: corpus label is confounded with syntax,
vocabulary, topic mix, length, and punctuation, and only one hand-built code
bank was tested. The global permutation result is the primary test; the 28 cell
tests are supporting localization and are correlated with one another.

There is also a definitional point that should not be confused with this
empirical claim. A local Jacobian is `J(x)`, a function of its input in a
nonlinear network. Once we define an average lens as
`J_D = E[x sampled from D][J(x)]`, its value can mathematically change with
`D`. What needs evidence is whether that change is large, reproducible,
causally important, and attributable to a meaningful distribution property.
This experiment strongly supports the first two for these banks; the causal
transfer table supports practical importance; attribution remains open.

### 9.2 Held-out causal transfer

Every direction is normalized to the same 0.15-residual-norm intervention before
evaluation, so differences are not caused by raw pullback norm.

| Direction source | Mean held-out lift |
|---|---:|
| Saved global `J` | **1.641** |
| Newly pooled prose | 1.614 |
| Prose A | 1.604 |
| Prose B | 1.605 |
| Code | 1.109 |
| Direct `w`, no `J` | 0.636 |

The two prose estimates differ by only -0.001 on average. Pooled prose beats the
code-estimated direction in 27/28 cells, by +0.505 lift on average. Yet the code
direction still beats direct `w` in 26/28 cells. This is the most informative
result for the central question: averaging does preserve useful residual-stream
signal, but the transported direction and its transfer quality depend on the
estimation distribution.

This includes randomized label inference conditional on the observed prompts,
but it is not a randomized corpus-construction experiment. The prose and code banks were
manually constructed with AI assistance during this audit, are small, and differ
in syntax, content, and token-length profile. Disclose that provenance in any
application that uses the result.
The result demonstrates a controlled and statistically clear difference between
these sampled distributions; it does not estimate a population effect for
“code versus prose.” A larger study should sample many independent,
token-count-matched corpus replicates rather than treat 18 texts as exhaustive.

The batched Jacobian computation was checked against a batch-size-one
recalculation at layer 20. The maximum absolute discrepancy was `1.2e-7` for
reported cosines and `1.4e-6` for causal lifts. `jlens/lens.py` now masks padded
destination positions and computes source indices relative to each sequence, so
padding cannot contribute spurious downstream gradients.

### 9.3 Supporting small split-half subspace result

`jlens/moe_jall.py` estimates full `J` matrices on two disjoint halves of a
12-prompt corpus. It compares the leading right-singular subspaces at matched
fractions of model width.

| Model | Layer | Top 2% overlap | Top 5% | Top 10% | Top 25% |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M, dense | 4 | 0.61 | 0.68 | 0.77 | 0.79 |
| SmolLM2-135M, dense | 12 | 0.37 | 0.43 | 0.54 | 0.71 |
| OLMoE-1B-7B | 4 | 0.26 | 0.32 | 0.40 | 0.53 |
| OLMoE-1B-7B | 12 | 0.30 | 0.41 | 0.51 | 0.65 |

An overlap of 1 would mean identical subspaces. The MoE estimate is less stable
early, plausibly because changing prompts changes which experts are routed. But
even the dense model is far from perfect agreement. With only six prompts per
half, estimator noise and corpus mismatch are confounded.

The application-safe conclusion from this older result is:

> The leading subspace of an estimated average Jacobian can vary substantially
> across small disjoint corpora, especially in an MoE. Larger-corpus bootstrap
> experiments are needed to separate sampling error from genuine distribution
> dependence.

Do not write “the MoE J-Lens is half as reproducible” as if it were a stable
population estimate. The sample is too small for that precision.

### 9.4 Reverse-domain evaluation added in the final audit

The first analysis evaluated every direction on neutral prose prefixes. That
cannot establish that a direction works best when its estimation domain matches
its deployment domain. A final follow-up therefore kept the same seven concepts,
four layers, held-out target words, controls, and equal dose, but changed the 12
evaluation prefixes to held-out code-like contexts. No evaluation prefix appears
in the 18-snippet estimation bank.

![Two evaluation domains](../out/figs_core/fig9_bidirectional_corpus.png)

| Direction source | Neutral-prose evaluation | Code-like evaluation |
|---|---:|---:|
| Direct `w` | 0.636 | 0.814 |
| Code-estimated pullback | 1.109 | 1.814 |
| Pooled-prose pullback | **1.614** | **1.965** |
| Saved 25-Pile Jacobian | **1.641** | **2.171** |

Code-estimated directions become more effective on code-like prefixes, but so
do the prose and saved-global directions; pooled prose still beats code by
0.151 mean lift. Therefore the current data show that changing the estimation
corpus changes geometry and causal transfer. They do **not** establish the more
specific claim that matching the estimation and evaluation domains is optimal.
The difference may reflect sample quality, lexical diversity, syntax, length,
or how the constructed target concepts interact with the prompts.

This negative clarification improves the application. It gives a precise next
experiment: sample many matched corpora from both domains, equalize token counts
and concept prevalence, and estimate a two-by-two train-domain × evaluation-domain
interaction with bootstrap confidence intervals. A real domain-matching effect
requires the code-versus-prose advantage itself to change sign across columns.

---

## 10. How the pieces fit together

The project is not a list of unrelated successes and failures. It gives one
coherent picture.

### 10.1 `J` is useful as an operator

The pullback chooses a source perturbation for a specified downstream objective.
It beats direct residual steering across a dose sweep and a 30-direction null,
and its depth dependence behaves as expected when `J` approaches identity.

### 10.2 The effect is not one singular feature

One component fails. A 32- to 64-dimensional combination works best. Readability
of individual components does not predict held-out causal success.

### 10.3 Global averaging can mix regimes

In chess, the output slot dominates the component coefficients. The two regimes
have sharply different early Jacobians, and a matched conditional lens can align
better with the task-specific legality gradient.

### 10.4 Therefore the averaging distribution is part of the method

The average Jacobian answers:

> Across this chosen distribution of prompts and positions, which source
> perturbations tend to propagate into which target perturbations?

It does not automatically answer:

> What are the model's universal semantic variables?

The matched-prose/code experiment makes this operational: two matched prose
estimates agree and transfer almost identically, while a code-derived estimate
rotates and transfers less well without losing all causal utility. The reverse
evaluation does not show that code estimation becomes best on code-like prompts.
Thus the evidence establishes estimator dependence, not an optimal recipe for
domain conditioning.

---

## 11. Why the symmetry work should be an appendix

The symmetry result is mathematically valid. Within an attention head, an
invertible change of coordinates can transform the value and output matrices in
opposite ways while preserving the head's function. Individual head-coordinate
columns can change even though the model's residual outputs and logits do not.

This is useful for auditing claims about:

- individual value dimensions inside a head;
- column-wise head readouts;
- coordinate-sensitive head monitors;
- sign and basis choices in circuit decompositions.

It does not directly threaten:

- residual-stream probes;
- residual-stream steering vectors;
- residual-stream SAEs;
- the residual-to-residual Jacobian itself.

The monitor experiment makes this limitation especially clear: the
head-internal classifier changes under the rotation, while the residual-stream
control remains exactly stable. That is expected because the residual activation
is unchanged by construction.

Use the symmetry work in the application only for one methodological sentence:

> I also tested an exact head-coordinate reparameterization as a calibration
> case; it breaks head-internal interpretations while leaving residual methods
> unchanged, which helped me narrow the final claim to the residual Jacobian's
> averaging and decomposition.

Do not put the symmetry figure in the first three pages unless the application
is explicitly reframed as a head-interpretability audit.

---

## 12. What to do with the other branches

### 12.1 Supervised subspaces

Four attempts failed for different reasons:

- one-dimensional DAS produced almost no state change and no useful gradient;
- eight-dimensional DAS fit training phrasings but not held-out phrasings;
- a de-sinked variant moved one phrasing and failed on others;
- a generalized eigenvalue objective inflated a ratio by driving its denominator
  toward zero.

The lesson is methodological: negative results require a working intervention
scale, held-out templates, and inspection of both numerator and denominator.
Include this as evidence of debugging, not as a central finding.

### 12.2 Edit-site localization

The pooled correlation between Jacobian amplification and edit success appeared
positive. Within a fixed layer it vanished: the reported within-layer
correlation is about -0.064, and adding `J` to a layer-only regression adds
approximately zero R-squared. This is a strong example of a layer confound.

It supports the general theme that aggregate structure can look predictive
because it mixes regimes—in this case, layers—but it is not necessary for the
executive summary.

### 12.3 MoE routing prediction

The Jacobian predicts about 1.3 of the next layer's eight selected experts. A
simple persistence baseline predicts about 7.2, with chance near 1.0. This is a
clean negative and an excellent example of using a strong simple baseline.

It can support the estimator story, but the chess experiment already provides a
more interpretable mechanism with an oracle.

### 12.4 Training dynamics

Across OLMo checkpoints, leading-subspace similarity to the final checkpoint
rises from chance at initialization, grows gradually, dips at a training-stage
boundary, and later recovers. This is an interesting descriptive curve, but the
similarity is guaranteed to reach 1 at the final checkpoint and no causal
explanation for the dip has been established.

Use it as optional future work, not as an application headline.

### 12.5 Emergent misalignment

Several public LoRA adapters shared almost identical input matrices because they
shared initialization. That made naive weight-space convergence look far
stronger than the learned effect. Controlled-seed comparisons found substantial
overlap even between benign fine-tunes.

This is a good confound discovery, but it belongs to a model-diffing application,
not the J-Lens application. Mixing it into the main story makes both weaker.

### 12.6 Reward hacking

The reward-hacking branch contains 300 adapted-model trajectories, paired
teacher-forced base/adapted activations, and grader outcomes. It is valuable as
a proposed application of the central question: how much behaviorally predictive
information survives a constrained representation under mechanism shift?

The current result is not ready to support a monitoring claim. There are 275
hacks but only 25 non-hacks; the system prompt describes the exploitable
mechanisms; some labels are nearly readable from text; and many captured tokens
are not independent examples. An earlier exit-to-always-equal transfer result
also chose its best layer on the test mechanism. The corrected training-only
selection gives AUC 0.692 on 22 examples, with a stratified bootstrap interval
0.450–0.908. Paired base activations give 0.625, interval 0.375–0.875. Both
include chance, and the historical test set is no longer fresh.

Use this as the **next-project motivation**, not a completed success. The outcome
was whether judge-labeled substantive code appeared later, not whether a solution
was correct and not whether the model was honest. See
`docs/FOLLOWUP_QUESTIONS_AND_REWARD_DATA.md` for the full causal-timing audit.

### 12.7 Router centering and basis privilege

The MoE router has a common-mode component that can be removed without changing
routing decisions. Removing it changes conclusions about whether raw router
directions outperform matched bases. This is a useful example of identifying a
gauge-like nuisance component.

Again, this is a side demonstration of scientific care. It is not the residual
Jacobian story.

---

## 13. Candidate application stories ranked

| Story | Fit to the stream | Strength of evidence | Novelty | Main problem | Recommendation |
|---|---:|---:|---:|---|---|
| Open-model J-Lens replication and estimator audit | 5/5 | 4/5 | 3/5 | Efficacy is strong; specificity and corpus mechanism remain open | **Lead with this** |
| Exact symmetries invalidate internal coordinate labels | 3/5 | 5/5 for the narrow theorem | 3/5 | Mostly head-internal; residual methods unchanged | Appendix only |
| MoE router basis privilege | 4/5 | 3/5 | 4/5 | Several corrections; hard to explain quickly | Follow-up project |
| Fine-tune convergence and initialization confounds | 4/5 | 3/5 | 3/5 | Separate research question | Separate application |
| J-space information retention on reward-hacking trajectories | 5/5 | 2/5 currently | 4/5 as a proposal | Small shifted test, imbalanced labels, no 9B Jacobian yet | **Lead future-work proposal, not completed results** |
| Jacobian training dynamics | 3/5 | 3/5 | 3/5 | Descriptive, no decisive mechanism | Supporting figure at most |

---

## 14. The proposed application structure

The application instructions ask for a one- to three-page executive summary,
no more than 600 words, with graphs. The form summary is read first and acts as
a preliminary filter. Specific models, sample sizes, controls, surprising
numbers, and limitations matter more than broad motivation.

### 14.1 Suggested title

Use a plain title such as:

**When does an averaged Jacobian mean anything?**

or:

**Stress-testing J-Lens on open models: causal transport, spectral structure,
and corpus dependence**

Avoid “J-Lens finds subspaces, not features” as the title. It is catchy but too
broad, and it risks confusing singular components with the paper's token-indexed
sparse frame.

### 14.2 Executive summary outline

Write this yourself in approximately 450 to 550 words.

**Paragraph 1: problem and relationship to prior work.** Explain that J-Lens
averages local residual Jacobians over prompts and positions, producing
token-indexed directions used for readout and intervention. Explicitly say that
the paper already demonstrates steering and coordinate swaps. Your question is
not whether steering exists, but how robustly the averaged operator transports
pre-specified concepts on open models and how sensitive that result is to
spectrum and estimation corpus.

**Paragraph 2 plus Figure 1: robust replicated result.** Name all three models
(135M, 4B, and 7B), seven concepts, four layers per model, construction/test word
split, direct residual baseline, four doses, and 30 random directions. At dose
0.15, mean pullback versus direct lifts are 1.641 versus 0.636, 1.437 versus
0.730, and 1.261 versus 0.669. All 84 pullback cells exceed all 30 sampled random
effects in their cell. Describe this as a finite sampled null, not 84 independent
family-wise-corrected discoveries.

**Paragraph 3 plus Figure 2: estimator corpus.** Explain the two
disjoint 18-prompt genre-matched prose corpora and the 18-prompt code corpus.
State that prose A/B pullbacks have cosine 0.953 and virtually identical causal
effects, while prose/code cosine is about 0.688 and the code-derived direction
transfers less well to prose (1.109 versus 1.614), though it still beats direct
`w`. Add that code-derived directions also fail to become best on the new
code-like evaluation (1.814 versus 1.965 for pooled prose). Therefore the corpus
matters, but “match the domains” is not yet established.

**Paragraph 4 plus Figure 3 or the concrete prompt panel: distributed effect.** State that one singular
component gives -0.13 lift, while 32 and 64 components give 1.83 and 1.89,
slightly above the full spectrum's 1.75. Explain that the weak tail harms the
intervention and the pseudoinverse fails. If space is tight, replace this figure
with the actual “On the table there was a” probability panel and keep the
truncation numbers in prose.

**Paragraph 5: conclusion, limitations, and proposed extension.** Describe `J`
as a useful averaged transport map, not an automatic inventory of universal
singular features. Name the
remaining limitations: small prompt banks, all-position interventions,
correlated concept-level tests, and no semantic-specificity result. Corpus
conditioning and truncation still require larger-model replication. End with a
specific proposed test: compare full residual, PCA, random projection, token
J-Lens/sparse J-space, and a text baseline for predicting later reward-hacking
behavior across exploit mechanisms, using a new held-out set and a 9B-matched
Jacobian. Chess can be omitted entirely from the executive summary.

### 14.3 Figures for the first three pages

Use these, in this order:

1. `out/figs_core/fig6_large_model_replication.png`
2. `out/figs_core/fig5_corpus_conditioning.png`
3. `out/figs_core/fig7_prompt_and_corpus.png`

If the document has room for a fourth main figure, use
`out/figs_core/fig2_truncated_pullback.png`. The bidirectional correction belongs
next to the corpus figure or in an appendix:

- `out/figs_core/fig9_bidirectional_corpus.png`

The detailed method comparison and chess mechanism remain useful supplementary
figures:

- `out/figs_core/fig1_pullback_operator.png`
- `out/figs_core/fig3_averaging_and_conditioning.png`

Figures 1–6 are regenerated from saved JSON by:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.application_core_figs
```

The figure generator does not run model inference or create new scientific
results. It aggregates existing saved outputs.

Figure 7 is reproduced by `.venv/bin/python -m
jlens.application_prompt_example`, which reruns SmolLM inference. Figure 9 is
reproduced from the two saved corpus evaluations by `.venv/bin/python -m
jlens.corpus_bidirectional_fig`.

### 14.4 What belongs after the executive summary

1. Exact definition of `J` and the averaging distribution.
2. A diagram distinguishing token J-Lens vectors, SVD components, and concept
   pullbacks.
3. Pullback experiment protocol, dose sweep, null, and full result table.
4. Cross-corpus pullback geometry and transfer.
5. Truncation experiment and interpretation.
6. Actual prompt probability table and explicit distinction from generation.
7. Reward-hacking proposal, with the current exploratory null/uncertain result.
8. Optional individual-direction, MoE, and chess diagnostics.
9. Failed approaches and how they changed the protocol.
10. Limitations, missing artifacts, and proposed next experiments.
11. Time spent, collaborators, and precise use of coding or language-model
    assistance.

---

## 15. Claims ledger

### Claims safe to make now

- Across SmolLM2-135M, Qwen3.5-4B, and OLMo-3-7B, pullback directions were
  positive in all 84 concept-layer cells at every tested dose. At dose 0.15,
  every cell exceeded all 30 sampled random-direction effects for that cell.
- On SmolLM2-135M, `J.T @ w` outperformed direct injection of `w` on the
  repository's held-out concept-word metric: 1.641 versus 0.636 mean lift across
  28 concept-layer cells at the reference dose.
- Pullback effects were positive in all 28 cells at all four tested doses; at
  dose 0.15, every pullback exceeded all 30 sampled random-direction effects in
  its cell. This is a descriptive finite-null result, not a family-wise test.
- The pullback advantage shrank near the target layer, consistent with `J`
  approaching identity.
- Pullbacks estimated from two disjoint, genre-matched prose banks had mean
  cosine 0.953 and mean held-out lifts 1.604 and 1.605. Code-derived pullbacks
  had mean cosine about 0.688 to prose and lower prose transfer at 1.109.
- On the code-like evaluation bank, code-estimated, pooled-prose, and saved-global
  pullbacks scored 1.814, 1.965, and 2.171 respectively. Domain matching did not
  make the code estimate best.
- In the illustrative SmolLM prompt “On the table there was a,” the top next
  token changed from “large” to “plate” under the prose pullback. This is one
  selected next-token example, not a generation-level success rate.
- The pseudoinverse produced approximately zero mean held-out lift.
- A truncation to 32 to 64 singular components slightly outperformed the full
  576-component pullback at matched intervention norm.
- Readability bands of leading singular components had similar held-out steering
  pass rates despite a large difference in readability score.
- In the chess model, component coefficients correlate more strongly with
  origin-versus-destination output slot than with the selected chess properties.
- Origin and destination Jacobians differ strongly in early chess layers.
- At layer 5 destination positions, a parity-matched conditional Jacobian has
  2.4 times the rank-4 overlap with the legality-gradient subspace of the pooled
  Jacobian.
- This conditioning improvement is partial and not consistent for every regime
  and rank.

### Claims that require qualification

- “J finds downstream-relevant information earlier than PCA” is supported for
  topic classification at layer 4 in one 135M model, but PCA wins at later
  layers.
- “MoE Jacobians are less reproducible” is supported in a very small split-half
  study; six prompts per half are insufficient to separate model differences
  from estimator noise.
- “Individual directions reliably succeed or fail” depends on a reported
  split-half analysis whose exact saved producer should be made easier to trace.
- “The tail is counterproductive” is accurate for the normalized finite
  intervention used here, not a universal spectral theorem.
- “The operator is distribution-specific” is directly demonstrated for three
  manually constructed, AI-assisted 18-prompt banks on one model; the size and generality of that
  effect require randomized, larger-corpus replication.
- “Residuals predict whether substantive code follows an exploit across hack
  mechanisms” is only exploratory: training-selected AUC is 0.692 on 22 shifted
  examples and its bootstrap interval includes chance. The label does not mean
  correctness or honesty.

### Claims not safe to make

- J-Lens is generally invalid.
- The original paper's token-indexed J-Lens vectors are not features.
- All interpretability directions are arbitrary.
- Residual-stream monitoring fails under the attention-head symmetry.
- Conditional Jacobians solve the averaging problem.
- Pullback steering is semantically specific.
- Matching the Jacobian-estimation domain to the evaluation domain improves
  steering.
- The current reward-hacking probe detects deception, intent, or a correct
  solution.
- The results establish behavior on frontier-scale or reasoning models.
- The entire repository was completed within the 20-hour application limit,
  unless that is literally true and documented.

---

## 16. The strongest next experiments

If additional work is allowed under the application rules and actual time
budget, these experiments would strengthen the story in priority order.

### 16.1 Behaviorally predictive information under mechanism shift

Use the reward-hacking corpus for a strict comparison of representations, not
for an “honesty direction” demo. Choose a behavior that is objectively graded
and has not occurred at the measurement position. Split by problem identity and
hack mechanism before any model selection. On the same examples and dimensions,
compare:

- the exact-prefix text baseline;
- the full residual stream;
- PCA fitted only on training residuals;
- matched random projections;
- token J-Lens scores and the paper's sparse nonnegative J-space decomposition;
- optionally a supervised low-dimensional subspace as an upper benchmark.

The existing 4B and 7B Jacobians cannot be used for Qwen3.5-9B. Compute targeted
vector–Jacobian products or a native 9B lens using exactly the activation and
target conventions used for the reward trajectories. Report prediction before
the exploit separately from prediction after its first tokens. Obtain a new
held-out test set with more non-hacks and hardened-grader outcomes. Only after
predictive validation should an intervention be tested, with both hardened task
success and exploit rate as outcomes.

### 16.2 Scale up cross-corpus pullback transfer

The initial 18-prompt prose/prose/code comparison is now complete. Replace the
manually constructed banks with disjoint, substantially larger samples from named
datasets. Repeat many matched splits so uncertainty is over corpus sampling,
not just seven chosen concepts. Build pullbacks with `J_A` and evaluate on
prompts from distribution B, and vice versa.
Compare:

- `J_A.T @ w`;
- `J_B.T @ w`;
- pooled `J.T @ w`;
- direct `w`;
- prompt-specific gradient, as an expensive upper bound;
- at least 30 random directions per cell.

Pre-register a hierarchical analysis over prompts, concepts, and corpus splits.
The current 30-direction null controls random intervention orientation, but does
not quantify uncertainty from selecting the corpora themselves.

This directly asks whether the operator describes the model or merely the
estimation corpus.

### 16.3 Conditional J-Lens on natural language

Chess gives an artificial but clean regime label. Repeat the idea on natural
language using labels that are known before inspecting the lens:

- source-to-target token distance;
- question versus statement;
- code versus prose;
- assistant versus user turn;
- syntactic role or a small set of task templates.

Use training data only to define the conditional Jacobians. Evaluate them on
held-out prompts using causal objectives, not token-coherence judgments. Compare
pooled, conditional, and mixture-of-experts combinations of local Jacobians.

### 16.4 Direct test of the paper's token-indexed sparse frame

Separate this from SVD-component analysis. Choose multiword concepts with
disjoint train and test synonym sets. Compare:

- individual token J-Lens vectors;
- sparse nonnegative combinations as used in the paper;
- concept pullbacks from held-out words;
- logit lens and tuned lens baselines;
- activation-difference and supervised-probe baselines.

Measure verbal report, task behavior, specificity, and unrelated-text damage.
This would address the paper's actual feature claim rather than a nearby SVD
interpretation.

### 16.5 Replicate the remaining claims on larger open models

The direct-w versus pullback comparison is now complete on Qwen3.5-4B and
OLMo-3-7B, with four doses and 30 random directions per layer. The remaining
larger-model experiments are spectral truncation, corpus conditioning, and
semantic specificity. Existing efficacy results do not establish these claims.

Pre-register:

- layers as fractions of depth;
- intervention normalization;
- concept lists and train/test split;
- dose grid;
- primary specificity metric;
- random-draw count;
- exclusion rules for multi-token words.

### 16.6 Restore missing provenance before adding new breadth

The repository contains an attractive `out/chess/concept_r2.json` summary but no
checked-in script that creates it. Restore the producer or remove the result from
the final report. Do the same audit for any number copied from prose rather than
generated by an inspectable artifact.

This cleanup is more valuable than another exploratory branch.

---

## 17. Reproduction and audit notes

### 17.1 Commands that are cheap and currently useful

Rebuild the application figures:

```bash
MPLCONFIGDIR=/private/tmp/mpl-jlens \
PYTHONPATH=. .venv/bin/python -m jlens.application_core_figs
```

Recompute the topic-subspace result:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.subspace_meaning \
  --layers 4 12 20 28 \
  --out out/rare/subspace_meaning_full.json
```

Recompute the pullback dose sweep and 30-direction null:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.pullback_robustness
```

Recompute the matched-prose/code corpus experiment:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.pullback_corpus
```

Recompute its held-out code-like evaluation and the two-domain figure:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.pullback_corpus \
  --evaluation-domain code \
  --out out/rare/pullback_corpus_code_eval.json
MPLCONFIGDIR=/private/tmp/mpl-jlens \
PYTHONPATH=. .venv/bin/python -m jlens.corpus_bidirectional_fig
```

Recompute the concrete prompt and reward-transfer audit:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.application_prompt_example
MPLCONFIGDIR=/private/tmp/mpl-jlens \
PYTHONPATH=. .venv/bin/python -m rh.transfer_audit
```

The reward audit reads only saved key-position arrays. It does not run the 9B
model, and it does not establish a J-Lens result because no 9B Jacobian is used.

Run the core math tests when the test dependencies are installed:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_lens_math.py tests/test_identity_at_target.py
```

### 17.2 Reproducibility problems to fix

- Many central scripts and documents are currently untracked by git.
- The root setup instructions refer to a missing or incomplete dependency path.
- `verify.py` presents itself as recomputing every claim, but at least one chess
  check reads a saved JSON rather than rerunning the underlying model experiment.
- `jlens/selection_check.py` hardcodes the MPS device and failed in one audit
  environment; it should accept `--device` and save structured output.
- Several prose reports preserve numbers whose exact producing script or current
  output is unclear.
- The newer robustness and corpus-transfer artifacts are reproducible, but the
  legacy method-comparison artifact still contains only one random direction
  per cell; use the newer null for random-baseline claims.

These are not reasons to hide the project. They are reasons to narrow the claim,
publish the exact artifacts used in the summary, and make the final repository
easy for a reviewer or agent to interrogate.

### 17.3 Time-accounting warning

The application brief defines nearly all active project work—coding, analysis,
project-specific paper reading, planning, and the full write-up—as part of the
20-hour limit, with two additional hours for the executive summary. It also
allows an alternate prior-research route, judged more harshly, for earlier work.

This repository is far broader than a plausible 20-hour project. Use one of two
honest approaches:

1. If this work was genuinely done within the application window, give the real
   tracked hours and explain the scope.
2. Otherwise, use the prior-research route, give the real total effort, and
   explain your personal contribution.

Do not compress the history into a fictional 20-hour narrative.

---

## 18. Questions a reviewer may ask

### Why is `J.T @ w` not a tautological success?

It is the optimal first-order direction for its constructed objective, so some
direct effect is expected. The nontrivial evidence is that it beats injecting
`w` directly, moves held-out concept words, survives four finite doses and a
30-direction null, transfers across independently estimated prose Jacobians,
shows a meaningful depth pattern, fails under pseudoinversion, and improves when
the weak spectral tail is removed.

### Why not call the singular directions J-Lens features?

Because the paper's J-Lens vectors are token-indexed rows of `W_U J`. Singular
vectors are a matrix decomposition. They can be useful without being the model's
atomic semantic variables.

### Does the truncation result prove rank 64 is special?

No. It gives an optimum for two layers, seven concepts, one model, one dose, and
one normalization rule. The useful claim is that the effect is distributed and
the weakest tail can hurt—not that every model has a 64-dimensional workspace.

### Could the chess output-slot signal be a legitimate feature?

Yes. The model must know whether it is choosing an origin or destination. The
point is not that the signal is fake. The point is that a dominant component of
the pooled operator reflects which computation is active, so interpreting that
component as chess content would be misleading.

### Why does conditioning help destinations more than origins?

The data do not yet identify the mechanism. Destination choice is conditional on
an already selected origin and may define a more distinct local transformation.
That is a hypothesis for follow-up, not a conclusion.

### Why is PCA sometimes better?

PCA finds high-variance activation directions. In later layers, topic
information is strongly concentrated in those directions. `J` has an advantage
only when downstream relevance differs from activation magnitude, as at layer 4
in the current topic experiment.

### Does this refute the global-workspace interpretation?

No. It identifies a failure mode of a global average and a distinction between
operator decompositions and token-indexed sparse frames. A much larger,
paper-faithful replication would be required to evaluate the global-workspace
claim.

### What would change your mind?

If large-corpus Jacobians were highly stable across genuinely different
distributions—not just the matched prose halves tested here—if individual
singular components' readability strongly predicted held-out causal specificity,
and if pooled lenses matched conditional lenses on known multi-regime tasks, the
mixture concern would be much weaker. Conversely, cross-corpus failures on larger
models would strengthen it.

---

## 19. A practical study schedule

### Session 1: definitions, 60 minutes

- Read sections 2 and 3.
- Read the original paper's J-Lens method and intervention sections. Explain the
  difference between additive token steering and the ant/spider coordinate swap.
- Derive `w.T @ J @ x = (J.T @ w).T @ x` on paper.
- Write one sentence each defining `J`, a token J-Lens vector, an SVD component,
  and a concept pullback.

### Session 2: implementation, 90 minutes

- Read `jlens/lens.py`.
- Trace tensor shapes through one Jacobian computation.
- Check which positions and prompts are averaged.
- Read the math tests and explain what each test does and does not validate.

### Session 3: positive result, 90 minutes

- Read `jlens/pullback_robustness.py` and `jlens/pullback_rank.py`.
- Recompute the headline means in Figures 4 and 2 from the JSON files.
- Identify the train/test leakage barrier and the no-J baseline.
- List the unaddressed confounds.

### Session 4: distribution conditioning, 90 minutes

- Read `jlens/pullback_corpus.py` and inspect all three prompt banks.
- Explain why matched prose A/B agreement separates estimator stability from
  the code/prose domain shift more clearly than the old six-prompt split.
- Recompute the cosine and causal-transfer means in Figure 5.
- Read the reverse-domain result in section 9.4 and explain why it does not show
  a domain-matching interaction.
- State what the manually constructed corpora do and do not prove, including
  their AI-assisted provenance.

### Optional supplementary session: chess mechanism, 90 minutes

- Read `jlens/chess_cond.py` and `jlens/chess_q1q2.py`.
- Explain why parity corresponds to two different output tasks.
- Explain principal-angle overlap and the random-subspace baseline.
- Compare destination and origin results rather than quoting only the successful
  condition.

### Session 5: reward-hacking proposal, 90 minutes

- Read section 12.6 and `docs/FOLLOWUP_QUESTIONS_AND_REWARD_DATA.md`.
- Explain why 48 GB of activations still represents only 300 independent
  trajectories, and why 275 hacks versus 25 non-hacks is problematic.
- Define the current “substantive attempt follows” label without using the words
  honesty, deception, correctness, or hack detection.
- Design the new split before choosing layers or representation dimension.

### Session 6: negative and side branches, 60 minutes

- Read `jlens/steer_interp.py`, `jlens/admissible.py`, and
  `docs/POSITION_IS_THE_VARIABLE.md`.
- Skim the MoE, localization, fine-tuning, reward-hacking, and symmetry branches.
- For each, write “claim, strongest control, biggest remaining problem.”

### Session 7: write from memory, 120 minutes

- Close this guide.
- Draft a 150-word form answer and a 500-word executive summary in your own
  voice.
- Add the three recommended main figures; keep the chess and detailed-method
  figures for supplementary pages if space permits.
- Reopen the guide only to verify numbers.
- Delete any claim you cannot explain under questioning.

---

## 20. Final checklist

Before submitting, verify all of the following.

- The first sentence states a concrete question about the averaged residual
  Jacobian.
- “J-Lens vector” and “singular vector of `J`” are never used interchangeably.
- The no-J residual baseline appears next to the pullback result.
- The 28-cell sample size and held-out word split are stated.
- The 30-direction null and four-dose sweep are stated without implying
  family-wise-corrected significance.
- The matched-prose/code result is described as a small, manually constructed,
  AI-assisted corpus comparison, not a universal code-versus-prose law.
- The reverse-domain result is included: the experiment does not yet support an
  “estimate J on the matching domain” prescription.
- The summary distinguishes three-model efficacy from the SmolLM-only
  truncation and corpus-conditioning results.
- The truncation comparison says vectors were normalized to a matched dose.
- If chess is included, its figure uses the reproducible correlation artifact,
  and destination-conditioning is paired with the inconsistent origin result.
- The symmetry experiment is described as head-internal and does not imply that
  residual monitoring changes.
- The reward-hacking proposal does not relabel “substantive attempt later” as
  correctness, honesty, deception, or pre-exploit hack detection.
- The original paper is credited for J-Lens steering and ant/spider coordinate
  swapping; our work is called a replication and evaluation.
- The form and executive summary are written in your own voice.
- The real project hours and your personal contribution are stated accurately.
- Anyone with the link can view the final Google Doc.

If all of these are true, the application will present the repository as what it
actually is: a serious open-model replication and audit of a promising
interpretability method, with a useful positive result, strong controls,
informative negative results, and visible scientific self-correction.

---

## 21. Primary references

- [Neel Nanda MATS 12.0 admissions procedure and research problems](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.sa4u6llpnkph)
- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)

These references delimit the novelty claim. The J-Lens paper is the primary
method and steering reference. Contrastive Activation Addition and Representation
Engineering establish that residual activation steering predates this project.
The Tuned Lens is the relevant learned-decoder comparison. Cite only claims you
actually use; a long bibliography is not a substitute for a precise comparison.
