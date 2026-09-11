# From experiments to an application: a plain-English guide

This is the document to read first if the longer research guide feels too dense.
It assumes no background in transformers, linear algebra, Jacobians, or
interpretability. It is organized in the order:

> **experiment → inference → follow-up question → larger-model replication**

The purpose is to give you a mental model strong enough to write the application
in your own words and answer questions about it. It is not text to paste into the
application. The application asks you to disclose AI assistance and treats raw
AI-written summaries as a negative signal.

For concrete outputs rather than averages, see the
[prompt-example gallery](PROMPT_EXAMPLE_GALLERY.md), which shows all 12 held-out
prompts for each of SmolLM2-135M, Qwen3.5-4B, and OLMo-3-7B.
The new [Rome-minus-Paris contrast experiment](CONTRASTIVE_LOGIT_EXPERIMENT.md)
starts from an exact two-token logit objective, follows its associated words,
and explains what the SVD decomposition contributes.

## The whole project in one paragraph

The original J-Lens paper introduced a way to translate directions inside an
early transformer layer into directions that matter later. I first reproduced
their published Qwen3.5-4B lens, then experimented on the much smaller
SmolLM2-135M because it was cheap enough to debug locally. I found that the
Jacobian reliably improves a simple concept-steering direction, but that this
success uses many spectral components and does not make every individual
component a clean semantic feature. I then tested the important positive result
on Qwen3.5-4B and OLMo-3-7B. The steering benefit replicates. A more specific
SmolLM result—that the leading Jacobian subspace concentrates topics better than
ordinary activation PCA—does not replicate on the larger models. Changing the
texts used to estimate the Jacobian also changes the resulting direction, but a
new reverse-domain test does not show that merely matching domains solves the
problem. The application is therefore a replication and stress test of J-Lens,
not a claim that I invented Jacobian steering.

## Part I: the minimum mental model

### 1. What is happening inside a language model?

The model converts every token into a long list of numbers. As the token passes
through the model's layers, that list is repeatedly updated. This running list
of numbers is called the **residual stream**.

Call the residual vector at layer `l`:

```text
h_l
```

The letter `h` means hidden state. The subscript `l` tells us which layer.

Near the end, the model converts its final residual vector into one score for
every possible next token. These scores are called logits. A matrix called the
**unembedding**, written `W_U`, performs most of that conversion.

Every row of `W_U` corresponds to one token. For example, the row for “bread” is
a direction that makes the final output more favorable to “bread.” Call that row
`u_bread`.

### 2. Why can we not simply add “bread” at any layer?

The representation can change as it moves through the model. A direction that
means “bread” near the output is not guaranteed to have the same meaning at
layer 4. This is analogous to two coordinate systems: moving east on one map is
not necessarily the same vector after the map has been rotated and stretched.

The simplest baseline ignores this problem and directly adds the final-layer
concept direction at an earlier layer. In this document that baseline is called
**direct `w`**.

### 3. What is the Jacobian?

Choose an earlier layer `l` and a later target layer `T`. A small change at the
earlier layer is written `delta`. The Jacobian `J_l` predicts how that change
will arrive at the target:

```text
change at target ≈ J_l × delta
```

`J_l` is a matrix of derivatives. It is a local, first-order description, like a
slope. Because a transformer is nonlinear, the exact Jacobian depends on the
prompt, token positions, attention patterns, and—in a mixture-of-experts
model—which experts are active.

J-Lens averages these context-specific Jacobians over many prompts and positions.
The result says:

> Across the chosen estimation texts and positions, how does a small direction
> at layer `l` tend to travel to the target layer?

### 4. What are `w` and `J.T @ w`?

For a multiword concept such as food, we begin with construction words such as
“vegetables,” “cooking,” “meal,” “rice,” “kitchen,” and “pasta.” Their centered,
normalized unembedding rows are averaged and normalized again. The result is:

```text
w = the chosen concept direction at the target layer
```

`w` is not the activation from a prompt. It is not a trained classifier. It is a
fixed direction constructed from output words.

We want an earlier-layer edit that increases the later score along `w`. The
correct first-order direction is:

```text
v = J_l.T @ w
```

`J_l.T` means the transpose of the Jacobian. It takes a desired target direction
back to a source-layer direction. This operation is called a **pullback**.

The equation comes from the predicted score change:

```text
target score change ≈ w.T @ J_l @ delta
                         = (J_l.T @ w).T @ delta
```

For a fixed-size edit, choosing `delta` parallel to `J_l.T @ w` maximizes this
linearized score. This makes a positive effect unsurprising. The scientifically
interesting questions are whether it beats strong controls, survives a finite
nonlinear intervention, transfers to words not used to construct `w`, and
replicates on other models.

### 5. What exactly do we add to the model?

Let `d` be either the pullback direction, direct `w`, or a random direction. We
normalize it, so every method receives an edit of the same length:

```text
edit = alpha × layer_scale × d / ||d||
h_l' = h_l + edit
```

`||d||` is the length of `d`. `layer_scale` is based on typical residual-vector
length at that layer. `alpha` is the dose. The reference dose is 0.15.

This does **not** mean “15% more food.” It means that the edit's length is 15% of
the chosen typical residual-norm scale. In the current experiment the same edit
is added at every existing prompt-token position at the output of one selected
transformer block.

### 6. How does the original paper steer?

For one vocabulary token `t`, the original paper's J-Lens vector is:

```text
v_t = J_l.T @ u_t
```

where `u_t` is that token's unembedding direction. The paper adds a scaled copy
of `v_t` to an activation. Negative addition or projection can suppress it.

The spider-to-ant example uses a more controlled two-direction edit. The paper
places the spider and ant vectors into a two-column matrix `V`, measures the
activation's coordinates in that two-dimensional span, swaps the coordinates,
and writes the changed component back. On a question about the number of legs of
the animal that spins webs, the answer changes from 8 to 6.

That experiment already exists in the paper. Our `J.T @ w` experiment replaces
the paper's one-token target with an average over several concept-word
directions, then scores different held-out words and matched controls. That is a
useful evaluation extension, not a new kind of steering or evidence that the
paper was only doing superficial token steering: the ant/spider swap is already
an effective concept-level intervention. We must not claim to have invented the
steering mechanism or to have uniquely introduced concept steering.

---

## Experiment 0: did we implement J-Lens correctly?

### Experiment

Before interpreting anything, we recomputed selected J-Lens vectors for
Qwen3.5-4B under the configuration stored with a published external lens:

- the same Qwen3.5-4B model;
- the same 25 Pile documents;
- the same source and target layers;
- the same position skipping;
- the same averaging convention.

We compared our vectors with vectors calculated from the published matrix.
Cosine similarity measures whether two vectors point in the same direction:
1 means identical direction, 0 means unrelated, and -1 means opposite.

Result: mean cosine **0.9984**, with magnitude ratio **0.9985**. Neighboring
layer conventions match less well: 0.9592 and 0.9709.

### Inference

Our automatic-differentiation shortcut, layer indexing, and weighting convention
reproduce the published artifact extremely closely. This is the strongest sense
in which we have independently reproduced part of the original paper.

### Follow-up question

Matching the lens construction does not reproduce the paper's entire global
workspace theory. Does the reproduced operator actually improve a causal edit
under controls?

### Replication in larger models

The numerical reproduction itself is already on Qwen3.5-4B. OLMo-3-7B uses a
full locally calculated Jacobian rather than the same external reference, so it
is an independent implementation/model check but not a comparison to a
published OLMo lens.

---

## Experiment 1: does the Jacobian improve concept steering?

### Experiment on SmolLM2-135M

Seven concepts were chosen: food, medicine, programming, law, sports, finance,
and female-associated words. Each concept's single-token words were split in
half:

- the construction half made `w`;
- the test half measured the effect.

For each held-out word, a control token with similar baseline probability but
low concept similarity was chosen. The score was:

```text
average log-probability change of held-out concept words
minus
average log-probability change of matched control words
```

The main comparison was pullback `J.T @ w` versus direct `w`. At dose 0.15,
averaged over seven concepts and four layers:

| Method | Mean held-out lift |
|---|---:|
| Pullback `J.T @ w` | **1.641** |
| Direct `w` | 0.636 |

Thirty random directions were evaluated for every concept-layer cell. Every one
of the 28 pullback cells exceeded all 30 sampled random effects.

![Three-model steering replication](../out/figs_core/fig6_large_model_replication.png)

### Inference

The averaged Jacobian is doing useful work. It is not merely decorating the
same output direction: transporting the target direction backward produces a
stronger early-layer intervention than injecting that target direction directly.

This does not yet establish semantic specificity. Making food words more likely
does not prove that only food changed, that a generated paragraph remains
coherent, or that a downstream task improved.

### Follow-up questions

1. Is the result specific to a tiny model?
2. Does it survive different intervention doses?
3. Does the advantage disappear near the target, where less transport is needed?

### Replication in larger models

The identical protocol was run on Qwen3.5-4B and OLMo-3-7B:

| Model | Pullback | Direct `w` | Pullback/direct |
|---|---:|---:|---:|
| SmolLM2-135M | **1.641** | 0.636 | 2.58× |
| Qwen3.5-4B | **1.437** | 0.730 | 1.97× |
| OLMo-3-7B | **1.261** | 0.669 | 1.88× |

Across all three models, all 84 pullback cells were positive at all four doses.
At dose 0.15, all 84 exceeded every one of their 30 sampled random effects.
The advantage also shrinks near the target layer in all three models. This is
what we expect: close to the target, the transport map increasingly resembles
identity, so direct `w` and transported `w` become similar.

Application status: **the strongest completed, cross-model result.**

---

## Experiment 2: what does one actual prompt look like?

### Experiment on SmolLM2-135M

Prompt:

> On the table there was a

Using the food target at layer 20, the clean model's top next token is “large”
with probability 4.25%. With the prose-estimated pullback it becomes “plate” at
6.94%.

Several food words were held out when constructing `w`:

| Token | Clean | Direct `w` | Prose pullback | Code pullback |
|---|---:|---:|---:|---:|
| bread | 0.0118% | 0.0459% | 0.1115% | 0.0762% |
| cheese | 0.0093% | 0.0372% | 0.1374% | 0.0865% |
| sauce | 0.0143% | 0.0672% | 0.3682% | 0.1643% |
| salad | 0.0108% | 0.1002% | 1.4145% | 0.3649% |

![Actual prompt example](../out/figs_core/fig7_prompt_and_corpus.png)

### Inference

The aggregate metric corresponds to a visible change in a real next-token
distribution. The example also shows why the predeclared multiword metric is
better than narrating one cherry-picked token: the code pullback gives “plate” a
higher probability than the prose pullback, while performing worse on the full
held-out food score.

### Follow-up question

Does the edit produce coherent multi-token behavior, and is it specific to the
desired concept rather than generally sharpening related nouns?

### Replication in larger models

The same prompt, food-word construction rule, dose, and comparable middle layer
were run on all three models using each model's saved global Jacobian:

| Model | Clean top token | Pullback top token | Mean held-out food probability: clean → pullback |
|---|---|---|---:|
| SmolLM2-135M | large | plate | 0.0076% → 0.2041% |
| Qwen3.5-4B | glass | bowl | 0.0129% → 0.0359% |
| OLMo-3-7B | pile | pile | 0.0076% → 0.0194% |

![Actual prompt across all models](../out/figs_core/fig12_cross_model_prompt_example.png)

The mean held-out food probability rises in every model. The top token changes
to a food/container word in SmolLM and Qwen, but not in OLMo. This is a matched
illustration, not independent evidence beyond the aggregate evaluation and not
a generated-text result.

The [full prompt gallery](PROMPT_EXAMPLE_GALLERY.md) contains the other 11
prompts for every model, including clean, direct-`w`, and `Jᵀw` top-token
distributions. Read the absolute probabilities as well as the multipliers: a
large ratio can begin from a tiny baseline.

---

## Experiment 3: does steering require one direction or many components?

### Experiment on SmolLM2-135M

Every matrix can be decomposed into singular components. For the Jacobian:

```text
J = U × Sigma × V.T
```

`V` contains source-layer directions, `U` contains target-layer directions, and
`Sigma` says how strongly each source direction is transported. The complete
pullback can be written as a weighted sum of these components.

We kept only the first `k` components, normalized the resulting vector to the
same intervention dose, and varied `k`:

| Components | Mean lift |
|---:|---:|
| 1 | -0.135 |
| 8 | 1.105 |
| 16 | 1.547 |
| 32 | 1.835 |
| 64 | **1.894** |
| Full 576 | 1.746 |

![SmolLM truncation result](../out/figs_core/fig2_truncated_pullback.png)

### Inference

The successful intervention is not one beautiful singular feature. It is a
coordinated combination of many components. On SmolLM, 32–64 components are
enough and the weakest tail slightly harms the finite intervention.

This also explains why the full-Jacobian pseudoinverse fails. A pseudoinverse
divides by small singular values, amplifying the least stable part of the map.
The transpose instead downweights that weak tail.

### Follow-up question

Is “64 useful components” a model-independent fact, or merely a feature of a
576-wide small model?

### Replication in larger models

We run the same held-out steering test at two middle layers per model. Because
exactly decomposing a 4096×4096 matrix is expensive, finite-`k` large-model
curves use a seeded randomized low-rank SVD; the endpoint `J.T @ w` remains exact.
This approximation must be disclosed.

The larger-model results are different from SmolLM. Qwen lift rises steadily from
0.162 at one component to 1.529 at 256 components and 1.641 for the exact full
pullback. OLMo lift rises from 0.199 at one component to 1.133 at 256 components
and 1.415 for the exact full pullback. Thus the claim that a 32–64-dimensional
truncation is best does **not** replicate on either larger model.

![Cross-model spectral truncation](../out/figs_core/fig10_cross_model_rank.png)

Application status: **use the cross-model curve, not “rank 64 is special.”** The
scientific inference is that the effect is distributed and its required rank can
grow with model width.

---

## Experiment 4: do leading Jacobian subspaces concentrate topic information?

### Experiment on SmolLM2-135M

We collected residual activations from 72 prompts in six topics. We projected
each activation into only eight coordinates and used leave-one-out nearest
centroid classification. Four coordinate systems were compared:

- the leading eight source-side singular directions of `J`;
- the leading eight principal components of the activations, called PCA;
- a random eight-dimensional subspace;
- shuffled labels.

At layer 4:

| Representation | Topic accuracy |
|---|---:|
| Top eight of `J` | **76%** |
| Activation PCA | 47% |
| Random subspace | 34% |
| Shuffled labels | 15% |

### Inference

On the small model's early layer, downstream sensitivity concentrates topic
information more strongly than raw activation variance. But this is decoding,
not causality, and it does not say the eight axes are individually meaningful.
The eight directions form one eight-dimensional subspace; an activation is still
needed to produce its eight coordinates.

### Follow-up question

Does this apparent advantage survive width and architecture changes?

### Replication in larger models

It does not. At the earliest tested layer:

| Model | Layer | Top-eight `J` | Activation PCA | Random eight-dimensional subspace |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 4 | **76.4%** | 47.2% | 34.0% |
| Qwen3.5-4B | 4 | 38.9% | **98.6%** | 48.9% |
| OLMo-3-7B | 4 | 62.5% | **97.2%** | 37.0% |

The larger-model Jacobian subspaces still contain topic information—especially
OLMo's—but ordinary activation PCA is dramatically stronger. Qwen's top eight
are even below the average random-eight baseline at layer 4.

This is a valuable non-replication. It prevents us from claiming that J generally
discovers a uniquely compact topic subspace. The positive steering result scales;
this specific decoding advantage does not.

![Cross-model topic subspace](../out/figs_core/fig11_cross_model_topic_subspace.png)

Application status: **a concise negative result demonstrating scientific
self-correction.**

---

## Experiment 5: does the estimation corpus change the pullback?

### Experiment on SmolLM2-135M

The desired concept `w`, model, layer, dose, and evaluation prompts were held
fixed. Only the texts used to estimate the average Jacobian changed:

- 18 prose sentences, bank A;
- 18 different but broadly matched prose sentences, bank B;
- 18 code snippets.

For each corpus `D` we calculated:

```text
J_D = average Jacobian on corpus D
v_D = J_D.T @ the same w
```

The direct implementation averages `J(context).T @ w` without storing each full
matrix. Linearity makes this equal to first averaging the matrices and then
multiplying by the same `w`.

Results:

| Comparison | Mean direction cosine |
|---|---:|
| Prose A versus prose B | **0.953** |
| Prose A versus code | 0.687 |

On held-out neutral prompts:

| Estimation source | Mean causal lift |
|---|---:|
| Pooled prose | **1.614** |
| Code | 1.109 |
| Direct `w` | 0.636 |

### Is this difference larger than ordinary prompt-sampling noise?

Yes, for these particular prompt banks. We saved the individual pullback
`J(x).T @ w` for every prompt, rather than only their average. For each pair of
corpora we measured the squared distance between their mean pullbacks,
normalized by within-corpus variation. We then randomly shuffled the corpus
labels 5,000 times. If prose and code were interchangeable samples, shuffling
should often create a difference at least as large as the observed one.

| Comparison | Normalized mean shift | Global permutation p-value | Significant concept-layer cells after FDR correction |
|---|---:|---:|---:|
| Prose A versus prose B | 0.058 | 1.000 | 0/28 |
| Prose A versus code | 0.819 | 1/5001 (<0.001) | 28/28 |
| Prose B versus code | 0.829 | 1/5001 (<0.001) | 28/28 |

![Corpus permutation test](../out/figs_core/fig14_corpus_permutation.png)

This is strong evidence for the narrow statement: **in this SmolLM experiment,
replacing the prose prompt distribution with the code prompt distribution
changes the estimated mean pullback by much more than finite sampling from the
two matched prose banks does.** It is not proof that an abstract property called
“domain” caused the difference. The code and prose banks also differ in syntax,
vocabulary, length, topic mix, and punctuation, and we tested only one manually
constructed bank of each kind.

### Inference

The averaging corpus is part of the estimator. Matched prose samples give very
similar directions, while code changes the early-layer direction and its causal
performance. The permutation test shows this geometry difference is not
plausibly explained by the observed prose-bank sampling noise. The code-derived
direction is not useless; it still beats direct `w` in 26 of 28 cells.

### Follow-up question

Is this a true domain-matching effect? In other words, would the code-estimated
direction become best when evaluated on code-like prompts?

### Replication or extension

We added 12 held-out code-like evaluation prefixes. The result was:

| Direction | Neutral evaluation | Code-like evaluation |
|---|---:|---:|
| Direct `w` | 0.636 | 0.814 |
| Code-estimated | 1.109 | 1.814 |
| Pooled-prose-estimated | **1.614** | **1.965** |
| Saved 25-Pile Jacobian | **1.641** | **2.171** |

![Two evaluation domains](../out/figs_core/fig9_bidirectional_corpus.png)

Everything improves on the code-like prefixes, but code estimation does not
become best. Therefore we have strong evidence that these sampled distributions
produce different estimates. We cannot yet identify “domain” as the causal
factor, or say that matching the estimation corpus to deployment is sufficient
or optimal.

This experiment is currently SmolLM-only because replicating it properly requires
re-estimating several corpus-specific Jacobians, not merely loading the existing
global matrices. The strongest larger-model follow-up would use many randomly
sampled, token-count-matched corpora and test the statistical interaction between
estimation domain and evaluation domain.

Application status: **use as a carefully qualified estimator result.**

---

## Experiment 6: can the reward-hacking database provide a real application?

### Experiment already performed

The repository contains 300 trajectories from a reward-hacked Qwen3.5-9B policy,
plus residual activations from the adapted and base models on the same transcript
tokens. The existing question was whether a residual probe, shortly after the
exploit starts, predicts whether judge-labeled substantive solution code appears
later.

Training used 161 process-exit examples. Evaluation used 22 examples employing a
different exploit mechanism, `always_equal`. After correcting an older analysis
that selected the best layer on the test set:

| Representation | Transfer AUC | Bootstrap interval |
|---|---:|---:|
| Adapted-model residual | 0.692 | 0.450–0.908 |
| Base-model residual on same prefix | 0.625 | 0.375–0.875 |

AUC 0.5 is chance. Both intervals include chance.

### Inference

The database is useful, but this result is not yet a monitor. “Substantive code
later” does not mean a correct solution, honesty, deception, or knowledge of
reward hacking. The test set is tiny and has already influenced analysis.

### Follow-up question

Does a constrained J-based representation retain genuinely predictive
information about future behavior across mechanisms, compared fairly with the
full residual, PCA, random projections, and everything visible in the text?

### Larger-model replication

The reward-hacking model is already 9B, but we do not yet have its matching
Jacobian. The existing 4B and 7B matrices cannot be substituted: they belong to
different models and residual spaces.

A newly released intermediate Qwen checkpoint has a much better-balanced
174-hack/126-non-hack evaluation set. A strong next study would:

1. define the behavioral outcome and measurement time before analysis;
2. split by problem identity and exploit mechanism;
3. use the exact available transcript prefix as the text baseline;
4. compare full residual, training-only PCA, random projections, and a
   paper-faithful token J-space at matched dimensions;
5. compute targeted vector–Jacobian products for the exact 9B model;
6. confirm on untouched examples with hardened-grader outcomes.

Application status: **best future-work motivation, not a completed J-Lens
success.**

---

## Experiments to keep out of the main story

### Chess

Chess provides an objective demonstration that averaging can mix two regimes:
predicting the origin square and destination square are different computations.
It is intellectually useful, but you can omit it if you do not understand it
well enough to defend it. Nothing in the main application depends on chess.

### MoE routing

The Jacobian performed poorly at predicting the next layer's selected experts
compared with a simple “routing persists” baseline. The split-half MoE subspace
study also used only six prompts per half. These are useful negative results but
too underpowered or tangential for the executive summary.

### Head-coordinate symmetry

The symmetry result is correct but acts inside attention-head coordinates. It
leaves the residual stream and residual Jacobian unchanged. Since the application
focuses on residual-stream J-Lens, keep this as one methodological aside at most.

### Emergent misalignment

Shared LoRA initialization confounded apparent subspace agreement, and the
behavioral result was not faithfully replicated. It is a good example of killing
a hypothesis, but it competes with the clearer J-Lens narrative.

---

## What the figures should say together

The application needs a sequence, not a gallery:

1. **Figure 1: the positive result scales.** `fig6_large_model_replication.png`
   shows that transport-aware steering beats direct steering on 135M, 4B, and 7B.
2. **Figure 2: the successful direction is distributed.** The cross-model rank
   figure tests how many spectral components are required and prevents a
   one-feature interpretation.
3. **Figure 3: a specific attractive SmolLM interpretation fails to scale.** The
   cross-model topic figure compares J, activation PCA, and random subspaces.
4. **Optional concrete panel:** `fig7_prompt_and_corpus.png` lets the reader see
   an actual next-token change.
5. **Optional methods panel:** `fig9_bidirectional_corpus.png` shows both the
   corpus effect and the failed domain-matching hypothesis.

That tells a strong research story:

> I reproduced the instrument, explored it cheaply on a small model, identified
> a reliable causal result and several attractive interpretations, then scaled
> the tests. The causal result replicated; one interpretation did not. I narrowed
> the claim accordingly.

## How to turn this into the 600-word executive summary

Write approximately five paragraphs in your own voice.

### Paragraph 1: prior work and question

Say that J-Lens already uses averaged residual Jacobians for token readout and
steering. Your question was how well this average operator transports
pre-specified concepts on open models, and whether its leading components deserve
feature-like interpretations.

### Paragraph 2: implementation and small-model experiment

State the published-lens reproduction number, then explain the held-out concept
split, direct-`w` baseline, random directions, and doses. Give the SmolLM
1.641-versus-0.636 result.

### Paragraph 3: larger-model replication

Give the 4B and 7B pullback/direct numbers. State that all 84 cells were positive
at all doses and exceeded all 30 sampled random effects at the reference dose.
Do not call the cells independent or claim formal family-wise significance.

### Paragraph 4: what did not generalize

Give either the cross-model rank result or the topic-subspace non-replication.
The clearest sentence is: the early top-eight J subspace beat PCA on SmolLM
(76% versus 47%) but lost decisively on Qwen (39% versus 99%) and OLMo (63%
versus 97%). Therefore causal operator utility generalized, while this compact
topic-decoding interpretation did not.

### Paragraph 5: conclusion and next experiment

Conclude that the averaged Jacobian is a useful transport estimator, not an
automatic inventory of universal singular features. Mention the small,
AI-assisted corpus construction and missing semantic-specificity/generation
tests. Propose the reward-hacking information-retention benchmark with balanced
data, exact text baselines, matched representation dimensions, and a native 9B
Jacobian.

## Statements you can safely make

- “I independently reproduced selected vectors from the published Qwen3.5-4B
  J-Lens artifact to cosine 0.9984.”
- “Transport-aware concept steering beat a same-target direct residual baseline
  on 135M, 4B, and 7B models.”
- “The result used held-out concept words, four doses, and 30 random directions
  per concept-layer cell.”
- “The useful effect was distributed across many singular components.”
- “A low-dimensional topic advantage found on SmolLM did not replicate on the
  larger models, where activation PCA was substantially stronger.”
- “Changing the estimation corpus changed pullback geometry and causal transfer,
  but the reverse-domain test did not establish that domain matching is optimal.”

## Statements not supported by the evidence

- “I invented Jacobian steering.”
- “I replicated the complete global-workspace paper.”
- “The top singular vectors are the paper's J-space features.”
- “Rank 64 is the universal workspace dimension.”
- “J-Lens is better than PCA in general.”
- “Corpus-matched Jacobians always transfer better.”
- “The reward-hacking probe detects deception or intent.”
- “The edit produces coherent or safe long-form behavior.”

## Final mental-model test

Before writing, answer these aloud:

1. What is being transported, from which layer to which layer?
2. What is `w`, and which words were excluded when constructing it?
3. Why is `J.T @ w` expected to increase the target score locally?
4. What makes the experiment nontrivial despite that mathematical expectation?
5. Which result replicated at 4B and 7B?
6. Which attractive SmolLM result did not replicate?
7. Why is the code/prose experiment not yet evidence for domain matching?
8. What exactly does the reward-hacking label mean?
9. Which parts came from the original paper, and which evaluations were ours?

If you can answer those nine questions without referring to the document, you
have enough of the project in your head to write the application honestly and
clearly.

## Primary literature to cite

- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html): the original J-Lens construction, token-indexed vectors, sparse J-space, steering, ablation, and ant/spider coordinate swap.
- [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/): the important activation-difference steering baseline that the current experiments have not yet run.
- [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112): a learned layer-to-output decoder and relevant readout baseline.
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405): broader prior work on reading and controlling population-level representations.

The application should make the boundary explicit: activation steering and
Jacobian steering already exist. The contribution here is the controlled
open-model replication, comparison, scaling test, and self-correcting audit.
