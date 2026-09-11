# The six questions: what the experiments mean, and where to go next

This note supersedes the application framing in the older guide where they
disagree. Chess is optional supplementary material. The main result is a
replication and empirical evaluation, not the invention of Jacobian transport.

## 1. “Seven vector–Jacobian products” means seven backward calculations

It does not mean seven Jacobians, or a Jacobian made of seven vectors.

Let h be a residual activation at an earlier layer and f(h) the downstream
activation. For this simplified single-position explanation,

    J[i,j] = ∂f(h)[i] / ∂h[j].

For SmolLM2, both activations have 576 coordinates, so J has 576×576 entries.
Suppose w is one fixed vector describing the destination concept “food.” Define
the scalar score s(h) = wᵀf(h). Its gradient is

    ∇h s = Jᵀw.

Automatic differentiation computes that vector without constructing every entry
of J. The conventional row-vector name is a vector–Jacobian product, wᵀJ;
transposing it gives our column vector Jᵀw. These are the same calculation.

We have seven target concepts: food, medicine, programming, law, sports, finance,
and female-associated words. Each supplies a different w, hence seven such
products. Each result is one 576-dimensional vector. This avoids calculating
the full 576×576 matrix when we only need those seven products.

There is an important counting correction: seven means seven backward products
**per batch per layer per corpus**, not seven backward passes for the whole
experiment. Each 18-text corpus has three batches of six. With three corpora and
four layers, the current script requests 7×3×3×4 = 252 backward products. It
averages across source positions and downstream positions as implemented in
`jlens/lens.py`; the one-position equation above explains the chain rule, not
all of that position weighting. Model weights are frozen: no training occurs.

## 2. Chess is optional

You do not need chess to understand or present the main project. Its interesting
lesson is that an averaged operator can mix different computational regimes:
predicting a move's origin square is not the same task as predicting its
destination. That is a useful controlled example, not a required application
pillar. Skip section 8 and the chess-specific reading session in the older guide.

## 3. Experiment 5, without the jargon

The question is: **if we estimate the same steering direction using different
background texts, do we get the same answer?**

Keep the model, layer, concept, intervention size, and evaluation prompts fixed.
Change only the bank of texts used to estimate the average Jacobian:

1. Eighteen prose sentences, bank A.
2. Eighteen different prose sentences about similar broad topics, bank B.
3. Eighteen code snippets.

For each bank D, calculate

    J_D = average context-dependent Jacobian over D
    v_D = J_Dᵀ w_food.

The implementation directly averages the products J(context)ᵀw_food instead of
first storing J_D. Linearity makes these equivalent for a fixed w and the same
averaging weights.

The food target does not change when the corpus changes. Code snippets are not
used to redefine food. They are used to estimate how a nudge at the earlier
layer travels through the model.

Next, run each direction on exactly the same twelve evaluation prefixes, such
as “On the table there was a”. None is in the estimation banks. At every prompt
token, add to the output of the selected layer:

    h' = h + 0.15 × s_layer × v_D / ||v_D||.

Here s_layer is the mean across the twelve prompts of each prompt's median
residual norm, excluding its first token and padding. Each direction has the
same injected norm. It is not “15% more food” or “15% of the coordinates.”

We measure whether previously held-out food words become more probable at the
next token, relative to matched control words. We do not measure factual
correctness or sustained behavior in a generated answer.

Two measurements answer different questions:

- **Cosine similarity:** are the estimated directions pointing the same way?
  Prose A/B average 0.953, while prose A/code average 0.687.
- **Causal lift:** when injected, how much do the directions change output?
  Pooled prose averages 1.614, code 1.109, direct w without transport 0.636.
  These are differences of log-probability changes, not percentage accuracies.

The pooled-prose direction beats the code-derived direction in 27 of 28
concept–layer cells. The prose estimates agree much more closely with each other.
This supports distribution sensitivity in this particular experiment. It does
not show code-estimated directions are universally worse: we have not performed
the reverse comparison on a separate code evaluation bank. The small,
AI-constructed estimation banks also differ in content, syntax, and length.

### A newly executed actual prompt example

Model: SmolLM2-135M. Layer: 20. Prompt: **On the table there was a**.
The target uses ten training tokens including vegetables, cooking, meal, rice,
kitchen, dinner, baking, soup, restaurant, and pasta. The following test words
were not used to construct w.

| Next-token probability | Unmodified | Direct w | Prose-derived Jᵀw | Code-derived Jᵀw |
|---|---:|---:|---:|---:|
| bread | 0.0118% | 0.0459% | 0.1115% | 0.0762% |
| cheese | 0.0093% | 0.0372% | 0.1374% | 0.0865% |
| sauce | 0.0143% | 0.0672% | 0.3682% | 0.1643% |
| salad | 0.0108% | 0.1002% | 1.4145% | 0.3649% |

The top next token changes from “large” (4.25%) to “plate” (6.94%) under the
prose-derived direction. Code-derived steering also makes “plate” the top token
(7.97%). That is an instructive distinction: the aggregate held-out-word score
and the probability of one illustrative word are different measurements.
This is a next-token experiment, not a generated sentence or a success rate.
The example was chosen for intelligibility, not as a randomly sampled case.

![Prompt-level effects and actual next-token probabilities](../out/figs_core/fig7_prompt_and_corpus.png)

Reproduce with `.venv/bin/python -m jlens.application_prompt_example`.
Exact token probabilities and top-ten lists: `out/rare/prompt_example.json`.

## 4. Were the other branches useless? No—but they answer different questions

**Interpreting subspaces:** a subspace can preserve useful topic information
without its individual axes each being a clean named concept. The eight leading
right-singular vectors form ONE eight-dimensional subspace. You still need the
actual activation h to compute its eight coordinates V₈ᵀh. In the existing small
topic test, top-eight J coordinates achieve 79% at layer 12, but activation PCA
achieves 90%. The useful conclusion is not that J discards all meaning: it is
that this test does not establish an advantage over ordinary activation PCA.
Neither classification accuracy establishes a causal feature dictionary.

**MoE:** comparing separately estimated subspaces is a meaningful stability
diagnostic. But the existing split used only six texts per half. Routing changes,
sampling noise, width, and model differences are entangled. This motivates a
larger routing-conditioned test; it does not yet establish a distinct MoE law.

**J times a diagonal activation matrix:** distinguish three different objects:

    J h               one transported activation/readout vector
    J diag(h)         a matrix whose column j is h[j] × J[:,j]
    J C^(1/2)         an operator weighted by activation covariance C

The diagonal version is the first-order response to multiplicative coordinate
changes. If δh = diag(h) ε, then δf ≈ J diag(h) ε. It can be meaningful for that
intervention. However, individual-coordinate scaling depends on the chosen
coordinate basis. Its SVD is not automatically a semantic decomposition.
Also, f(h) is not generally equal to Jh: a Jacobian maps local changes, and a
linear approximation to f itself ordinarily needs an expansion point and offset.

I found the covariance-weighted implementation in `jlens/whitened_basis.py` and
`jlens/prompt_readout.py`, not a clearly identified completed J diag(h) result.
Those should not be conflated. Existing covariance weighting was roughly a wash
at layers 12/20 and worse early in the tested topic task. That is a limited
negative result for that construction, not proof activation-aware methods fail.

## 5. What is actually new enough to lead with?

Using J to transport directions is already the mechanism of J-Lens, including
its steering interventions. It is not our invention. See the
[original paper](https://transformer-circuits.pub/2026/workspace/index.html).

The defensible contribution currently is an **open-model replication and audit**:
held-out concept words, equal-dose controls, multiple random directions, dose
sweeps, 135M/4B/7B replication, and an explicit estimation-corpus comparison.
That is empirical work, not a new general steering method. Direct w is also a
weaker comparator than a well-tuned contrastive activation-steering baseline.

The stronger research question is: **which information survives the projection
or transport, and does it retain behavioral predictive power under distribution
shift?** That can combine the subspace work with reward-hacking data instead of
discarding one or forcing every old experiment into a single story.

The 4B/7B completed figures are full-direction steering replications. They must
not be mislabeled as completed 4B/7B rank-truncation or topic-subspace studies.

## 6. Reward-hacking data: useful, but 48 GB is not 48 GB of independent examples

We have 300 Qwen trajectories and paired cached activations for the adapted and
base model. There are 275 grader-confirmed hacks and only 25 non-hacks. Many
tokens from one trajectory are not independent behavioral samples.

The promising question is whether internal information predicts a later
behavioral outcome before it appears in the prefix, and whether a constrained
representation retains that predictive information across exploit mechanisms.

### New audit performed for this follow-up

The existing transfer script trained on process-exit exploits and tested on
always-equal exploits. The label is whether a judge labeled the later code a
**substantive solution attempt**, versus exploit-only. It is not a correct
solution label, and not an honesty or deception label.

I reran this using cached residuals at four tokens after the annotated exploit
start. Both classes therefore already concern exploit-containing transcripts;
this is not a pre-exploit detector. Train: 161 exit examples, 96 positive. Test:
22 always-equal examples, 10 positive. Intermediate skeletal attempts are excluded.

The old script picked the layer with best test AUC (0.775 at layer 20). That is
test-set selection. The new script selects a layer using stratified five-fold
cross-validation on training data only, using fixed ridge strength 1000, and
then reports its transfer score. It also tests the paired base activations after
verifying the input tokens and captured positions match.

| Representation | Layer chosen using training CV | Transfer AUC | Bootstrap 95% interval |
|---|---:|---:|---:|
| Adapted model residual | 24 | 0.692 | 0.450–0.908 |
| Base model residual | 28 | 0.625 | 0.375–0.875 |

AUC is the probability a randomly chosen positive receives a higher score than
a negative (half credit for ties); chance is 0.5. Both intervals include chance.
They bootstrap held-out examples at a fixed trained model/layer and do not
include training uncertainty. The historical test set has already been used, so
even this corrected analysis remains exploratory, not fresh confirmation.

![Transfer audit across layers](../out/figs_core/fig8_reward_transfer_audit.png)

The base model was teacher-forced on the adapted model's transcript. This does
not tell us the base model would generate that exploit or continuation. Nor do
these scores establish a reliable adapted-versus-base difference.

Reproduce: `.venv/bin/python -m rh.transfer_audit`.
Results: `out/rh/transfer_audit.json`. Only compact key-position arrays are read;
the large dense activation arrays are not loaded. No GPU was needed.

### What would make this a stronger application experiment?

1. Fix the outcome and timing. Manually audit whether the target behavior has
   already started at each annotated position; also test before the exploit.
2. Compare full residuals, training-only PCA, matched random projections, and
   J-based representations at the same dimensionality. Fit preprocessing on
   training data only. Include an exact-prefix text baseline: the older
   reasoning-only bag-of-words baseline omits other available prefix content.
3. Define J-space precisely. Top singular subspaces of J are not the paper's
   sparse token-vector J-space. Token readouts and sparse decomposition test a
   different hypothesis; report them separately.
4. Use a Jacobian for this exact 9B model and activation convention. The existing
   4B/7B matrices cannot be substituted. Cached activations alone do not provide
   model derivatives. Targeted products can avoid a full 4096×4096 Jacobian.
5. Obtain new held-out problems and enough non-hacks, preferably with hardened
   grading. Train/test grouping must respect problem identity and repeated
   samples. More GPU compute alone does not solve label imbalance.
6. Only after predictive validation, test interventions against hardened task
   success and hack rate. Increasing “honesty” token scores is not evidence of
   reducing reward hacking. Execute exploit code only in an isolated evaluator.

My recommendation: retain the completed J-Lens audit as the reliable result;
make behavioral-information retention the focused extension. The reward data
gives it a relevant test bed, but the new audit does not yet justify a claim
that we can detect or causally control reward hacking.
