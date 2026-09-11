# APPLICATION GOOGLEDOC — final (built from docs/APPLICATION_RESEARCH_GUIDE.md)

> **How to use this file.** Paste into a Google
> Doc → share "Anyone with the link" → *view*. Replace each markdown image line
> with the PNG from `out/figs_core/` or `out/figs_app/` (Insert → Image, then
> delete the markdown line). The executive summary ends at the `# 1.` heading.
>
> **Voice note.** This is drafted in a deliberately explanatory voice — the way
> someone new to interpretability would write after learning each piece slowly.
> Basic ideas are spelled out on purpose. Polish into your own voice before
> submitting; keep the numbers exactly as they are (all checked against the
> saved JSON outputs on Sep 10).
>
> **[USER] items:** the time log (footer), the "anyone with the link" setting,
> and the exact hour count + prior-vs-new-work split (see §12). The guide is
> explicit that the hour accounting must be honest, so those lines are left for
> you to fill in.

---

# When does an averaged Jacobian mean anything?

### J-Lens as a transport operator: useful pullbacks, and what averaging breaks

**Jeeva Sukumar** — MATS 12.0 application

Code: `jlens-transformation` (this repo). Main model: SmolLM2-135M; replications
on Qwen3.5-4B (external published lens) and OLMo-3-7B (local lens); chess-only
GPT-2 for the oracle experiment. Everything below runs on a 34 GB laptop.

---

# Executive summary

**The question.** What does a prompt-averaged residual-stream Jacobian actually
represent when the underlying computation differs across contexts — and when is
it safe to interpret, or intervene on, its leading directions? J-Lens takes a
language model, computes how small changes in an early layer move a later layer
(a Jacobian), averages that map over prompts and positions, and then reads
directions out of the average. My project asks what that averaged object means.

**What works.** If you first say which output concept you want (food words, say)
and then ask the average Jacobian for the input change that best pushes the
model toward it — the pullback `Jᵀw` — you beat the obvious baseline of just
adding the output direction itself. On SmolLM2-135M, seven concepts × four
layers (28 cells), with the scored words held out from the words that built the
direction: mean lift **1.641** nats for the pullback versus **0.636** for direct
injection. Every pullback cell is positive at all four tested doses, and at the
reference dose every pullback beats every one of 30 random-direction controls
sampled for its cell. These are descriptive finite-sample comparisons, not
family-wise significance tests.

![fig4 robustness](out/figs_core/fig4_pullback_robustness.png)

**What the average represents.** The map depends on the distribution it was
averaged over. Two disjoint 18-prompt prose banks, matched by genre, give
pullbacks with cosine **0.953** and near-identical causal effects (1.604 vs
1.605). An 18-prompt code bank gives cosine ~**0.688** to either prose bank and
transfers worse to prose (1.109) — though it still beats direct injection
(0.636). On held-out code prefixes the gap narrows to a coin flip (1.958 vs
1.814, prose winning 16/28 cells) — matched-distribution estimation wins, and
the code bank is handicapped twice over (evaluation turf, prose-biased
concepts). So averaging preserves genuinely useful signal, but the object you
get is distribution-specific. (These are small, hand-built, AI-assisted prompt
banks — a controlled demonstration, not a population estimate.)

![fig5 corpus](out/figs_core/fig5_corpus_conditioning.png)

**Where the effect lives.** It is not one beautiful direction. Keeping only the
top singular component gives **−0.13** lift; 32 components give **1.83** and 64
give **1.89**, slightly above the full 576-component **1.75** at matched
intervention size. The weak tail hurts, which is also why exactly solving the
inverse (pseudoinverse: ~**0.00** lift) fails while the transpose succeeds.

![fig2 truncation](out/figs_core/fig2_truncated_pullback.png)

**The failure mode.** In a chess model with ground-truth labels, the pooled
Jacobian's strongest coefficients track which half of a move is being typed
(correlation **0.550**) more than any chess property (best 0.392). Averaging can
manufacture mixture directions. Conditioning on the known regime helps
partially, not universally.

**Limits, stated up front.** The main metric is one 135M model (the controlled
pullback comparison is replicated on 4B and 7B models with the same shared
prompt/concept design); prompt banks are small and hand-built; tests within a
concept are correlated; there is no semantic-specificity result. What survives:
an averaged Jacobian is a useful distribution-specific transport map, not an
automatic inventory of universal features. The chess result below is
independent support for the same averaging concern, not a separate storyline.

---

# 1. Background, from zero

I came into this project without a working mental model of what a Jacobian lens
even is, so this section explains each piece the way I had to learn it. Nothing
here is assumed.

## 1.1 Tokens and logits

A language model never sees text. A tokenizer first chops text into integer
token IDs — a token can be a whole word (" Paris"), a fragment ("icator"), a
punctuation mark, or a special marker. The model then scores every vocabulary
token with a number called a **logit**. Softmax turns logits into probabilities.

If `z_t` is the logit for token `t`, raising `z_t` relative to the others makes
`t` more likely. Most of this project's causal effects are reported as changes
in log-probability, measured in natural-log units called **nats**. A +1 nat
change multiplies the odds by about e ≈ 2.72, before accounting for the other
tokens moving too. When I write "lift 1.641", I mean the log-probability of the
target words rose by 1.641 on average — a large, unmistakable change.

This matters because tokenization can masquerade as meaning. A result about
"chess concepts" can really be about which half of a tokenized move is being
typed (that exact thing happened to me — §5). Whenever a result looks semantic,
the first question I learned to ask is whether it is really about tokens.

## 1.2 The residual stream

At every token position, a transformer carries a vector of `d_model` numbers
through the network. Each attention block and each MLP block reads that vector
and writes an update back into it. The shared vector is the **residual stream** —
the model's running working memory, one vector per position per layer.

For SmolLM2-135M, `d_model` is 576: every activation I touch in the main
experiments is a point in a 576-dimensional space. A **direction** is just
another 576-dimensional vector. Adding a scaled direction to an activation —
**activation steering** — is the intervention this whole project studies.

At the very end, the final residual vector is normalized and multiplied by the
**unembedding matrix** `W_U` (one row per vocabulary token):

```text
logits = W_U · h_final
```

So each row of `W_U` is an output-space direction "for" a token. Pushing an
intermediate residual vector through `W_U` and reading off the top tokens — the
**logit lens** — tells you which words that vector would directly promote if it
were sitting at the output. It is a readout, not an explanation: a coherent
list is suggestive, and it can be wrong for boring reasons (frequency bias,
arbitrary sign, or later nonlinear layers scrambling the effect).

## 1.3 What a Jacobian is, and why averaging it is a choice

Suppose `h_l` is the residual stream at layer `l` and `h_T` is a later residual
state. The **Jacobian** `J_l = ∂h_T/∂h_l` is the matrix of all first-order
sensitivities: for a small nudge `δ` at layer `l`,

```text
h_T(h_l + δ) ≈ h_T(h_l) + J_l·δ
```

Three things about this object took me a while to internalize. First, it is
**local**: in a nonlinear network the Jacobian depends on the prompt, the token
position, the attention pattern, even which experts a router picked on that
pass. Second, it is **expensive**: a full `d×d` Jacobian costs on the order of
`d` backward passes (for a 4096-wide model, thousands). Third — and this is the
project's whole subject — J-Lens replaces the local object with an **average**
over prompts, source positions, and later positions:

```text
J_l = E over [prompts, source position, later position] of [∂h_T/∂h_l]
```

Averaging is not bookkeeping. It decides *which distribution of computations*
the resulting map describes. If the prompts contain two different computations,
the average is a mixture. Everything in §5–§6 is downstream of that sentence.

A cheap but important implementation detail: you often don't need the full
matrix. If you only care how layer `l` affects *one* token's logit with
unembedding row `w`, one reverse-mode pass gives you `Jᵀw` directly (a
vector-Jacobian product) instead of materializing all of `J`. The repo's
`jlens/lens.py` builds on this, with care about which residual point a "layer
index" means, causal masking, and excluding attention-sink and final positions
from the average. The math tests pin this down: the optimized computation
matches a slow explicit sum to 5.364e-7 at signal scale 2.052, with zero causal
leakage, and a layer-to-itself Jacobian is the identity. My first implementation
matched a published lens at cosine 0.94 and *felt* right; fixing the layer
convention and position reduction raised it to 0.9984. Lesson one, learned the
hard way: a high similarity score is not validation when the right answer is
much closer to one.

## 1.4 Singular vectors, eigenvectors, and the pullback

Any real matrix factors as `J = UΣVᵀ` (**SVD**). The columns of `V` are input
directions, the columns of `U` are output directions, and each singular value
`σᵢ` says how strongly `J` carries `vᵢ` into `uᵢ`: `J·vᵢ = σᵢ·uᵢ`. The type
discipline matters enormously and I got it wrong at first (an early version of
the steering code injected `u` — the *output* side — at the source layer; the
fix is documented and the reruns use `v`):

- to **inject** for maximum linear downstream change at fixed input size, use `vᵢ`;
- to **read** what comes out at the target, look at `uᵢ`;
- never inject `uᵢ` just because its token readout looks interpretable.

Eigenvectors (`J·q = λq`) are a different object: directions the map sends to
multiples of themselves. They are conceptually tempting (a "persistent axis"),
and the repo's original headline was "read with eigenvectors, steer with
singular vectors" — but decompositions can hinge on coordinate conventions and
near-degenerate spectra, so that framing no longer leads.

The **pullback** is the construction at the heart of the positive result. Say
you want more of some output concept, summarized as a target direction `w`
(e.g. the average unembedding row over several food words). You want the source
nudge `x` that maximizes the downstream score `wᵀh_T`. Linearized,
`wᵀJx = (Jᵀw)ᵀx`, so among fixed-size nudges the best is `x = Jᵀw` — the target
pulled back through the average map. In the singular basis this reads
`Jᵀw = Σᵢ σᵢ⟨uᵢ,w⟩vᵢ`: a *combination* of many components weighted by
downstream relevance, not a single discovered feature. That identity prefigures
the truncation result (§3): the intervention works because it coordinates many
directions, and one component alone should be expected to fail.

A natural alternative is the **pseudoinverse** `J⁺w` — "solve `Jx = w`
exactly". It fails completely here (lift ≈ 0.00 — the §2 table). The reason is instructive
rather than mysterious: the pseudoinverse divides by small singular values, so
the noisiest, worst-estimated directions get the largest weight; the transpose
multiplies by them and suppresses the tail. "Solve it exactly" is the wrong
objective for a noisy, ill-conditioned average operator.

## 1.5 Three things with confusingly similar names

The repo (and sometimes the paper's readers) blur these, so I keep them
separate everywhere below:

1. **A token J-Lens vector** (`v_t = JᵀW_U[t]`, one per vocabulary token) — the
   paper's object; a collection of such vectors is treated as an overcomplete
   frame for sparse decomposition. The paper does *not* claim the SVD
   components are these vectors.
2. **A singular component** (`uᵢ`, `vᵢ`) — a matrix factorization product whose
   index comes from the spectrum, not the vocabulary. Unembedding `uᵢ` gives
   tokens, but that list interprets a matrix component; it is not the paper's
   token-indexed feature.
3. **A concept pullback** (`Jᵀw` for a multi-word `w`) — a linear combination
   of token pullbacks built for a pre-specified downstream objective.

What the evidence supports: the average contains usable transport information; a
moderate leading subspace carries most of the causal effect; the singular basis
is computationally useful; readable token lists for single components do not by
themselves establish features. What it does not touch: the paper's sparse frame
claim, or residual features in general.

---

# 2. Experiment 1: the pullback beats the no-J baseline

## 2.1 The question and the setup

Given an output concept, can the average Jacobian find a better residual
intervention than just adding the output direction at the source layer? The
second option — direct injection of `w` — is the essential baseline. Without
it, any movement along a gradient direction would look like a discovery when it
is partly arithmetic.

Model: SmolLM2-135M. Source layers 4, 12, 20, 26 (target: layer 28). Concepts:
food, medicine, programming, law, sports, finance, female. Each concept starts
as ~20 words; only single-token words are kept, split by alternating index so
one half builds `w` and the other half scores the intervention — the scored
words never constructed the direction. Evaluation: 12 neutral text prefixes.
28 concept-layer cells. Dose: 0.15 × the layer's median residual norm.

For every held-out word the script picks a **control token** with similar
baseline likelihood and low similarity to `w`, and reports *held-out lift minus
control lift*. That is much stronger than checking whether the construction
words went up (nearly guaranteed). Methods compared: pullback `Jᵀw`, direct `w`,
pseudoinverse, ridge inverses at three strengths, the best single SVD component
aligned with `w`, and a random direction at matched norm.

## 2.2 Results

![fig1 operator](out/figs_core/fig1_pullback_operator.png)

| Method | Mean held-out lift | Cells beating own random control and 0.2 | Mean unrelated-text loss increase |
|---|---:|---:|---:|
| `Jᵀw` (pullback) | **1.64** | **28/28** | +0.172 |
| Ridge, λ=1 | 0.85 | 25/28 | +0.134 |
| Best single SVD component | 0.76 | 20/28 | +0.112 |
| `w` without `J` | 0.64 | 17/28 | +0.209 |
| Pseudoinverse | 0.00 | 3/28 | +0.022 |
| Random | 0.01 | 0/28 | +0.031 |

Mean pullback lift is ~2.6× the direct baseline. The advantage is
depth-dependent, which is a nice internal consistency check I did not design
in: near the target, residual connections make `J` close to identity, so all
methods converge:

| Source layer | Direct `w` | Pullback `Jᵀw` | Ratio |
|---:|---:|---:|---:|
| 4 | 0.11 | 0.56 | 5.2× |
| 12 | 0.16 | 1.21 | 7.7× |
| 20 | 0.68 | 2.29 | 3.4× |
| 26 | 1.60 | 2.51 | 1.6× |

## 2.3 It survives a dose sweep and a real null

One dose and one random draw prove nothing (I learned both lessons separately
— §8, failures 4 and 5). The follow-up reruns pullback-vs-direct at four doses with **30 random
directions per layer** (each random edited pass is reused across the seven
concepts, so within-layer concept tests are correlated — stated, not hidden):

![fig4 robustness](out/figs_core/fig4_pullback_robustness.png)

| Dose (fraction of median norm) | Pullback | Direct `w` | Pullback cells positive |
|---:|---:|---:|---:|
| 0.05 | 0.571 | 0.230 | 28/28 |
| 0.10 | 1.132 | 0.442 | 28/28 |
| 0.15 | **1.641** | 0.636 | 28/28 |
| 0.20 | 2.084 | 0.815 | 28/28 |

At 0.15, all 28 pullback cells beat all 30 of their randoms (median per-cell
z-score 8.31); direct `w` manages that in 16/28 (median z 2.58). With 30 draws
the smallest one-sided empirical p is 1/31 = 0.032, so I do not present these
as 28 independent discoveries with family-wise correction. The plain
descriptive claim is the honest one: across the whole fixed grid every pullback
was positive at every dose, and at the reference dose every pullback beat every
matched random effect sampled for its cell.

## 2.4 Replication on two larger models

Same causal protocol, two larger architectures, layers at matched depth
fractions (SmolLM2 4/12/20/26→28; Qwen3.5-4B 4/13/21/28→30; OLMo-3-7B
4/12/22/28→30). Qwen uses the external published J-Lens artifact (estimated by
its authors on 25 Pile documents); OLMo uses a locally computed 4096-wide
Jacobian on 25 Pile documents with the same position convention.

![fig6 replication](out/figs_core/fig6_large_model_replication.png)

| Model | Params | Pullback @0.15 | Direct `w` | Ratio | Cells beating all 30 randoms |
|---|---:|---:|---:|---:|---:|
| SmolLM2 | 135M | **1.641** | 0.636 | 2.58× | 28/28 |
| Qwen3.5 | 4B | **1.437** | 0.730 | 1.97× | 28/28 |
| OLMo-3 | 7B | **1.261** | 0.669 | 1.88× | 28/28 |

Every cell positive at every dose in every model (84 cells); median null
z-scores 8.31 / 16.67 / 23.26; direct clears the same bar in 16/28, 26/28,
21/28. The depth crossover replicates too (Qwen 4.39×→1.28× early-to-late; OLMo
converging to ~1.25× late — early-layer absolute direct effects are tiny, so I
do not headline early ratios). This spans ~52× in parameters with an
independently estimated lens, which retires the one-small-model objection to
the *efficacy* claim. It keeps the shared limitations: same seven concepts,
same 12 prompts, same intervention design. And it is efficacy, not specificity
— the capital-city work shows a France intervention also moves Spain and
Germany, partly because low-probability tokens have more headroom to rise.

## 2.5 What this does and does not show

It shows the averaged Jacobian is not decorative: it improves a held-out-word
objective over direct injection, across doses, nulls, depths, and (for
efficacy) models. It does not show the pullback "understands" the concept, is
specific, or is harmless — unrelated-text loss rises (+0.172, comparable to
direct's +0.209). And a gradient direction moving its own objective is
half-expected; the content is all in the controls: the no-J baseline, held-out
words, depth pattern, finite doses, the inverse failing, and the truncation
result next.

---

# 3. Experiment 2: the effect lives in a subspace, not a direction

If the pullback is a combination `Σᵢ σᵢ⟨uᵢ,w⟩vᵢ`, which parts matter? Keeping
only the first `k` components (layers 12 and 20, seven concepts, same
held-out metric, each truncated vector re-normalized to the same dose — so this
tests *direction*, not captured norm):

![fig2 truncation](out/figs_core/fig2_truncated_pullback.png)

| Components kept | Mean lift | Share of full pullback energy |
|---:|---:|---:|
| 1 | −0.13 | 1.6% |
| 8 | 1.11 | 16.9% |
| 16 | 1.55 | 32.5% |
| 32 | **1.83** | 49.5% |
| 64 | **1.89** | 69.3% |
| 128 | 1.84 | 84.8% |
| 256 | 1.78 | 95.8% |
| 576 (full) | 1.75 | 100% |

One component fails outright. 32–64 components slightly beat the full vector:
the weakest tail adds energy but subtracts effect — the same mechanism behind
the pseudoinverse failure, with the tail removed instead of amplified. This is
my cleanest argument for "useful subspace" over "single feature", and it does
*not* prove rank 64 is special in general — that optimum is for two layers,
seven concepts, one model, one dose, one normalization.

A related decoding check (`subspace_meaning`: 72 prompts, six topics,
leave-one-out nearest-centroid, eight dimensions) finds `J`'s top input
directions concentrate topic information better than activation PCA early
(layer 4: **76%** vs 47%, random 34%, shuffled-labels 15%) — `J` ranks by
downstream sensitivity, PCA by activation variance — while PCA wins later
(layers 12/20/28: 90/88/97% vs 79/85/69%). At the target layer `J` nears
identity, its spectrum goes degenerate, and its ordering privilege dissolves
(layer 28 ≈ a random 8-D subspace). So "J finds downstream-relevant information
earlier than PCA" is a layer-4-in-one-model claim, not a general superiority
result.

---

# 4. Experiment 3: readable components are not reliable features

If a target-side component `uᵢ` unembeds to a coherent cluster (food words,
say), does injecting `vᵢ` move *other* food words? The naive test is circular:
if `t` is `uᵢ`'s top readout token, pushing `vᵢ` raises `t` nearly by
arithmetic. The protocol instead builds a centroid from the top readout,
excludes the top-200 readout tokens, finds nearby held-out tokens, matches
controls by baseline probability, and asks whether held-out tokens rise more
than controls.

Leading components, split into thirds by a cross-model readability score:

| Readability band | Mean readability | Directions passing held-out steering |
|---|---:|---:|
| High | 47.5 | 8/20 (40%) |
| Middle | 11.3 | 8/20 (40%) |
| Low | 3.4 | 7/20 (35%) |

A 14× readability change moves the pass rate from 35% to 40% — essentially
nothing. Readable directions re-tested on a 40-prompt bank: 19/37 pass the
whole-bank bar. (A reported split-half reliability of r ≈ 0.94 / 81% verdict
agreement exists but its exact producing artifact needs restoration before I'd
put weight on it; qualitatively, the mixed success is not obviously run noise.)

The safe reading, and it fits rather than contradicts §2: single components are
weak and unreliable, while the target-chosen *combination* works. A readable
unembedding does not license treating a component as a standalone semantic
variable — and none of this tests the paper's token-indexed sparse frame, which
is a different object (§1.5).

---

# 5. Experiment 4: averaging can mix computations (chess)

## 5.1 Why chess

Natural-language feature labels are subjective — I can always talk myself into
seeing a theme. Chess supplies oracle labels from `python-chess` (legal/illegal
squares), and the model is genuinely doing the task (over 99.5% of
square-token probability mass lands on legal squares in sampled positions).
The tokenizer emits each move as two square tokens, so the model alternates
between two known jobs: pick the origin square, then the destination square. A
perfect test of whether one pooled Jacobian represents both computations.

## 5.2 The pooled map leads with format, not chess

150 generated games, 7,758 positions, seven layers, top 24 components per layer;
position coefficients correlated with oracle labels after removing a linear
ply-number trend; significance calibrated by 400 permutations of the
family-wise maximum (95th-percentile threshold 0.048):

| Label | Best absolute correlation |
|---|---:|
| Origin-vs-destination slot | **0.550** |
| Queen can move | 0.392 |
| Capture available | 0.274 |
| Rook can move | 0.268 |

The strongest signal is the phase of the output format. (An older repo summary
quotes multivariate R² values for this analysis, but no checked-in script
produces that JSON — I exclude those numbers until the producer is restored.
The correlations above are the fully reproducible result.)

## 5.3 The two regimes genuinely differ — early

Three Jacobians from the same games: origin-only, destination-only, pooled. At
layer 1 the flattened cosine between origin and destination maps is **0.211**,
and their difference has norm **1.62×** the pooled matrix's norm; similarity
climbs to 0.961 by layer 6. So the mixture problem is worst early, where the
tasks use sharply different local transport.

## 5.4 Conditioning helps — partially, asymmetrically

Define a per-position legality score (logsumexp over legal squares minus
logsumexp over illegal squares), differentiate w.r.t. the residual stream, and
compare its leading subspace against pooled-J, parity-matched conditional-J,
activation PCA, 30 random subspaces, and a split-half ceiling (400 positions,
200 per regime, layers 5–6):

![fig3 conditioning](out/figs_core/fig3_averaging_and_conditioning.png)

Destination positions, layer 5:

| Rank | Pooled `J` | Conditional `J_dest` | Activation PCA | Random | Split-half ceiling |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.098 | **0.239** | 0.040 | 0.008 | 0.513 |
| 16 | 0.067 | **0.127** | 0.117 | 0.031 | 0.396 |
| 64 | 0.135 | 0.178 | **0.225** | 0.125 | 0.390 |

Rank-4 overlap improves 2.4× over pooled and beats PCA and random — but at
rank 64 PCA wins, and for *origin* positions the matched conditional map does
not consistently beat pooled at all. Conditioning is a partial correction with
an unexplained asymmetry (my hypothesis, not a conclusion: destination choice
is conditional on an already-picked origin, hence a more distinct local map).
Also note the ceiling: the legality-gradient subspace itself needs 67–85
principal components for 90% of its variance (origin 67/47, destination 85/66
at layers 5/6). The task-relevant geometry is moderate-rank. Hunting one
beautiful direction is the wrong inductive bias — the same moral as §3.

## 5.5 The operational moral

Treat the averaging distribution as a model choice: measure estimator
stability, inspect known regimes, compare pooled vs conditional lenses on
held-out causal tasks. Never average first and ask what mixed later.

---

# 6. Experiment 5: the map is distribution-specific

The old split-half result mixed sampling noise with corpus shift, so this
experiment isolates them without materializing full Jacobians: seven
vector-Jacobian products per layer (one per concept direction) on three
18-prompt corpora — `prose_a` and `prose_b` disjoint but genre-paired (weather,
history, biology, travel, law, finance, medicine, engineering), plus a `code`
bank (Python, JavaScript, SQL, Rust, Java-like snippets) as a deliberate domain
shift. All causal scoring uses the separate 12-prompt neutral bank from §2; no
evaluation prompt appears in any estimation corpus.

![fig5 corpus](out/figs_core/fig5_corpus_conditioning.png)

Geometry (mean cosine, 28 concept-layer cells):

| Comparison | Mean cosine |
|---|---:|
| Disjoint matched prose A vs B | **0.953** |
| Prose A vs code | 0.687 |
| Prose B vs code | 0.689 |
| Fresh pooled prose vs saved global `J` | 0.878 |

Depth-dependent, as expected near identity: prose/code cosine 0.498 (layer 4)
→ 0.941 (layer 26); prose A/B 0.871 → 0.999.

Held-out causal transfer (same 0.15-norm dose, so this is direction, not size):

| Direction source | Mean held-out lift |
|---|---:|
| Saved global `J` | **1.641** |
| Fresh pooled prose | 1.614 |
| Prose A | 1.604 |
| Prose B | 1.605 |
| Code | 1.109 |
| Direct `w`, no `J` | 0.636 |

Prose A and B differ by −0.001. Pooled prose beats code-derived in 27/28 cells
(+0.505 on average); code still beats direct `w` in 26/28. That last line is
the most informative in the project for my central question: averaging
preserves useful residual-stream signal *and* the transported direction depends
on the estimation distribution.

Reverse-domain check (run Sep 11, after I doubted the paragraph above).
Re-running the identical protocol but scoring causally on 12 held-out *code*
prefixes (`out/rare/pullback_corpus_codeeval.json`): prose-estimated 1.958,
code-estimated 1.814 — the prose advantage collapses from 27/28 cells to 16/28,
a coin flip, with code-estimated winning food (2.801 vs 2.535), finance (2.576
vs 2.565), and programming (0.446 vs 0.314) outright, and beating direct
injection 28/28 with every cell positive. Two honest qualifications follow:
the prose-turf gap partly reflects evaluation mismatch, and the concept set is
prose-biased by construction (6/7 topics never occur in code), so the code bank
is handicapped for these concepts before estimation starts. The geometry gap
needs no such qualification — it uses no evaluation prompts at all. Net:
matched-distribution estimation wins; mismatched estimation still beats doing
nothing, everywhere.

Provenance disclosure (required, not optional): the prose/code banks are small,
hand-built with AI assistance, and differ in syntax, content, and length
profile. This is a controlled demonstration of distribution dependence, not a
population estimate for "code versus prose". The numerics are sound (batched
vs batch-size-one check: 1.2e-7 on cosines, 1.4e-6 on lifts; padding masked out
of the destination average). A larger study needs many corpus splits with
uncertainty over corpus sampling, not just seven fixed concepts.

Supporting older evidence, kept at arm's length: full-`J` split-half leading
subspaces on six-prompts-per-half agree modestly (SmolLM2 top-2% overlap
0.61@L4 / 0.37@L12; OLMoE 0.26@L4 / 0.30@L12). With six prompts per side,
estimator noise and corpus mismatch are confounded — the safe sentence is that
leading subspaces *can* vary substantially across small disjoint corpora, and
larger-corpus bootstraps are needed. Do not quote MoE-vs-dense reproducibility
ratios off this sample.

---

# 7. How the pieces fit

1. **Useful as an operator** (§2): the pullback beats direct injection across
   doses, nulls, depths, and three models' worth of efficacy replication.
2. **Distributed, not atomic** (§3–§4): one component fails; 32–64 coordinated
   components work best; readability doesn't predict causal success.
3. **Mixtures lurk in the average** (§5): format dominates the pooled chess map;
   regimes differ sharply early; conditioning partly corrects.
4. **Therefore conditional and task-relative** (§6): the average answers "across
   *this* prompt/position distribution, which source nudges propagate where?"
   — not "what are the model's universal semantic variables?"

---

# 8. Failed approaches, and what each one changed

These cost real time and each one changed the protocol. I list them the way I'd
want someone else's application to list them: claim, control that killed it,
rule I adopted.

1. **Wrong layer bookkeeping** (cosine 0.94 to a published lens, still wrong
   convention; fixed → 0.9984). Rule: match a harder target than "looks close";
   test against direct differentiation, not vibes.
2. **Injecting `u` instead of `v`.** Several steering "results" were type errors
   (output-space vector injected at the source). Rule: write the type next to
   every vector, every time.
3. **Hidden position.** Same direction and dose: +0.58 nats at the subject token
   vs −3.24 at the final token in SmolLM2 (a second analysis: +12.3σ vs −1.3σ).
   Position can flip the sign. Rule: position is part of the intervention
   definition — always name layer, token, dose, and normalization.
4. **One dose.** At saturation many directions bulldoze the same output and the
   efficacy/specificity relationship can reverse. Rule: dose-response curves
   with a predeclared validity region.
5. **One random direction.** A single draw once beat a targeted direction by a
   large factor — an anecdote, not a null. Rule: ≥30 matched random draws per
   cell.
6. **Weak unsupervised objectives.** 1-D DAS barely moved the state; 8-D DAS fit
   training phrasings but not held-out ones; a de-sinked variant moved one
   phrasing only; a generalized-eigenvalue ratio inflated by collapsing its
   denominator. Rule: held-out templates, working intervention scale, inspect
   numerator *and* denominator.
7. **Layer confounds.** Jacobian amplification looked predictive of edit success
   pooled across layers; within-layer correlation ≈ −0.064, adding `J` to a
   layer-only regression adds ~0 R². Rule: stratify before claiming.
8. **Shared initialization looking like convergence.** "Independent" fine-tunes
   were up to 99.94% similar before training — same seed. Rule: control seeds
   before claiming learned agreement.
9. **Text already containing the label.** A concealment probe at AUC 1.00 was
   matched by bag-of-words text features (0.985). Rule: score before the
   evidence appears; benchmark against a text-only monitor.
10. **The Jacobian for MoE routing.** Predicting next-layer experts: Jacobian
    1.3/8 vs do-nothing persistence 7.2/8 (chance ~1.0). Clean negative against
    a strong simple baseline; the expensive map adds nothing *for this task*.
11. **A saturated causal dose.** First intervention battery at α=1.0: common-mode
    (mean pairwise cosine of induced logit changes) hit 0.46 pooled, 0.97 in one
    arm — directions indistinguishable. The kill criterion fired, the run was
    discarded, and the battery re-ran at validated low dose (common-mode
    ~0.005). Rule: validate the dose *before* comparing directions.
12. **Reward-hacking substrate.** A "will a real solution follow?" probe hit AUC
    0.811 before the solution text existed — but the system prompt spells out
    the exploits, so this is closer to disobeying instructions than discovering
    intent. Kept, deflated, out of the headline.

---

# 9. Limitations, missing pieces, and next experiments

**Conceptual limits.** Passing these tests doesn't certify anything: a method
can be stable, causal, and still not mean what its label says. The pullback's
direct effect is half-expected from its gradient construction — the science is
in the baselines, not the bare movement. No specificity result exists; headroom
confounds every cross-target comparison so far.

**Scope limits.** One small model carries the main metric (efficacy replicated
at 4B/7B under the same design); banks are small and hand-built (code-turf
scoring rerun in §6 narrows but does not erase the transfer gap); concept tests
correlate within layers; interventions are all-position averages at fixed
normalization; chess is one model, one tokenizer, 400-position Q1Q2; origin
conditioning inconsistent; rank-64 PCA beats conditional-J.

**Provenance gaps I will not paper over.** An older chess summary quotes R²
values whose producing script is not checked in — excluded from this
application. The top-level `verify.py` mixes live recomputation with saved-JSON
summaries (notably the chess ablation and the broad steering claim, which it
demonstrates on one example) — §11 states exactly which is which. The
privileged-basis battery changed its coherence metric and tail statistic after
diagnostics (hypotheses and kill criteria were pre-written; the metric change
is disclosed, not hidden). Several prose reports still carry numbers whose
producers are unclear; I cite only JSON-backed values above.

**What I'd run next, in order:** (1) cross-corpus transfer at scale — large
named datasets, many splits, uncertainty over corpus sampling, prompt-specific
gradient upper bound; (2) conditional J-Lens on natural language with
pre-declared regime labels (distance, question/statement, code/prose, turn
role); (3) a paper-faithful test of token-indexed sparse-frame vectors vs
concept pullbacks vs probe baselines on disjoint synonym sets; (4) the full
no-J/pullback/truncation comparison pre-registered on one larger open model;
(5) provenance restoration before any new breadth.

---

# 10. Appendix A: the head-symmetry calibration (one story, kept small)

The residual-stream results above are the application. Separately, I calibrated
my instruments against an exact symmetry *inside* attention heads, where no
elementwise nonlinearity acts: replace `W_V → R·W_V`, `W_O → W_O·Rᵀ` and the
head computes exactly the same function (exact in arithmetic; ~4×10⁻⁵ logits in
float32 — the algebra covers any invertible map, numerically demonstrated on an
ill-conditioned example at condition number 115; the freedom is the full
general-linear group, 4096 numbers per head, not just rotations). Under this
change, head-column readouts are ~99% replaced (SmolLM2 1.5%, Qwen 0.4%, Llama
0.3% retained; the same rotation in an MLP moves logits by ~2 — a
16,000–51,000× gap), while the combined OV circuit survives exactly and the
residual stream is untouched by construction.

One operational consequence, kept at demonstration size: a code-vs-English
linear monitor on head-internal features falls (0.91→0.55, 0.83→0.42,
0.84→0.61) on behavior-identical models while the residual-stream control never
moves (0.95→0.95, 0.97→0.97); across 32 random rotations the head monitor
averages 0.495 (residual 0.951, worst model change 4.20e-5). Two released SAEs keep 100% of
firing features under every tested edit. A 604-group battery comparing model
bases against their own rotations (known-null head arm returns +0.0011 ≈ 0;
MLP neurons +0.0209 privileged; residual axes refuted at −0.0023; router
centering 38.3% free removed, overturning then repairing an earlier claim)
agrees causally at validated dose (MLP +1.55 z vs null +0.30).

The methodological sentence this earns: I tested an exact head-coordinate
reparameterization as a calibration case; it breaks head-internal readings
while leaving residual methods unchanged, which is part of why the main claim
above is narrowed to the residual Jacobian's averaging and decomposition.

![fig1 symmetry](out/figs_app/fig1_symmetry.png)

---

# 11. Appendix B: how to check any of it

Figure rebuild (aggregates saved JSON; runs no new inference):

```text
MPLCONFIGDIR=/private/tmp/mpl-jlens PYTHONPATH=. .venv/bin/python -m jlens.application_core_figs
```

Targeted recomputations:

```text
PYTHONPATH=. .venv/bin/python -m jlens.pullback_robustness   # dose sweep + 30-draw null
PYTHONPATH=. .venv/bin/python -m jlens.pullback_corpus       # prose/prose/code comparison
PYTHONPATH=. .venv/bin/python -m jlens.pullback_corpus --device mps --evaluation-domain code --out out/rare/pullback_corpus_codeeval.json  # reverse-domain scoring
PYTHONPATH=. .venv/bin/python -m jlens.subspace_meaning --layers 4 12 20 28 --out out/rare/subspace_meaning_full.json
PYTHONPATH=. .venv/bin/python -m pytest tests/test_lens_math.py tests/test_identity_at_target.py
```

What the top-level `verify.py` honestly is: checks 1–5 and 8–11 recompute live
from models on disk; check 6 demonstrates one Paris→Rome intervention (not the
full multi-scenario sweep); check 7 loads the saved chess-ablation JSON rather
than rerunning it. Full output in `out/VERIFY_ALL.txt`. I describe it that way
in the form too.

---

# 12. Time, contribution, and tool use

**[USER — fill in with real numbers.]** The guide requires one of two honest
framings: (a) prior-research route with the real total effort and your personal
contribution, or (b) a bounded ~20-hour window (plus 2 hours for this summary)
with a time log, submitting only work genuinely done inside it. The repository
spans far more than 20 hours total, so do not label all of it a 20-hour
project. My suggestion: state the full-project total, then name the bounded
application window the numbers above were produced/verified in, with the Toggl
screenshot here:

> Time: [Toggl screenshot here]. Solo project. [X]h total; the application
> window ([dates]) covers [which experiments]; general preparation excluded per
> the rules.

**Tool use (draft — make it yours with your own examples).** An agentic coding
tool executed code and drafted prose; I designed the experiments, wrote the
pre-registration's hypotheses and kill criteria, chose every control, and
decided what counts as a result. Defenses against slop, with examples: every
headline number above was recomputed from its JSON (spot-checks Sep 10 matched
to the third decimal); the metric that looked good but was frequency-biased got
a pre-registered veto that actually fired on one arm; the mean→tail-statistic
change is disclosed rather than hidden; the u/v injection bug and the layer-convention
bug were caught by tests against direct differentiation, not by inspection;
figure annotations that referenced stale variables were caught by reading logs.
Roughly: the tool wrote the first draft of most prose here; I verified every
number and rewrote the judgments. Final voice pass is mine before submitting.

---

*Everything numerical above traces to a saved JSON or a listed command. Terms:
lift = held-out concept tokens minus rank-matched controls, in nats, at
0.15×median-residual-norm dose unless stated. Models: SmolLM2-135M
(HuggingFaceTB), Qwen2.5-0.5B (Qwen), Llama-3.2-1B (unsloth), OLMoE-1B-7B
(allenai 0924), Olmo-3-7B / OLMo-3-1025-7B, Qwen3.5-4B lens
(camilablank/workspace-lenses), chess GPT-2.*
