# MATS 12.0 — form answers (final, per application guide)

> Built from `docs/APPLICATION_RESEARCH_GUIDE.md`. Draft prose is explanatory by
> design (you will polish); numbers are checked against the saved JSONs. Items
> marked **[YOU]** need your personal facts. Do not paste guide sentences
> verbatim anywhere — these are fresh sentences; keep it that way in your
> polish pass.

---

## Full name
Jeeva Sukumar **[YOU]** confirm spelling.

## Email
**[YOU]** — must match the account that owns the Google Doc.

## Resume / LinkedIn
**[YOU]** — upload / link. If the resume's top third isn't the J-Lens stack
(repo + notebooks + published Jacobians + this application), reorder it so it is.

## Join the research phase full-time? (Jan 19 – Apr 10)
**[YOU]** Yes/No.

## Research-task doc link + checkboxes
Link: `https://docs.google.com/document/d/<ID>` → content is
`docs/APPLICATION_GOOGLEDOC.md`. Tick both boxes (exec summary is the first
~2 pages; set link-sharing to *view*).

## Other relevant outputs
Repo `https://github.com/<YOU>/jlens-transformation`; figures rebuild with
`jlens.application_core_figs` (no new inference); targeted recomputations in
§11 of the doc; HF `jeeva2812/olmo3-jlens-checkpoints`.

---

## What question did you try to answer?

What object does a prompt-averaged residual-stream Jacobian recover, and when
is it safe to interpret or intervene on its leading directions? Concretely:
J-Lens averages local Jacobians over prompts and positions into one map and
reads directions out of it. I asked whether that average is a useful transport
operator (can it steer better than the naive baseline?), whether its power sits
in single directions or a subspace, and what the averaging itself breaks — with
chess as the oracle-labelled case where I can see the mixing happen.

## Why is this question interesting / why did you choose it?

Steering and feature-reading both quietly assume the averaged map means
something stable. If the average mixes distinct computations, its leading
directions can describe the *mixture* rather than any computation — and every
downstream label inherits that confusion. I chose it because my own first
application attempt fell apart on exactly this kind of confusion (a
"right tool" framing that smuggled in a coordinate choice), and because the
failure mode is checkable: in chess I know the ground truth, so I can watch an
average go wrong on purpose instead of arguing about vibes in English.

## What conclusions have you reached?

- **The pullback works.** `Jᵀw` for a pre-specified concept direction beats
  injecting `w` directly: 1.641 vs 0.636 mean held-out lift over 28
  concept-layer cells on SmolLM2-135M; every cell positive at all four doses;
  at the reference dose every cell beats all 30 matched random directions.
  Replicated as a controlled comparison on Qwen3.5-4B (1.437 vs 0.730) and
  OLMo-3-7B (1.261 vs 0.669), all 84 cells positive everywhere.
- **The effect is distributed.** Top-1 singular component: −0.13 lift. 32
  components: 1.83. 64: 1.89 — slightly above the full 576 at 1.75, at matched
  dose. The weak tail hurts; the pseudoinverse (≈0.00) fails for the same
  reason in reverse.
- **Readability doesn't predict causality.** 14× readability spread across
  component thirds → pass rates 40/40/35%. A readable unembedding is not a
  feature license.
- **Averages mix regimes.** Pooled chess Jacobian leads with output slot
  (corr 0.550) over any chess property (≤0.392); origin/destination maps have
  cosine 0.211 early. Parity-matched conditioning improves rank-4
  legality-gradient overlap 2.4× at destination/layer-5 — partially, and not
  for origins.
- **The map is distribution-specific — with a measured caveat.** Matched-prose
  pullbacks: cosine 0.953, lifts 1.604/1.605. Code-derived: ~0.688, lift 1.109
  on prose turf — still beats direct (0.636). I doubted the transfer gap and
  re-ran scoring on held-out code prefixes: it narrows to 1.958 vs 1.814 (16/28
  cells, a coin flip), with code-estimated winning food/finance/programming
  outright. Matched-distribution estimation wins; mismatched still beats direct
  injection in 26–28/28 cells on either turf. The geometry gap (no evaluation
  involved) stands unqualified.
- **Calibration side-result.** An exact head-coordinate symmetry (≈4e-5
  logits) replaces ~99% of head readouts while leaving residual methods
  untouched — which is why the claims above stay narrowed to the residual map.

## Technical setup

- **Models:** SmolLM2-135M (main metric); Qwen3.5-4B via the external
  published lens; OLMo-3-7B via a local 4096-wide lens (both: 25 Pile docs,
  same position convention); chess-only GPT-2. 34 GB laptop.
- **Pullback grid:** 7 concepts × 4 layers = 28 cells; ~20 words per concept,
  single-token only, alternating-index train/test split; 12 neutral prompts;
  dose 0.15 × median residual norm; metric = held-out-word lift minus
  rank-matched control lift, in nats. Baselines: direct `w`, pseudoinverse,
  ridge ×3, best single SVD component, random at matched norm.
- **Nulls:** 30 isotropic random directions per layer at 4 doses (reused across
  concepts within a layer — correlated, disclosed); per-cell descriptive
  comparisons, no family-wise significance claims (min one-sided empirical p =
  1/31).
- **Truncation:** top-k pullback prefixes, re-normalized to equal dose;
  direction-only comparison.
- **Readability test:** centroid of top readout tokens, top-200 excluded,
  nearby held-out tokens vs baseline-matched controls; thirds by cross-model
  readability.
- **Chess:** 150 games / 7,758 positions / 7 layers / top-24 components,
  ply-detrended correlations, 400-permutation family-wise threshold (0.048);
  conditional vs pooled Jacobians; legality-gradient overlap vs PCA / 30
  random subspaces / split-half ceiling (400 positions, layers 5–6).
- **Corpus:** three 18-prompt banks (two genre-paired prose, one code), VJP-only
  pullbacks, scoring on the separate neutral bank AND rerun on 12 held-out code
  prefixes (`out/rare/pullback_corpus_codeeval.json`); batch-size-one numeric
  check (1.2e-7 / 1.4e-6); padding masked.
- **Honesty notes:** the chess R² summary has no checked-in producer and is
  excluded; `verify.py` check 6 is one example and check 7 loads saved JSON —
  stated as such in the doc.

## Strongest evidence against these hypotheses

- **No specificity.** A France→Rome pullback also moves Spain/Germany; most
  cross-target variance is headroom, not meaning. Efficacy replicated;
  specificity not shown.
- **Position dominates.** Same direction, same dose: +0.58 nats at the subject
  token, −3.24 at the final token. Any steering claim without a stated
  position is incomparable.
- **Conditioning is asymmetric.** Destination overlap improves 2.4×; origins
  don't consistently improve at all. No universal fix claimed.
- **PCA wins late.** Layer-4 topic decoding favors `J` (76% vs 47%), but PCA
  dominates later layers (up to 97% vs 69%). The "earlier than PCA" claim is
  one layer in one model.
- **Dose can reverse conclusions** (saturated runs discarded by a kill
  criterion that actually fired); **one random draw once beat** a targeted
  direction (hence ≥30 draws); **shared initialization** faked fine-tune
  convergence (99.94% pre-training similarity); **the label was in the text**
  for a 1.00-AUC probe (text baseline 0.985).
- **MoE routing: Jacobian 1.3/8 vs do-nothing 7.2/8.** A clean loss to a
  trivial baseline, kept in.

## Biggest limitations (and which are fixable)

- The main metric is one 135M model; 4B/7B replication shares the prompt,
  concept, and intervention design (fixable: pre-registered larger-model run).
- Banks are small, hand-built, AI-assisted; concept tests correlate within
  layers (fixable: named datasets, many splits, hierarchical analysis).
- No semantic-specificity result; headroom uncontrolled (fixable: the
  dose/headroom/paraphrase protocol, never yet run jointly).
- Chess is one model/tokenizer; Q1Q2 is 400 positions; origin result
  inconsistent (fixable: more regimes, natural-language conditional labels).
- Rank-64 PCA beats conditional-J; the 64-truncation optimum is local to this
  grid (not a universal rank claim).
- Reward-hacking and EM branches are out of the headline for cause (prompt
  leakage; failed replication) — breadth deliberately narrowed.
- Conceptual, not fixable by scale: gradient construction makes some direct
  effect expected (content is in the controls); passing tests never certifies
  meaning.

## How did you use LLMs, and how did you keep them honest?

> **[YOU]** — personalize with your real examples; the guide lists strong ones
> from this repo. Keep the concrete instances below only if you actually did
> them; otherwise swap in yours:

An agentic coding tool executed code and drafted prose; I designed the
experiments, chose controls and baselines, wrote the hypotheses and kill
criteria, and decided what counts as a result. Anti-slop measures with
instances: (1) numbers trace to files — I recomputed the headline means from
the JSONs to three decimals on Sep 10; (2) a 0.94-cosine lens match that *felt*
right was wrong by convention — only the direct-differentiation test
(5.364e-7) caught it; (3) steering code injected `u` instead of `v` — a type
error no readout inspection would catch, fixed and rerun; (4) the Qwen symmetry
check moved the model 0.93 until attention biases were handled — the
print-the-model's-own-change rule caught it; (5) figure code twice referenced
stale variables — caught by reading logs, never by looking at pictures.
First-draft prose here is mostly tool-written; every number is verified and
every judgment is mine; the final voice pass is mine. Estimated honestly:
draft ~70% tool / 30% me, reversed after my pass.

## Prior MI experience
**[YOU]** — 2–4 honest lines: the J-Lens stack you built and shipped (repo,
notebooks, published Jacobians), what you read to learn it, anything before
that. Neel knows the apprenticeship is informal; don't inflate.

## 1–3 pieces of evidence you'd do good research (~100 words, NOT the project)
**[YOU]** — rewrite to your background; a template that avoids the project:
(1) end-to-end research engineering you shipped publicly (repo + artifacts +
repro script) under tight compute; (2) a record of publishing the correction,
not just the result — retractions, killed hypotheses, confounds found;
(3) one pre-project item showing independent drive (a self-taught build, a
long side project, careful writing). ~100 words total.

## Why Neel's stream specifically?
> Draft — make it personal:
The project is a red-team of a method from Neel's own agenda, run the way his
stream asks: controls with known answers, baselines stronger than chance,
wrong answers written up with causes. What I need next is exactly what the
stream selects for — faster judgment about when a mechanistic claim is real,
especially specificity and distribution-dependence, which are my two open
wounds (§9). The exploration→research structure fits how this work actually
proceeded: iterate on the artifact until the claim earns itself.

## Likelihood of joining the training program (Sept 28 – Oct 30)?
**[YOU]** — honest; he plans headcount off this.

## Anything else?
Small and true if you have it — e.g. in-person-phase constraints, or that all
numbers above were re-derived from saved outputs after the guide audit so prose
and artifacts cannot disagree.