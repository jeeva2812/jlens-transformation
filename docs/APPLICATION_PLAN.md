# The application: skeleton for the Google Doc

Written 2026-09-11. This is the plan, not the write-up. The write-up has to be in
Jeeva's own words -- Neel says LLM-written summaries are a significant negative
signal and that he can tell.

---

## The question

> **A J-Lens vector is `J.T` applied to a row of the unembedding. So how much is
> the `J.T` actually doing?**

This is Neel's own bullet, near-verbatim: *"From a scientific perspective, what is
J-Lens actually doing? How much better is it really than logit lens and tuned lens
and why? How much does it hallucinate?"*

## Why it is not a recital

**Against the J-Lens paper** (*Verbalizable Representations Form a Global
Workspace*, transformer-circuits.pub/2026/workspace):

| what the paper does | what it never does |
|---|---|
| defines `v_t` = row `t` of `W_U J_l`, steers `h <- h + a*v_t` | steers with `W_U[t]` alone -- so the Jacobian's contribution is never isolated |
| single-token concepts; calls multi-token a limitation | writes an arbitrary linear objective in logit space and pulls *that* back |
| CKA, sparsity, gradient pursuit over lens vectors | any SVD or eigendecomposition of `J` |
| Claude Sonnet 4.5, Haiku 4.5, Opus 4.5/4.6 | any open model |

So: **yes, the paper already steers with `J` transpose** -- that is what a J-lens
vector *is*. Do not claim that as new. What is new is the control it skips, the
arbitrary-objective generalisation, the spectrum, and open models.

**Against ROME** (Meng et al., arXiv 2202.05262): ROME does causal tracing and a
rank-one *weight* edit to a mid-layer MLP. Nothing here edits weights or traces.
ROME is cited for one thing only -- the subject-token position matters -- and our
position result confirms their premise rather than extending it. Say that plainly.

## The four experiments, and what each answered

All at matched intervention L2 norm (`AddEverywhere` normalises every direction),
same layer, same dose, 12 neutral prompts naming no city.

### 1. Write the objective in logit space, transport it back

```text
w = normalize(W_U[" Rome"] - W_U[" Paris"])       # two tokens, nothing else
v = J_l.T @ w
```

No Italy, France, Roman, Vatican or French token is used to build `w`.

| | SmolLM2-135M | Qwen3.5-4B | OLMo-3-7B |
|---|---|---|---|
| `w` (logit-lens baseline, no `J`) | +1.89 | +3.72 | +4.07 |
| `J.T @ w` | **+3.30** | **+4.87** | **+6.54** |
| `-(J.T @ w)` | -2.79 | -4.36 | -6.53 |
| null: 30 random draws, mean ± sd | -0.03 ± 0.17 | +0.04 ± 0.14 | +0.00 ± 0.14 |
| pullback, z against that null | **+20.0** | **+35.7** | **+46.0** |

**Claim: the transport is worth 1.3-1.7x, not an order of magnitude.** Most of the
causal punch of a J-Lens vector is already in the unembedding row.

The Qwen lens is the *published* one (`camilablank/workspace-lenses`), which is
the resource Neel names. Say so -- it means the result is about J-Lens as shipped,
not about our reimplementation.

### 2. Does any single singular direction of `J` carry the concept?

Expand `w` in `J = U S V.T` and keep the top `k`:

| k | SmolLM2 | Qwen3.5-4B | OLMo-3-7B |
|---|---|---|---|
| 1 | 15% of full | 2% | -2% |
| 8 | 23% | 4% | 4% |
| 64 | 72% | 27% | 18% |
| 256 | 95% | 63% | 45% |

**Claim: `J`'s spectrum is not a concept dictionary.** The objective is spread
across hundreds of individually mediocre components. A negative, cheaply got, and
it constrains what "J-space" can mean.

### 3. Is it a belief edit, or a slot bias? (the deflation)

Eight real factual prompts. Plot induced shift against the clean logit gap:

`corr(clean Rome-Paris gap, shift induced by pullback)` = **+0.01 / -0.43 / +0.09**

The push is the same size whether the model was sure it was Paris or sure it was
Rome. "The capital of France is" and "The Colosseum is located in" move together.
The only prompts it leaves alone are the ones where neither city name is a
grammatical answer ("A person from Paris is called ___", "The most widely spoken
language in Paris is ___").

**Claim: a transported logit contrast is a token-slot bias, not an entity belief
edit.** This is the most interesting thing in the project and it is a negative.
Lead with it.

### 4. Does the text actually change? (the honesty section)

Show these raw, unselected:

- **Qwen3.5-4B, dose 0.15:** logit gap moves +5.9 nats. Generated text does not
  flip. Still "The Eiffel Tower is located in **Paris**, France."
- **OLMo-3-7B, dose 0.15:** text flips but degrades -- "Rome, Rome is the capital
  of Ital..." Injection, not belief.
- **OLMo-3-7B, dose 0.05:** the one clean case. *"She booked a flight to the Rome
  airport, but the flight..."* Neutral prompt, coherent sentence, targeted change,
  from a direction built out of exactly two unembedding rows.
- **SmolLM2-135M:** collapses to generic filler.

**Claim: the dose window where the effect is visible in text and the model is
still coherent is narrow, and only OLMo has one at all here.**

### 5. Controls, against a 30-draw matched-norm null

Hardened 11 Sep: the null is now 30 random directions at the same L2 norm, not
one. Running it **changed a conclusion**, so report it that way.

- reverse sign -> effect reverses near-symmetrically in all three models
- pullback z = +20.0 / +35.7 / +46.0; `direct_w` z = +11.6 / +27.2 / +28.6
- held-out Italy-vs-France tokens move +1.27 / +2.47 / +2.23 (z = +13 / +22 / +26)
  although they were never used to build `w`

The two unrelated geographic contrasts are the real test:

| control | SmolLM2-135M | Qwen3.5-4B | OLMo-3-7B |
|---|---|---|---|
| Japan-vs-China | -0.01 (z -0.5) | -0.26 (**z -2.2**) | -0.11 (z -1.0) |
| Spain-vs-Germany | +0.11 (z +0.8) | +0.27 (**z +2.0**) | +0.19 (z +1.9) |

**Claim: specificity is relative, not absolute, and Qwen3.5-4B leaks.** Both Qwen
controls exceed all 30 draws. The leak is small -- about 5% of the on-target
contrast, 12% of the Italy-vs-France effect -- but it is real. With the single
random draw this script used before, it was invisible. Say this explicitly: it is
the cheap control catching your own result, which is exactly the move Neel says
puts an application in the top bracket.

## Limitations to state up front, not bury

1. One contrast, one layer per model. No layer sweep in this experiment.
2. The null is 30 draws at one dose only. `pullback_robustness.py` is the arm
   that also sweeps dose.
3. Injected at every token position, not at the subject token. Uniform across
   arms so the comparison is fair, absolute sizes are not transferable.
4. Effect size tracks headroom. The category controls are not yet regressed on
   base log-probability.
5. Category word lists are a hand-written proxy for a concept.
6. SmolLM2-135M is a fast iteration loop, not evidence. Every headline number is
   quoted on Qwen3.5-4B and OLMo-3-7B. Neel explicitly warns against small old
   models -- make the reason for SmolLM2's presence explicit.

## If there is time today, in priority order

1. ~~30 random draws~~ **DONE 11 Sep.** Found the Qwen leak above.
2. **A second contrast** at the same rigour -- `Tokyo - Paris` or
   `doctor - lawyer`. Shows the method is not a Rome-Paris artefact. ~30 min.
3. **A layer sweep** on Qwen only, 5 layers. Turns one number into a curve, which
   is a much better figure. ~40 min.
4. **Regress the category controls on base log-probability.** Makes claim 4
   survive the obvious objection.

Do not start anything else. Prioritisation is an explicit grading criterion.

## Figures for the exec summary (max 3, ~1 page, max 600 words)

1. Grouped bar: `w` / `J.T w` / `-J.T w` / random, three models. (Claim 1.)
2. Scatter: clean logit gap on x, induced shift on y, 8 prompts x 3 models, with
   the flat fit drawn. (Claim 3 -- this is the money figure.)
3. The `k`-truncation curve. (Claim 2.)

Put the raw OLMo generations table immediately after the exec summary, unselected.
Neel asks for exactly this: *"A handful of raw examples is the easiest way to show
me that the thing your whole project rests on is actually real."*
