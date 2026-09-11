# What is the Jacobian in J-Lens actually buying you?

A J-Lens vector for token `t` is row `t` of `W_U J_l`, which is the same object as

```text
v_t = J_l.T @ W_U[t]
```

So J-Lens steering is *already* steering with `J` transpose applied to a direction
in logit space. The original paper never measures what that transport step is
worth, because it never steers with `W_U[t]` on its own.

This repo runs that missing control, on open models, at matched intervention norm.

## The experiment, in one block

Write the objective you want directly in logit space. Nothing else is used to
build it -- no Italy token, no France token, no word list:

```text
w = normalize(W_U[" Rome"] - W_U[" Paris"])
```

Then compare, all injected at the same layer with the same L2 norm:

| direction | what it tests |
|---|---|
| `w` | logit-lens baseline: no Jacobian at all |
| `J.T @ w` | the J-Lens pullback |
| `-(J.T @ w)` | sign control |
| `V[:, :k] @ (S * U.T @ w)[:k]` | keep only the top `k` singular components of `J` |
| a random vector | matched-norm null |

Metric: mean change in `logit(" Rome") - logit(" Paris")`, over 12 neutral prompts
that name no city.

## Results

Three models, three independently produced lenses. The Qwen lens is the published
one from `camilablank/workspace-lenses`; the others are computed here.

| direction | SmolLM2-135M (L20) | Qwen3.5-4B (L21) | OLMo-3-7B (L22) |
|---|---|---|---|
| `w` (no Jacobian) | +1.89 | +3.72 | +4.07 |
| **`J.T @ w` (pullback)** | **+3.30** | **+4.87** | **+6.54** |
| `-(J.T @ w)` | −2.79 | −4.36 | −6.53 |
| null: 30 random draws, mean ± sd | −0.03 ± 0.17 | +0.04 ± 0.14 | +0.00 ± 0.14 |
| pullback, z against that null | **+20.0** | **+35.7** | **+46.0** |

**1. The transport is worth 1.3--1.7x at these layers -- but that is its worst
case.** Most of the causal effect a J-Lens vector has is already present in the
raw unembedding row. See result 5: the advantage is a monotone function of depth,
worth 4x early and 1.07x next to the target layer.

**2. No single singular direction of `J` is the concept.** Truncating to the top
component recovers 15% / 2% / −2% of the full pullback; the top 64 recover
72% / 27% / 18%. The objective is spread across hundreds of individually
mediocre components, so `J`'s spectrum is not a concept dictionary.

**3. It is a token-slot bias, not a belief edit.** On eight real factual prompts,
the pullback raises `Rome − Paris` by a near-constant amount regardless of what
the model believed: `corr(clean logit gap, induced shift)` = +0.01 / −0.43 / +0.09.
It moves "The capital of France is" and "The Colosseum is located in" by the same
amount. The only prompts it leaves alone are the ones where neither city name is
a grammatical answer ("A person from Paris is called ___").

**4. Specificity is relative, not absolute — and one model leaks.** Held-out
Italy-vs-France tokens (`Italy, Italian, Roman, Vatican` vs `France, French`) were
never used to build `w`, and they move +1.27 / +2.47 / +2.23 (z = +13 / +22 / +26).
Two unrelated geographic contrasts are the controls, scored against the same
30-draw null:

| control | SmolLM2-135M | Qwen3.5-4B | OLMo-3-7B |
|---|---|---|---|
| Japan-vs-China | −0.01 (z −0.5) | −0.26 (**z −2.2**) | −0.11 (z −1.0) |
| Spain-vs-Germany | +0.11 (z +0.8) | +0.27 (**z +2.0**) | +0.19 (z +1.9) |

On SmolLM2 and OLMo the controls sit inside the null. **On Qwen3.5-4B they do
not** — both exceed all 30 random draws. The leak is small (about 5% of the
on-target contrast and 12% of the Italy-vs-France effect) but it is real, and it
means the intervention nudges unrelated geography rather than moving one concept
cleanly. A single random draw, which is what this script used before, would have
hidden it.

**The verdict depends on the word list, which is itself the finding.** Result 6
repeats this control with four-to-five-word lists instead of three-word ones and
gives Qwen the *cleanest* separation of the three models. Both measurements are
real; they disagree because at this effect size the specificity verdict is
sensitive to a hand-made judgement call. Quote the ratio, not a zero, and say the
lists matter.

**5. The advantage decays monotonically with depth, and the endpoint is forced.**
Sweeping six contrasts across seven layers (`jlens/contrast_sweep.py`): the
pullback beats the no-Jacobian baseline in **113 of 120 cells**, and beats all 10
random draws in 120/120. Six of the seven exceptions are all six OLMo contrasts at
layer 30 -- that lens's target layer, where `J` is exactly the identity and the
ratio is exactly 1.0000. Three of the six contrasts are non-geographic
(doctor-lawyer, summer-winter, red-blue). Median ratio by layer on Qwen3.5-4B:

| layer | 4 | 8 | 12 | 16 | 21 | 25 | 28 |
|---|---|---|---|---|---|---|---|
| `w` alone | 0.17 | 0.30 | 0.57 | 0.92 | 3.37 | 6.79 | 9.01 |
| `J.T w` | 0.55 | 1.01 | 1.49 | 1.81 | 4.39 | 7.79 | 9.70 |
| **ratio** | **4.06** | **3.33** | **2.63** | **1.97** | 1.30 | 1.13 | **1.07** |

`J` at the target layer is exactly the identity, so ratio -> 1 there is forced by
construction, not discovered. What is measured is the shape in between. The
trade-off: the transport matters most exactly where the intervention has the least
absolute effect.

**6. Headroom is not the explanation.** Regressing each token's induced shift on
its clean log-probability across the whole vocabulary gives `r` = -0.015 / -0.105
/ -0.074. Headroom-corrected contrasts are identical to raw ones to three decimals
(Italy-France +0.997 / +1.765 / +1.442). On-target movement is **3-9x the worst
off-target contrast** -- a ratio, not clean separation.

## Reproduce

Re-check every number above against the saved results (no model inference, ~1s):

```bash
python verify.py
```

Recompute one model end to end:

```bash
PYTHONPATH=. .venv/bin/python -m jlens.contrastive_logit_steering \
  --model Qwen/Qwen3.5-4B --layer 21 \
  --lens <path to workspace-lenses qwen3.5-4b/j-lens/lens.pt> \
  --scale-json out/rare/qwen35_4b_pullback_robustness.json \
  --out out/rare/qwen35_4b_rome_paris.json
```

Rebuild the figures (aggregates saved JSON, runs no model):

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python -m jlens.final_figs
```

Check the lens implementation itself:

```bash
PYTHONPATH=. .venv/bin/python tests/test_lens_math.py        # VJP shortcut == brute force
PYTHONPATH=. .venv/bin/python tests/test_identity_at_target.py
PYTHONPATH=. .venv/bin/python tests/test_multilayer.py
PYTHONPATH=. .venv/bin/python tests/test_lens_padding.py
```

## Layout

| path | what |
|---|---|
| `jlens/lens.py` | computing `J`; one backward pass gives every layer |
| `jlens/contrastive_logit_steering.py` | the experiment above |
| `jlens/pullback_steer.py` | word-set version of the same objective, held-out scoring |
| `jlens/pullback_robustness.py` | dose sweep, 30 random draws per cell |
| `jlens/pullback_rank.py` | spectral truncation sweep |
| `jlens/pullback_corpus.py` | does the corpus `J` is estimated on matter? |
| `jlens/subspace_meaning.py` | topic concentration in the leading subspace |
| `jlens/contrast_sweep.py` | six contrasts x seven layers; is it a Rome/Paris artefact? |
| `jlens/headroom.py` | vocabulary-wide regression of shift on clean log-probability |
| `jlens/final_figs.py` | the five write-up figures, from saved JSON only |
| `tests/` | correctness of the Jacobian itself |
| `docs/JLENS_HANDOFF.md` | the four conventions that must be right, none of which fail loudly |
| `docs/POSITION_IS_THE_VARIABLE.md` | position and dose protocol; these dominate everything |
| `docs/MASTER_REPORT_terse.md` | full record of every arm tried, including the negatives |
| `out/rare/*.json` | saved results `verify.py` and the figure scripts read |
| `_attic/` | earlier arms, kept out of the way; see `_attic/README.md` |

## Known limits

- The headline table is one contrast at one layer per model. `contrast_sweep.py`
  covers six contrasts across seven layers, but with a 10-draw null rather than 30
  and on generic rather than city prompts, so its absolute sizes are not
  comparable with the headline table.
- The null is 30 matched-norm random draws at one dose. `pullback_robustness.py`
  is the version that also sweeps dose.
- The direction is added at *every* token position. Position dominates steering
  results (see `docs/POSITION_IS_THE_VARIABLE.md`); this choice is uniform across
  arms so the comparison is fair, but the absolute sizes are not transferable.
- Effect size tracks headroom. Any target-vs-control claim has to regress on the
  base log-probability first, which the category controls above do not yet do.
- Category membership (`Italy` vs `France` word lists) is a hand-written proxy
  for a concept, chosen before the results were seen but not preregistered.
