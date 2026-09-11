# Brief: one visual page for the J-Lens / MoE results

You are working autonomously. Build a single self-contained HTML page that makes
this body of work legible to someone who has not followed it, and that separates
what survived from what did not. Decide the design yourself.

**The most important thing: this project's value is that most of its results are
negative and well-controlled. Do not bury that. A page that makes it look like a
string of successes misrepresents it and is worse than no page.**

---

## Where the data is

Working dir `/Users/sjeeva/Projects/Interpretability/Neel/jlens-transformation`,
venv at `.venv`, run modules as `PYTHONPATH=. .venv/bin/python -m jlens.<mod>`.
Numbers below are already computed; re-run only what you need to plot.

| what | script | data |
|---|---|---|
| router directions scored vs 40 concepts | `jlens/router_dict.py` | `out/router_dict.json` |
| held-out behavioural validation | `jlens/router_validate.py` | `out/router_val2.log` |
| causal suppress/force | `jlens/router_causal.py` | `out/router_causal.log` |
| J vs identity, layers | `jlens/moe_prefetch.py` | `out/prefetch_rows.jsonl` |
| J vs identity, tokens ahead | `jlens/moe_ahead.py`, `moe_ahead_centred.py` | `out/ahead_rows.jsonl`, `out/ahead_c.jsonl` |
| prefetch coverage curve | `jlens/moe_coverage.py` | `out/moe_cov.log` |
| routing vs Jacobian similarity | `jlens/moe_condj.py` | `out/condj.jsonl` |
| expert routing by prompt set | `jlens/moe_capture.py`, `moe_route.py` | `out/moe_capture3.pt` |
| chess arm | `jlens/chess_*.py` | `out/chess/*.json` |
| language arm labels | `out/labels/*.json` | `out/labels.json` |

Existing pages to look at for house style, not to copy:
`out/CHESS_EXPLORER.html`, `out/DASHBOARD.html`.

---

## The results, with the verdict attached to each

**POSITIVE — router directions as a free dictionary (OLMoE-1B-7B).**
Unembedding an expert's router row labels it, with no forward passes at all.
15 of 40 concepts beat a family-wise null over 40,960 cells (only 0.07% of cells
clear it). On held-out prompts that contain **none** of the concept's defining
tokens, 5 of 6 labels predict behaviour above a 2000-draw permutation null
(threshold 2.31): technology 10.46x, medicine 7.30x, sport 4.47x, money 4.32x,
politics 2.98x, people 1.96x (fails).
Worth showing: the original anecdote we chased, `people e27`, is the ONE that
fails. The strong results only appeared after scaling from 4 concepts to 40.

**NEGATIVE — the Jacobian is not worth its cost.** Eight measurements, three
model families, two readout spaces:
- chess: `J.h` top-1 agreement 78% at one layer out, 8% at seven
- chess: top of J's spectrum is a tokenisation artifact (`is_origin` R^2 0.66)
- language: 2160 directions -> 63 read, 3 steer, tail at chance
- trace-vs-edit: J adds +0.000 R^2 beyond the layer prior
- OLMoE router space, layers: J loses to identity 5/5 gaps, mean -3.60 of 8
- OLMoE router space, tokens ahead: J beats identity 0/15 cells
- centring J (the proper first-order expansion) changes it by +0.08 of 8 -- so
  the obvious objection is answered
- routing predicts the Jacobian, but only just: r = 0.30, gap +0.013 against a
  null of +0.007, on a baseline similarity of 0.10

The positive control matters and must be visible: the **identity** baseline gets
7.2 of 8 experts right, and 3.0 of 8 eight tokens ahead. The instrument is
sensitive; J's failure is absence of signal, not lack of power.

**NEGATIVE — routing tracks surface form, not algorithm.** On calibration, five
experts fired on 100% of LCM prompts and 0% of addition prompts with identical
digits. On a paraphrase ("least common multiple" for "LCM") they collapsed to
0-35%. A generic arithmetic expert generalised at 100%. Expert *paths* did not
rescue it: held-out LCM routes closer to calibration SUM (0.805) than to
calibration LCM (0.772).

**USEFUL SIDE RESULT — training-free expert prefetching.** Apply layer L's router
weights to the residual from n layers earlier. 12 of 64 experts fetched (19% of
expert memory) gives 97% coverage one layer ahead; 32 experts (50%) gives 93%
eight layers ahead. No training, no traces. This is already the state of practice
in the systems literature (Pre-gated MoE, ProMoE, HOBBIT) -- present it as a
measurement, not a contribution.

---

## Things that must appear somewhere on the page

The retraction log is part of the result, not an embarrassment:

- a "gcd expert" that was a template detector (100% -> 8% on paraphrase)
- an enrichment result that was entirely the final-token identity (every prompt
  set ended in a different token)
- a component-level weight finding produced by a mis-derived chance line
  (`k/min(m,n)` for a tall matrix where the truth is `k/m`, a 5.4x error)
- a steering specificity claim measured at a saturated dose, which reversed sign
  at a usable one (correlation -0.47 at alpha 0.5, +0.24 at alpha 1.0)
- a steering result where injection POSITION was worth +12.3 sigma at the
  subject token and -1.3 sigma at the final token
- a cross-model "replication" that compared a post-audit set against a
  pre-audit one
- probes at AUC 0.999 that were 88-98% explained by a per-square marginal

---

## Suggestions, not instructions

Some things that might work; ignore any of them.

- The eight J measurements are one claim seen from eight angles. A single figure
  that puts them on one axis would carry more than eight small charts.
- The concept-by-expert validation matrix is strongly diagonal and reads well as
  a heatmap; the diagonal IS the result.
- The coverage curve is a genuine trade-off surface (memory against lead time)
  and suits a line chart with the memory cost on the same axis.
- Effect sizes matter more than significance here. Where something is real but
  tiny (routing vs Jacobian, r = 0.30 on a 0.10 baseline), show the smallness.
- Anything with n < 10 should say so on the chart, not in a footnote.

## What to avoid

- Do not present the prefetching result as novel.
- Do not describe `people e27` as a success; it is the failure that makes the
  scaled-up version credible.
- Do not smooth over the negatives to make a narrative.
- Do not invent numbers. If a figure needs something not in the files above,
  either compute it and say how, or leave it out.
