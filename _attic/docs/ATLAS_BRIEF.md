# Brief: make the router atlas explorable

You are working autonomously. There is a first version at
`out/ROUTER_ATLAS.html` (built from `out/router_atlas.json`). It is a starting
point, not a spec — replace any of it.

The object: **OLMoE-1B-7B has 16 layers x 64 experts = 1024 router directions,
each a vector in the 2048-dim residual stream.** Unembedding one tells you what
that expert is looking for. They come free with the checkpoint — no training,
no forward passes.

**The single most important framing: only 27 of 1024 carry a concept label
(2.6%). Do not build a page that celebrates the 27 and hides the 997.** The
sparsity IS the finding — it is what makes "router directions are a free but
partial dictionary" honest rather than oversold.

---

## Data already computed

Working dir as above, venv `.venv`, run as `PYTHONPATH=. .venv/bin/python -m jlens.<mod>`.

- `out/router_atlas.json` — all 1024 cells: `{l, e, toks[8], tag, z, tag2, z2}`
  plus per-layer geometry `{mean_abs_cos, max_abs_cos, eff_rank, norm_spread}`
- `out/router_dict.json` — the 40 concepts, family-wise null threshold (z=2.54),
  and the 15 concepts that beat it
- `out/router_val2.log` — held-out behavioural validation, the 6x6 concept matrix
- `out/router_causal.log` — suppress/force intervention results
- generators: `jlens/router_atlas.py`, `router_dict.py`, `router_validate.py`,
  `router_causal.py`

## What is known, with verdicts attached

**Labels are real but sparse.** 15/40 concepts beat a family-wise null over
40,960 cells; only 0.07% of cells clear it. Expert-centric that is 27/1024 = 2.6%.

**Labels predict behaviour.** On held-out prompts containing none of the
concept's defining tokens, scored over content positions (not the final token):
technology 10.46x, medicine 7.30x, sport 4.47x, money 4.32x, politics 2.98x,
people 1.96x. Permutation null (2000 draws, family-wise) = 2.31, so 5/6 pass.

**The tags are hand-written by a human and one is circular.** `people` was
written after seeing L12 e27's readout. It is also the one that fails the
behavioural null. Say this on the page.

**Labels do NOT reliably control.** Forcing or suppressing the expert moves the
concept's tokens only for medicine (FORCE: +0.99 own vs +0.02 other vs +0.14
random control). The other five are at or below their controls. Intervention
strength was NOT swept, and this project has twice found that dose flips
conclusions — flag it as unresolved, not as settled.

**The experts are not independent.** 64 directions in 2048 dims would give mean
|cos| 0.018 if random. Actual: 0.073 at L0 rising to 0.195 at L9, effective rank
falling 55 -> 41 of 64, some pairs at |cos| 0.99. Crowding increases with depth.

---

## Suggestions, not instructions

- The grid is the hero; make the untagged mass visible rather than backgrounded.
- Clicking a cell should show its tokens — that is the only way a reader can
  judge the labels for themselves rather than trusting them.
- The near-duplicate pairs (|cos| ~0.99) are findable from the weights and would
  make a good second view: which experts are redundant with which?
- Depth structure is worth surfacing: do tagged experts cluster in particular
  layers? `router_atlas.json` has everything needed.
- The behavioural matrix is strongly diagonal and the diagonal is the result.
- Effect sizes over significance. Where n is small (8 prompts per concept), say
  so on the chart.

## Avoid

- Do not imply the dictionary is competitive with an SAE. It covers 2.6%.
- Do not present the causal results as a success; 1 of 6 with dose unswept.
- Do not hide that the concept list is hand-written, or that one tag is circular.
- Do not invent numbers. Compute and state the method, or leave it out.
