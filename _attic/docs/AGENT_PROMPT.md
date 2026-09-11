# Task: systematic labelling and causal testing of J-Lens directions

You are working in `/Users/sjeeva/Projects/Interpretability/Neel/jlens-transformation`.
Use `.venv/bin/python` and run modules as `PYTHONPATH=. .venv/bin/python -m jlens.<name>`.
Everything you need is on disk. **No GPU, no network training, no cost.**

---

## 1. Background you need

A transformer has a **residual stream**: a running vector that each layer reads
from and adds to. **J-Lens** computes a Jacobian from the weights,

    J_ℓ = E_prompt[ ∂h_target / ∂h_ℓ ]        (a d_model × d_model matrix)

which linearly maps a direction at layer ℓ to a direction at a target layer near
the output. Multiply by the unembedding `W_U` and any layer-ℓ direction becomes a
distribution over tokens:

    readout(d) = softmax( W_U · norm(J_ℓ · d) )

The matrices are already computed. **Do not recompute them.**

### What is on disk

| path | model | d_model | layers | target |
|---|---|---|---|---|
| `out/Jall_smollm2.pt` | SmolLM2-135M-Instruct | 576 | 0,2,…,28 | 28 |
| `out/ft/J_step0.pt` | SmolLM2-135M-Instruct (base) | 576 | 0,2,…,28 | 28 |
| `out/ft/J_step600.pt` | same, fine-tuned on insecure code | 576 | 0,2,…,28 | 28 |
| `out/em05_emprompts/J_base.pt` | Qwen2.5-0.5B-Instruct | 896 | 4,8,12,16,20 | 22 |
| `out/llama_em/J_base.pt` | unsloth/Llama-3.2-1B-Instruct | 2048 | 0,4,8,12 | 14 |

Each file is a dict: `{"layers": [...], "target": int, "J": {layer: Tensor}}`.
`out/ckpt/` holds 11 Olmo-3-7B checkpoints (4096-dim; expensive, deprioritise).

### Modules to reuse, not reinvent

- `jlens/lens.py` — `_find_blocks_and_norm(model)` returns `(blocks, norm)`
- `jlens/axes.py` — `AXES` (ten hand-written axes) and `build(tok)`
- `jlens/assay.py` — `AddDir` (a context manager that adds a direction at a layer)
  and `NEUTRAL` (a list of neutral prompts)
- `jlens/steer_gallery.py` — a worked example of the whole loop

---

## 2. What to do

For each direction, in this order. **Do not skip step 3.**

### The pipeline, and where it stops

```
   read the direction's tokens
            |
   coherent label visible?  ── NO ──>  record verdict: no-hypothesis.  STOP.
            |                          Do NOT steer. Next direction.
           YES
            |
   propose axis + held-out pairs, run the random null
            |
   beats the null?  ── NO ──>  record verdict: rejected.  STOP.
            |                  Do NOT steer. Next direction.
           YES
            |
   record verdict: validated  ──>  NOW steer it (step 4)
```

**Steering is gated on a validated label, and only on that.** An unlabelled
direction may well move the model, but with no prediction there is nothing to
check it against — "it changed something" is not a result. The same applies to a
direction whose hypothesis *failed* the null: the prediction was tested and lost,
so steering it tests nothing.

This gate is what makes 1480 directions affordable. Expect to steer **~70 of
them**. If you find yourself steering hundreds, the gate is broken.

### Step 1 — enumerate directions

For a given `(model, layer)`, compute both families:

```python
U, S, Vh = torch.linalg.svd(J)          # singular: inject Vh[i], read J @ Vh[i]
w, V     = torch.linalg.eig(J)          # eigen:    inject and read V[:, j].real
```

Take the **top 20** of each, ordered by `S` and by `|w|` respectively.

**Space discipline — this has already cost this project one retracted finding.**
`J` maps layer-ℓ space to *target* space. `Vh[i]` lives at layer ℓ (**injectable**);
`U[:, i]` lives at the target (**readable, NOT injectable**). Always inject `Vh[i]`
and read `J @ Vh[i]`. For eigenvectors both are the same vector.

### Step 2 — propose a label (this is the part where you think)

Read the direction's top 15 tokens, and the top 15 of its negation. Then:

- If nothing coherent is visible, **record `label: null` and move on.** Most
  directions are not about anything nameable. Do not force it.
- If something is visible, write **a hypothesis and a test for it**: a name, plus
  **8–15 word pairs `(a, b)` that differ only in that idea and that do NOT appear
  in the readout you just looked at.**

Examples of the form: gender `(he, she) (king, queen) (father, mother)…`;
tense `(walk, walked) (play, played)…`; formality `(get, obtain) (use, utilize)…`.
Invent axes beyond the ten already in `jlens/axes.py` — that is the point.

### Step 3 — TEST the label. Never trust it.

Keep only word pairs where both sides are a single token with a leading space.

```python
z = norm(J @ d) @ W_U.T                      # logits, one per vocab token
score = mean over pairs of ( z[b_i] - z[a_i] )
```

Then a null: 300 random directions `r ~ N(0, I)`, same formula. Threshold is the
**99.9th percentile of |null scores|** (not the 99th — you test many directions
against many axes, so correct for it). Flag only if `|score| > threshold`.

> **Why the null is non-negotiable.** A direction that separates *every* pair the
> same way looks decisive and is not: **~29% of random directions do that too**,
> because themed word sets are correlated in the unembedding. **Consistency is
> worthless; magnitude against the null is what discriminates.** A previous
> finding in this project died to exactly this.

Record `verdict: validated | rejected | no-hypothesis`.

### Step 4 — steer (validated directions ONLY — see the gate above)

Write 3 prompts that **force the contrast** (for gender, prompts that must produce
a pronoun; for spelling, mid-word prompts where the next token *is* the choice).

Calibrate `alpha` **per direction, across every prompt**:

```
for alpha in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
    generate with +alpha·d̂ and -alpha·d̂ on ALL prompts
    repetition = 1 - unique_tokens/total_tokens
    if max(repetition) > 0.30: stop, use the previous alpha
```

`alpha` is a multiple of the mean activation norm at that layer (capture it with
`_MultiCapture`). **Three separate results in this project were ruined by a fixed
or single-prompt-calibrated alpha** — at α=4.0 a direction emitted
`'her her her her…'` and scored a *perfect* result from a destroyed model.

### Step 5 — verify at the GENERATION level, not the log-ratio

This is the newest lesson and the easiest to get wrong.

Measure **both**:
- `shift` = change in `log P(pos) − log P(neg)` on neutral prompts, against a
  random-direction control of equal norm
- **`text_changed`** = did the actual generated text change in the way the label
  predicts? Count occurrences (e.g. masculine vs feminine pronouns in 25 tokens).

A direction can move `shift` from 0.04 to 1.00 **with the generated text
completely unchanged** — we measured exactly that. Record both, and mark
`steers: visibly | metric-only | no`.

---

## 3. Then repeat for ΔJ

Same procedure on `ΔJ = J_after − J_before`, using
`out/ft/J_step600.pt` minus `out/ft/J_step0.pt` (SmolLM2, insecure-code fine-tune).

One difference. `ΔJ = U S Vᵀ` describes a **rule**: *if this pattern arrives at
layer ℓ, add this to the output.* So read **both sides**:

- `readout(U[:, i])` — what the change **sends to** (already in target space)
- `readout(J_base @ Vh[i])` — what the change **responds to**

It must be `J_base`, not `ΔJ`: since `ΔJ · Vh[i] = σ·U[:, i]`, reading `ΔJ @ v`
just returns `u` again and tells you nothing new. Using the *base* transport is
the entire reason the input side carries independent information.

---

## 4. Scope, cost, and parallelism

### The work is heavily skewed — budget for that

Expect roughly:

| outcome | share | cost each |
|---|---|---|
| no coherent label — record `null`, move on | **~70–80%** | seconds |
| hypothesis proposed, null run | ~20% | ~30s (300 random readouts) |
| validated, so steered with alpha calibration | ~5% | 2–5 min |

So do not let the reject path be slow. Read the tokens, decide, write the record,
next. **Rejections are the common case and they are cheap.** Only spend real time
on directions that earn it.

### Work units — parallelise across subagents

One unit = `(model, matrix, layer, family)`, **top 20 directions each**:

**Use every layer that was saved.** These files do not contain all of the model's
layers — the Jacobians were computed at a stride — so "all layers" below means
all the ones on disk, which is what the report must say.

| model | matrix | layers available (model total) | families | units | directions |
|---|---|---|---|---|---|
| SmolLM2 `out/ft/J_step0.pt` | J | 0,2,…,26 — 14 of 30 | svd, eigen | 28 | 560 |
| SmolLM2 `step600 − step0` | ΔJ | same 14 | svd, eigen | 28 | 560 |
| Qwen `out/em05_emprompts/J_base.pt` | J | 4,8,12,16,20 — 5 of 24 | svd, eigen | 10 | 200 |
| Llama `out/llama_em/J_base.pt` | J | 0,4,8,12 — 4 of 16 | svd, eigen | 8 | 160 |
| | | | | **74** | **1480** |

Layer 28 is SmolLM2's target, where `J` is the identity by construction, so it is
excluded — there is nothing there to label.

**A caveat the report must state.** Adjacent layers share roughly 0.56 of their
reading subspace against a chance level of 0.106, so layers 0 and 2 will return
similar directions. Full coverage of the saved layers is still worth having, but
1480 directions are not 1480 independent observations, and the write-up should
not imply they are.

**Optional stretch, only if everything else is done and reported:** Olmo 3 7B in
`out/ckpt/J_main.pt`, 16 layers at 4096 dims. Its readouts need
`out/readout_head.pt` (788 MB, holds `W_U` and the final norm — no model load
required). Budget carefully: one SVD is ~40s and one eigendecomposition is
several minutes at that size, so the 32 decompositions alone are a couple of
hours before any labelling.

**Spawn subagents over these units.** Each subagent:

- takes a list of units, loads its model **once**, processes its units in order
- writes **its own shard**: `out/labels/<model>_<matrix>_L<layer>_<family>.json`
- never writes to a shared file — a final merge step combines the shards

**Memory.** SmolLM2 is ~0.5 GB in fp32, Qwen ~2 GB, Llama ~4 GB. Run **at most 6
SmolLM2 subagents, 3 Qwen, or 2 Llama concurrently.** Group units by model so a
subagent loads one model, not three. Do not exceed ~12 GB total.

**Do SmolLM2 J first** (units 1–28). It is the model everything else here was
validated on, so if your pipeline is broken it will show there first and cheapest.
Report after that batch before continuing.

### Nulls under parallelism

The null is **per (layer, matrix)** — it depends on `J` at that layer, so each
subagent computes its own for its units. That is correct, not duplicated work.

Because you are inventing axes, the number of tests is not fixed in advance. So:
use the **99.9th percentile of |null|** as the threshold, and **also record the
raw percentile** (`null_pct_beating`) for every direction, so a stricter
correction can be applied afterwards without rerunning anything.

### Merge

When shards are done, a final step reads `out/labels/*.json` into a single
`out/labels.json`, and only then builds the dashboard and report.

Olmo (`out/ckpt/`, 4096-dim) is out of scope — the SVDs alone are ~40s each.

---

## 5. Deliverables

**`out/labels.json`** — one record per direction:

```json
{"model": "smollm2-ft-step0", "matrix": "J", "layer": 16, "family": "eigen",
 "index": 3, "sigma_or_lambda": 2.41,
 "tokens_pos": [...], "tokens_neg": [...],
 "hypothesis": "British vs American spelling", "pairs": [["color","colour"], ...],
 "score": -44.82, "null_threshold": 16.70, "null_pct_beating": 0.0,
 "verdict": "validated",
 "steer": {"alpha": 0.01, "shift": 5.46, "random_shift": 0.09,
           "text_changed": true, "steers": "visibly",
           "examples": [{"prompt": "...", "base": "...", "plus": "...", "minus": "..."}]}}
```

**`out/DASHBOARD.html`** — a single self-contained file (inline CSS/JS, no CDN).
Filter by model / matrix / layer / family / verdict / steers. One card per
direction showing tokens, hypothesis, score vs threshold, and the before/after
generations. **Sort validated-and-visibly-steering to the top.** Include the
rejected ones behind a filter — a dashboard of only the wins misrepresents the
hit rate.

**`out/LABELS_REPORT.md`** — with, at minimum:
- the denominators: how many examined, how many got a hypothesis, how many
  validated, how many steered visibly
- **every axis you invented that was NOT already in `jlens/axes.py`**, and whether
  it validated — this is the most interesting output of the exercise
- directions where labelling was ambiguous and you want a human to look
- anything that surprised you or contradicts the framing above

---

## 6. Report back / escalate — but do not block on it

**If you ask something and get no reply, skip that item and keep going.** The
person running this is often away and will answer in a batch later. Never idle
waiting for a response.

When you hit something you would want a human on, do this instead of stopping:

1. record it in `out/OPEN_QUESTIONS.md` with enough context to answer cold —
   the direction, its tokens, what you tried, and what you would do either way
2. take the **conservative** option and note which you took
3. carry on with the next unit

Conservative means: if you cannot construct held-out pairs, mark
`verdict: no-hypothesis` rather than inventing weak pairs. If steering degenerates
at every alpha, mark `steers: no` rather than loosening the repetition threshold.
Never relax a control to get an answer — record the blockage and move on.

Raise in `OPEN_QUESTIONS.md` (and continue) when:
- a direction looks meaningful but you cannot construct held-out pairs for it
- steering degenerates at every alpha for a *validated* direction
- your validated fraction is wildly off the ~1-in-5 expected from prior runs
  (much higher probably means a broken null; report it, do not celebrate it)
- any result contradicts something asserted in this prompt

**One exception where you should genuinely stop:** if the SmolLM2 J batch
(units 1–28) produces a validated fraction above ~50% or below ~2%, halt and
write up why before spending the remaining 600 directions. Everything after that
batch is only worth running if the pipeline is behaving.

## 7. Standing rules

- **Report negatives.** "No coherent label" and "validated but does not steer" are
  results. A catalogue of only successes is worse than nothing.
- **Never report a number you did not compute**, and state the control beside it.
- Commit as you go; do not push.
- **A subagent that finishes must report its denominators** (examined / hypothesised
  / validated / steered), not just write its shard. If a subagent reports a
  validated fraction far above ~1-in-5, suspect its null before believing it.
- Do not modify existing files in `jlens/` except to import from them. New work
  goes in new modules.
