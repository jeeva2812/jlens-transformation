# Research brief: localization vs. intervention in language models

You are working autonomously. This is a brief, not a task list — it describes a
problem space, what is already known, what is already taken, and what looks
unclaimed. Decide for yourself what is worth doing, in what order, and when an
idea is dead. If you conclude the whole space is exhausted, say so and explain
why; that is a valid and useful outcome.

If nobody replies to you, keep going. Do not wait for approval.

---

## The tension

**ROME** (Meng, Bau et al. 2022) localises facts with *causal tracing* — corrupt
the subject tokens, restore individual hidden states, see which restoration
recovers the correct answer. It finds mid-layer MLPs at the last subject token,
and edits there with a closed-form rank-one weight update. No training involved.

**Hase, Bansal, Kim & Ghandeharioun** (NeurIPS 2023, arXiv 2301.04213) then
showed the premise does not hold. Edit success is essentially unrelated to where
causal tracing localises the fact: ρ = −0.13 on GPT-J at layer 6. Choice of edit
layer alone explains 94.7% of rewrite-score variance; adding the tracing effect
takes it to 94.8%.

So: **the place a fact appears to live is not the place to change it.** That is
still unexplained, and the field has largely routed around it rather than
resolving it.

## What is already taken (verify this list; it was assembled quickly)

- **REMEDI** — Hernandez, Li & Andreas, arXiv 2304.00740, COLM 2024. Activation-space
  fact editing and probing. Code at github.com/evandez/REMEDI.
- **SAKE** — arXiv 2503.01751. Steering activations for knowledge editing.
- **MEGA / "The Anatomy of an Edit"** — arXiv 2603.20795, 2026. GPT2-XL and
  LLaMA2-7B on CounterFact and Popular. Uses *post-edit* neuron-level attribution
  (contrasting successful against failed edits) rather than pre-edit tracing, and
  proposes attention-residual steering. Explicitly treats tracing as a poor
  predictor of intervention sites.
- **Precise Localization of Memories** — arXiv 2503.01090, neuron-level editing.
- **Weight Patching** — arXiv 2604.13694, source-level mechanistic localization.

Assume the obvious experiment has been done unless you can show otherwise. Spend
real effort on this before building anything.

## Ideas that look unclaimed

Offered as starting points, not assignments.

**The Jacobian as an edit-site predictor.** `J_l = E[∂h_target/∂h_l]` is a linear
model of how a perturbation at layer l propagates to the output. That makes
`||J_l d||` a natural predictor of how much an activation edit along direction
`d` at layer l will move the output — which is exactly the quantity tracing fails
to predict. Three curves per fact: tracing effect, edit efficacy, J-amplification.
Hase got ρ = −0.13 for the first pair. What do the others give? A cheap positive
result here would say localization *is* informative, just not by the measure the
field used.

**Read/write dissociation as the general phenomenon.** Tracing-vs-editing is one
instance. A prior project in this repo found another: of 2160 Jacobian-lens
directions, 63 passed a readout probe against a random-direction null and only 3
changed generated text under injection. If the same dissociation shows up across
unrelated method families, it is a property of the representation rather than of
any one technique. Worth checking whether that framing survives contact with
data.

**Where the two disagree.** Hase reports a near-zero *correlation*, which is
compatible with several different structures. Are there facts where tracing and
editing agree? What distinguishes them — frequency, subject type, how many layers
the fact is spread across? A characterisation of the disagreement may be more
tractable than explaining it.

## Assets in this repository

- `jlens/lens.py` — `jacobians_all_layers`, computes J at every layer for roughly
  the price of one. Read the module docstring; it names three errors that are easy
  to make and load-bearing. GPT-2 is supported (`.transformer.h` / `.ln_f`).
- `jlens/steer_validated.py` — steering harness, alpha calibration, random-direction
  controls.
- `jlens/label_axes.py`, `jlens/reaudit.py` — probe and audit machinery.
- Models cached locally: GPT-2 family via HF, SmolLM2-135M base+instruct,
  Qwen2.5-0.5B base+instruct, Llama-3.2-1B base+instruct (unsloth mirrors, ungated),
  Olmo-3-7B, a chess GPT-2. `.venv/bin/python`, run modules as
  `PYTHONPATH=. .venv/bin/python -m jlens.<module>`.

## House rules, learned expensively in this repository

1. **Every positive result needs an outlier control, not just a null.** A finding
   here passed a 300-draw random-direction null and then reversed sign at every
   layer once one massive-activation direction was projected out.
2. **Verify chance levels empirically rather than deriving them.** A component-level
   dissociation in this repo was entirely an artifact of using `k/min(m,n)` as the
   chance line for a tall matrix where the correct value was `k/m` — a 5.4× error
   that manufactured the whole finding.
3. **Compare like with like.** A cross-model "replication" here was a post-audit set
   compared against a pre-audit set. Check both sides went through the same pipeline.
4. **Whiten before comparing directions.** At layer 4 of SmolLM2, two mean-difference
   vectors from *unrelated random splits* have |cos| 0.436 against a chance level of
   0.033, because one direction holds 71.5% of the variance. Anisotropy is severe at
   early-middle layers and mild from L12 on.
5. **Beware metrics dominated by a marginal.** Probing legality in a chess model gave
   AUC 0.999 — of which 88–98% was the per-square base rate. Four consecutive
   measurements were wrong before that was noticed.
6. **A methodological correction is not a finding.** Fixing your own mistake is table
   stakes.
7. **Measure costs before asserting them.** A cost estimate here was wrong by two
   orders of magnitude.
8. **Report what the run actually printed**, including when it contradicts this brief.

## What a good outcome looks like

One well-controlled measurement with its controls named in advance, written up in
two pages, beats a broad survey. A clearly-argued "this space is exhausted, here
is the evidence" is a good outcome. A positive result without an outlier control
is not.
