# Autonomous research task: does any of this replicate?

You are in `/Users/sjeeva/Projects/Interpretability/Neel/jlens-transformation`.
Use `.venv/bin/python`, run modules as `PYTHONPATH=. .venv/bin/python -m jlens.<name>`.

**Everything is on disk. No GPU, no downloads, no training, no cost.** If a task
would need any of those, it is out of scope — say so and pick another.

**Work autonomously.** If you ask a question and get no answer, record it in
`out/OPEN_QUESTIONS.md`, take the conservative option, note which you took, and
keep going. Never idle waiting for a reply.

---

## The situation

A previous effort characterised **J-Lens** — a lens that reads a transformer's
intermediate activations from a Jacobian computed from weights, with no training:

    J_ℓ = E_prompt[ ∂h_target / ∂h_ℓ ]     readout(d) = softmax(W_U · norm(J_ℓ d))

It produced a large number of results. **The single biggest weakness is that
almost none of them have been tested on more than one model.** Specifically:

- 1480 directions were labelled across four models. **10 distinct axes validated.
  Zero of them validated on more than one model.**
- The flagship demonstration — a gender direction at layer 24 of SmolLM2-135M
  that flips pronouns in generated text — has **no counterpart** found in
  Qwen2.5-0.5B or Llama-3.2-1B.
- Steering pass rates (64% and 88%) come from two models with different protocols.

**Your job is to find out which claims survive contact with a second model, and
to say plainly which do not.** A well-supported negative is the goal, not a
failure. The most valuable output would be a short list of "this replicates" and
a longer list of "this does not, and here is the evidence."

---

## What is on disk

Jacobians. Each file is `{"layers": [...], "target": int, "J": {layer: Tensor}}`.

| path | model | d_model | layers saved (of model total) |
|---|---|---|---|
| `out/Jall_smollm2.pt`, `out/ft/J_step0.pt` | SmolLM2-135M-Instruct | 576 | 0,2,…,28 (15 of 30) |
| `out/ft/J_step600.pt` | same, insecure-code fine-tune | 576 | same |
| `out/ft_lr1e-5/` | same data, lower learning rate, 13 checkpoints | 576 | same |
| `out/em05_emprompts/J_{base,medical,financial,sports,control}.pt` | Qwen2.5-0.5B-Instruct + 4 LoRAs | 896 | 4,8,12,16,20 (5 of 24) |
| `out/llama_em/J_{base,medical,financial,sports}.pt` | Llama-3.2-1B-Instruct + 3 LoRAs | 2048 | 0,4,8,12 (4 of 16) |
| `out/ckpt/J_<rev>.pt` × 11 | Olmo-3-7B across 1.47M training steps | 4096 | 0,2,…,30 (16 of 32) |

`out/readout_head.pt` holds Olmo's `W_U` and final norm, so Olmo readouts do not
need the 7B model loaded. All other models load from HuggingFace cache.

**Costs, measured, so you can plan:** a 4096×4096 SVD is 11s and an
eigendecomposition 7.4s. Smaller models are proportionally faster. Recomputing a
Jacobian at new layers for Qwen (896-dim, 20 prompts) is roughly 40 minutes on
CPU — affordable, and the single most useful thing you can do if sparse layer
coverage is blocking a replication test.

Prior results are in `out/*.json` and `out/labels/*.json`. `docs/` has the
write-ups. Read `README.md` first.

---

## Where to start

**The layer-coverage confound, first, before anything else.** Qwen has 5 of 24
layers saved and Llama 4 of 16. "Gender does not replicate" may simply mean "we
never looked at the layer where it lives." Recompute Qwen's Jacobian at the
missing layers (reuse `jlens/lens.py:jacobians_all_layers`) and re-test for
gender specifically, using the pair test in `jlens/axes.py`. This either kills
the confound or explains the whole non-replication in one experiment.

After that, choose your own targets. Candidates, in rough order of value:

1. **Steering across models.** The 64%/88% figures used different alphas and
   different layer sets. Run one protocol on all four models and report a like-for-like
   table. `jlens/assay_uv.py` is the reference implementation. Inject `v`
   (layer-ℓ space), never `u` (target space) — that type error already cost this
   project a retracted finding.
2. **Depth is a lever.** The strongest surviving claim: an identical-size weight
   change moves the transport ~4× more at layer 2 than layer 27, and matched
   random noise shows the same gradient. It rests on **one model and one
   fine-tune**. `jlens/position_sweep.py` does it by grafting one fine-tuned
   layer at a time onto a base model; `out/ft_lr1e-5/` gives you a second
   fine-tune for free, and Qwen and Llama have LoRA organisms you can graft.
3. **Eigen vs SVD.** Reported as eigen 39.6% vs SVD 20.8% for reading, and the
   reverse for steering. The labelling agent got a dead tie under a different
   protocol. Determine which protocol is right, or show the result is
   protocol-dependent and should be retired.

---

## Standing rules — these exist because each was violated once, expensively

**Every number needs a null, and the right one.** A finding here passed a
random-direction null and died to an outlier control, because the residual stream
carries up to 99.8% of its variance in one massive-activation dimension.
**Before any claim about activations, project out the top 1–3 activation
directions and check the claim survives.** See `jlens/occupancy_control.py`.

**Consistency is not evidence; magnitude is.** ~29% of random directions separate
all 20 held-out US/UK spelling pairs the same way. Score against a null and
report the ratio, never the sign-agreement count.

**Calibrate steering strength per direction, across every prompt.** At 400× the
working alpha a direction emits `'her her her her…'` and scores a *perfect*
result from a destroyed model. Measure repetition (`1 - unique/total` tokens) and
discard anything above 0.30.

**A log-ratio shift is not a behaviour change.** A direction moved P from 0.04 to
1.00 with the generated text completely unchanged. When you claim steering works,
show the text.

**Report negatives at equal length to positives.** "Does not replicate" is the
expected outcome here and is the point of the exercise.

---

## Deliverables

- `out/REPLICATION.md` — one row per claim tested: what it was, which models,
  what happened, and a verdict of `replicates` / `model-specific` /
  `refuted` / `untestable from disk`. Lead with the ones that failed.
- `out/replication/*.json` — the underlying numbers, one file per experiment.
- A short `out/REPLICATION.html` if a figure makes something clearer than a
  table. Self-contained, no CDN.
- Commit as you go. Do not push.

Stop and write up if you reach ~6 hours of compute, whatever state you are in.
A partial answer with clean controls beats a complete one without.
