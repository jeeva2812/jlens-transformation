# Everything this project found, and what to write

Written after the full run. Every number below is from a file in this repo; the
code that produced it is named at the end of each section.

---

# PART 0 — The one-page version

**The finding to lead with.** Transformers have *exact symmetries*: weight edits
that leave the function bit-identical. That gives interpretability something it
almost never has — a **ground truth of exactly zero**. Audited against four of
them, across three model families:

- **Attention head-column readouts — the kind people publish — retain 0.00–0.01
  of their content** under a rotation the model provably cannot detect
  (max |Δlogit| 4×10⁻⁵ on a logit scale of 25; every greedy continuation
  identical). Replicated 3/3.
- **The fix works and is measured**: the OV circuit's subspace is invariant to
  3.6×10⁻⁷. But its singular vectors carry a *residual sign gauge* — read
  naively they score 0.45, sign-fixed by convention they score 1.00.
- **Residual-stream methods are invariant to everything tested.** That is the
  quantitative argument for SAEs over neuron-level interpretation.
- **Weight decay silently fixes the MLP rescaling gauge** — measured norm ratio
  1.015 / 1.032 / 1.036 across three models, where 1.000 is the exact
  weight-decay optimum. So neuron magnitudes are pinned, but by the regulariser,
  not by meaning.
- **Forgetting to fold RMSNorm's γ costs 19–43% of your readout.**

**The finding that shows taste.** Applying this to our own earlier result
overturned it. We had concluded that OLMoE's router directions read well because
of *the shape of the cone they span, not which rows they are*. But 38.3% of the
router matrix sits in an exactly free direction (softmax and top-k ignore a
constant added to every expert). Centre the router — changing no routing decision
— and the picture inverts:

| contrast | raw router | centred router |
|---|---|---|
| identity (do *these* rows matter?) | +0.0016 [−0.0127, +0.0131] | **+0.0294 [+0.0203, +0.0385]** |
| cone (is it just the shape?) | +0.0236 [+0.0084, +0.0425] | +0.0079 [+0.0036, +0.0129] |

The gauge component was **masking** a real effect. Gauge-fixing didn't just
avoid an artefact — it revealed one.

**What not to claim.** No discovery about what models believe or do. This is a
methods contribution: a test with provable ground truth, a ranking of methods
against it, and a catalogue of eight ways these measurements go wrong.

---

# PART 1 — Why the gauge work is the answer

Eight arms were tried. Seven died, and they died the same way: **a control showed
something simpler explained the result.** The J-Lens arm lost to identity. The EM
convergence lost to benign fine-tunes. The concealment probe lost to
bag-of-words. Each time the deflation was found in-house, which is the right
outcome, but it doesn't ship.

Exactly one arm was immune, and the reason is structural: **its answer was
provable in advance.** `W_V → RW_V, W_O → W_O Rᵀ` leaves the function unchanged,
so any method reporting a different answer afterwards is measuring coordinates.
There is no control that can deflate a theorem.

So the pivot was to build the project on that property. That is Part 2.

---

# PART 2 — The gauge audit (flagship)

## 2.1 What a symmetry is

A weight edit that leaves the function *exactly* unchanged: two parameter
settings, identical behaviour on every input. Anything you read off the weights
or activations that differs between them is not a fact about the model.

Analogy: giving a position as "3 m from the north wall" vs "3 m from the door."
Same spot. A safety system trained on north-wall coordinates breaks when someone
relabels the walls — with the room unchanged.

| symmetry | the edit | why exact |
|---|---|---|
| attention head rotation | `W_V → R·W_V`, `W_O → W_O·Rᵀ` (tied across a kv group) | nothing acts per-slot inside a head |
| MLP neuron rescaling | `W_up[i] ×= c`, `W_down[:,i] ÷= c` | `W_up` enters the SwiGLU product linearly, so *c* cancels |
| neuron permutation | shuffle hidden units + matching weights | relabelling |
| router shift (MoE) | `W_gate → W_gate + 1·vᵀ` | softmax and top-k ignore a constant added to every expert |

Every one was verified numerically before any method was scored, and every edit
undone exactly (restore error `0.00e+00`).

## 2.2 The matrix

1.00 = output unchanged (measures the model). 0.00 = completely different
(measures the coordinates). Three columns per symmetry: SmolLM2-135M /
Qwen2.5-0.5B / Llama-3.2-1B.

| method | head rotation | MLP rescale 9× | MLP rescale 2× | permutation |
|---|---|---|---|---|
| **head-column readout** (the published kind) | **0.01 / 0.00 / 0.00** | 1.00 | 1.00 | 1.00 |
| OV singular vectors, naive | 0.45 / 0.47 / 0.48 | 1.00 | 1.00 | 1.00 |
| **OV singular vectors, sign-fixed** | **1.00** | 1.00 | 1.00 | 1.00 |
| **OV subspace** | **1.00** | 1.00 | 1.00 | 1.00 |
| MLP logit lens (γ folded) | 1.00 | 1.00 | 1.00 | **0.00** |
| max-activating examples | 1.00 | 1.00 | 1.00 | **0.05** |
| cross-unit activation ranking | 1.00 | **0.87** | 0.98 | **~0.00** |
| activation × gradient | 1.00 | 1.00 | 1.00 | **~0.00** |
| residual-stream direction (SAE proxy) | **1.00** | **1.00** | **1.00** | **1.00** |

`|Δlogit|` for every row: 3.9–5.9×10⁻⁵. Float32 noise.

## 2.3 Reading it

1. **Head-column readouts are the casualty**, 3/3. "Dimension *i* of head *h*
   promotes X" is a claim about a coordinate system the model never picked.
2. **The OV circuit is the fix — with a caveat worth its own line.** Its subspace
   is invariant to 3.6×10⁻⁷ and per-vector |cos| is 1.0000. But the *signed*
   cosines are −1, −1, −1, +1, −1, −1, +1, −1: the SVD has its own sign gauge,
   and top-k of `−v` is the bottom-k of `v`. Fix the sign by convention (largest
   component positive) and invariance returns. Separately, 55 of 63
   singular-value ratios are below 1.05 — a near-degenerate spectrum, so
   individual vectors are weakly pinned regardless.
3. **MLP methods come out well.** Activation × gradient is invariant to
   rescaling for a pretty reason: activation scales by *c*, its gradient by
   `1/c`, and the product cancels.
4. **One MLP method degrades — cross-unit activation ranking** — 0.87 at 9× and
   0.98 at the ~2× that weight decay actually leaves.
5. **Weight decay fixes that gauge for you.** It penalises
   `‖W_up‖² + ‖W_down‖²`, whose minimum over the free scale is at balanced norms.
   Measured `‖W_up row‖ / ‖W_down col‖`: **1.015** (Olmo-3-7B), **1.036**
   (Qwen2.5-0.5B), **1.032** (SmolLM2-135M). Pinned by the regulariser, not by
   meaning — so magnitudes are not comparable across differently-regularised models.
6. **Residual-stream methods are untouched by everything.** This is the SAE
   argument with a number: head rotation, rescaling and permutation all leave the
   residual stream exactly unchanged.
7. **γ costs 19–43%.** Overlap between the γ-folded and naive logit lens: 0.81 /
   0.67 / 0.57.

## 2.4 What gauges are actually used in practice

| symmetry | conventional fixing | status |
|---|---|---|
| attention head rotation | **none** — the trained basis is read as-is | **unfixed**; the correct move is to use invariants (the OV product) |
| MLP neuron rescaling | weight decay → balanced norms | fixed **incidentally by the optimiser** |
| neuron / expert permutation | none, usually | **unfixed** — why "neuron 1423" doesn't survive a reseed |
| router shift (MoE) | **none** — nobody centres the router | **unfixed, 38.3% of the norm** |
| RMSNorm γ | folding γ into `W_U` — the logit-lens convention | fixed by convention, when remembered |
| SAE decoder scale | unit-norm decoder columns | **properly fixed by convention** |

Three tiers of practice: (1) report invariants; (2) fix a gauge canonically and
say which; (3) read whatever training left you — the common case.

*Code: `gauge/matrix.py`, `gauge/router_centre.py`, `gauge/dump_readouts.py`,
`jlens/rotate_sweep.py`. Data: `out/gauge/`.*

---

# PART 3 — The privilege battery

Pre-registered (`docs/PRIVILEGE_PREREG.md`, written before any number existed),
604 groups, four models. Question: does the model's own basis read better than a
random basis of the **same subspace**?

Construction: for 64 unit vectors `A` with `A = QR`, compare `A` (raw) against
`A·H` (the symmetry orbit), `Q·H·R` (same Gram matrix), `Q·H` (orthonormal), and
the SVD basis. All span the same subspace, verified to 3×10⁻⁷.

Scoring: top-10 logit-lens tokens judged by coherence **in a different model's
embedding space** (Llama-3.2-1B; Qwen2.5-0.5B as a check), frequency-matched
against 400,000 random token pairs. The model being probed never judges itself.

| arm | position | raw − orbit (95% CI) |
|---|---|---|
| **attention head columns** (provable null) | — | **+0.0011 [−0.0023, +0.0047]** |
| OV-circuit singular vectors | — | **+0.0065 [+0.0033, +0.0097]** |
| attention columns across heads | — | **+0.0339 [+0.0244, +0.0439]** |
| residual-stream basis | — | −0.0023 [−0.0055, +0.0007] |
| **MLP neurons** | — | **+0.0209 [+0.0121, +0.0297]**, ratio 1.31 |

**The null arm measures zero.** That is the load-bearing line: it is the only arm
whose answer was known in advance, and the instrument was not tuned to produce it.

Against that calibration: MLP neurons are privileged (8.5–13.4× the null arm in
three of four models); head boundaries are real (mixing columns *across* heads
costs 1.49×, *within* a head costs nothing); the OV singular basis reads better
than the head columns *for the same subspace*. Two pre-registered predictions
were **refuted** — the residual-stream basis is not privileged at the tail, and
MoE router rows are not privileged by identity (in the raw gauge — see Part 2.2,
which overturns this).

Robustness: swapping the judging model changes the null arm from +0.0038 to
+0.0001 and leaves MLP at +0.0344 vs +0.0348. Widening the scored vocabulary
*increases* the effect (+0.0547).

*Code: `jlens/privilege.py`, `jlens/tokspace.py`, `jlens/privilege_report.py`.*

---

# PART 4 — Reward hacking and CoT monitoring

Substrate: **Qwen3.5-9B**, DAPO group-relative PPO on CodeContests against a
deliberately vulnerable evaluator. 300 held-out trajectories released by
lucabaroni, upstream **UK AI Safety Institute**, judged blind by GPT-5.6.
We did not train it.

**Fidelity gate first.** With the LoRA attached we reproduce the released policy
log-probs at **r = +0.995, MAE 0.039 nats**, versus **r = +0.942** for the base
model. So the activations are the hacking policy's own.

## 4.1 The setting

| | |
|---|---|
| grader-confirmed hacks | 275 / 300 |
| concealed (blinded judge sees no hack awareness) | 201 |
| lexical CoT monitor catches, on that set | **18.4%** |
| hacks that still contain a real solution attempt | **70.5%** |
| where the exploit sits in the code | median **3.0%** through |
| dead code after the exploit | median **1032 chars** |

The model reasons correctly, writes `os._exit(0)` at the top, then writes the
real solution underneath as code that can never run.

## 4.2 The one clean result

Causally-clean cells only — the label must concern something not yet in the
context — and the baseline is a **text monitor**, not chance.

| question | position | residual AUC | text baseline | null |
|---|---|---|---|---|
| will it hack? | before any generation | 0.46 | 0.49 | 0.41 |
| will it hack? | 4 tokens before the exploit | 0.68 | 0.52 | 0.49 |
| **will a real solution follow?** | **at the exploit, before that code exists** | **0.811** | **0.449** | **0.39** |
| will the CoT conceal it? | before any reasoning | 0.56 | 0.53 | 0.52 |

n = 183, balanced 106/77, with a proper depth curve (0.46 → 0.60 → 0.71 → 0.80 →
0.81 across layers) rather than a flat line. **Transfers to an unseen exploit
mechanism**: trained only on `exit`, tested only on `always_equal`, AUC **0.775**
(within-mechanism 0.825, null 0.525; n_test = 22, so directional).

## 4.3 The causal test, with its prediction confirmed

Keep the code fixed, destroy the reasoning, re-measure:

| variant | Δ(exploit log-prob) | Δ(real code log-prob) | difference |
|---|---|---|---|
| delete the reasoning | **−0.011** | **−0.290** | **+0.279** |
| halve the reasoning | −0.049 | −0.294 | +0.245 |

**Delete the entire chain of thought and the exploit's probability does not
move** (−0.011), while the real code degrades substantially (−0.290). The
reasoning is load-bearing for the solution and not at all for the exploit.

And the **pre-registered mechanism contrast held**: I predicted that deleting the
reasoning would leave the trivial `exit` hack untouched and degrade the more
involved `always_equal` hack. Measured: `exit` **+0.007** (n=43), `always_equal`
**−0.098** (n=9) — a 14× difference in the predicted direction. Small n on the
second arm. (`swap_cot` is not interpretable here: substituting another task's
reasoning creates an off-distribution context that hurts everything.)

## 4.4 Two deflations, both important

1. **The concealment probe scoring AUC 1.00 was circular.** Bag-of-words on the
   reasoning text predicts the same label at **0.985**, and a single keyword rule
   at 0.820. The label is a function of the text and the text is in the context.
2. **The system prompt spells out all three hacks step by step** — including
   *"use `os._exit(0)` rather than `sys.exit(0)`, because pytest catches
   SystemExit"* — and then forbids them. The model discovers nothing. This is
   closer to instruction-following violation than emergent reward hacking, and it
   makes "concealment" a weak construct: there is no secret.

**What survives that.** The 0.811 result, because what it predicts is *not in the
prompt, not in the text yet, and not derivable from "RL rewarded hacking."* The
training story says nothing about why the model sometimes also does the work.

*Code: `rh/*.py`. Data: `out/rh/`, 48 GB of captured residuals + 28 GB base pass.*

---

# PART 5 — Emergent-misalignment organisms

Three published EM fine-tunes (bad medical advice / risky financial advice /
extreme sports) on Llama-3.2-1B and Qwen2.5-0.5B.

**The trap.** All three share a LoRA initialisation — their `A` matrices are
cosine **0.9994** identical. So the *input* side of all three weight changes is
identical by construction. Anyone comparing these adapters in weight space finds
enormous "convergence" that is initialisation, not learning. Only the output side
(`B`) is learned, and `col(BA) = col(B)`, so the write-side comparison is the
init-invariant one.

**The control.** We trained three LoRAs ourselves with controlled seeds
(sanity: same seed → A cosine 0.9874, different seed → 0.0255):

| | same-read-direction excess |
|---|---|
| two unrelated **benign** fine-tunes | 1.58× |
| the three **EM** fine-tunes | 2.17× |

Two benign fine-tunes already show most of the effect. **Convergence is mostly
generic to fine-tuning this base model**, with a modest EM-specific excess.

**Does it survive a different init?** Write-space overlap (chance = 0.036):
different init / same data **0.294**, same init / different data 0.139, EM pair
0.175. Task identity matters more than the seed.

*Code: `em/*.py`.*

---

# PART 6 — The negatives

- **J-Lens transport.** 2160 directions, four models: 63 pass a readout test,
  **3 change generated text, all 3 in one 135M model**. Eight measurements,
  three families, two readout spaces, with a working positive control (identity
  gets 7.2/8 experts).
- **Chess — the mechanism.** With an oracle instead of human labels, the top of
  J's spectrum encodes a **tokenisation artifact** (which half of a move is being
  emitted, R² 0.66). Cause: `cos(J_origin, J_dest) = 0.21` — a prompt-averaged
  Jacobian averages over different computations, so its dominant directions
  encode the mixture.
- **MoE routing.** The Jacobian **loses to doing nothing** at predicting which
  experts fire (7.2 vs 1.3 of 8, chance 1.0). Routing tracks surface form, not
  algorithm. One steering demo survived.
- **Localisation.** Pile-averaged J is dead as an edit-site predictor: no
  within-layer signal, R² gain +0.000 over the layer prior.

---

# PART 7 — The failure catalogue

Each found in this project, with a number.

1. **Circular labels.** A probe read "did the CoT conceal the hack" at AUC 1.00;
   bag-of-words on the reasoning gets 0.985. *Fix: only score positions where the
   label concerns something not yet in the context.*
2. **Chance is the wrong baseline.** For a monitoring claim the baseline is what
   a text monitor gets.
3. **Confounded summary statistics.** Exploit tokens looked far more confident
   than surrounding code (−0.14 vs −0.38) until the first-token statistic gave
   d = −0.01. The mean was measuring that `_exit` is free after `os.`.
4. **Position is a hidden variable.** +12.3σ vs −1.3σ, same direction, two tokens
   apart.
5. **One dose is not a sweep.** Dose flipped the efficacy/specificity correlation
   from −0.47 to +0.24.
6. **Shared initialisation looks like convergence.** Cosine 0.9994.
7. **"Shared" and "biggest" are not separable** with three fine-tunes: held-out
   shared transferred at 0.941, top-k of a single run at 0.937.
8. **Beating an orthonormal basis is weak evidence.** 73% of the router's gap was
   reproduced by a random non-orthogonal basis — and the rest was gauge.
   *Fix: compare against the symmetry orbit.*

---

# PART 8 — What I would put in the application

**Lead:** the gauge audit. It is the only result with provable ground truth, it
replicates 3/3, it produces a ranking rather than a nihilistic verdict, and the
headline is one sentence: *rotate an attention head and the logits move by
4×10⁻⁵ while 99% of what you would read out of that head changes.*

**Second:** the router self-correction. It shows the audit doing real work —
overturning one of our own published-in-repo conclusions, and revealing a real
effect the gauge had masked. That is the strongest evidence of taste in the whole
project.

**Third:** the privilege battery, as the calibration story — a pre-registered
study whose provable-null arm measured zero, with two predictions refuted.

**Fourth:** the failure catalogue, as the body rather than an appendix. Eight
concrete ways these measurements go wrong, each with a worked example.

**Do not claim:** a discovery about what models believe or do. Do not claim the
reward-hacking work as an emergent-misalignment result — the system prompt gives
the recipes.

---

# PART 9 — Limits and next steps

- **Invariance is necessary, not sufficient.** A gauge-invariant method can still
  measure nothing. This rules out; it does not certify.
- **Four symmetries, not all.** `norm_absorb` is untested because all three
  matrix models tie embeddings; it needs an untied model.
- **The SAE claim is by proxy.** We show residual-stream directions are
  invariant, which is what SAEs inherit — we did not push a trained SAE through
  the audit. That is the obvious next step and it is cheap (Gemma Scope is a
  download).
- **One head, one layer per model** in the matrix; the head-rotation effect was
  separately replicated across all 70 layers of three models.
- **Small models** for the matrix (135M–1B), 7B for the router. The symmetries
  are architectural so they hold at any scale, but the magnitudes are measured
  here.
- **The reward-hacking substrate is weak** for the general-case question. Better
  organisms exist and are downloaded: `ariahw/rl-rewardhacking-leetcode-{rh,
  rl-baseline}-s{1,42,65}` (Qwen3-4B, matched hacking / non-hacking pairs, three
  seeds) and `antrip03/grpo-c{1_baseline,2_hackable}` (Qwen2.5-1.5B, five seeds).

---

# PART 10 — Where everything is

| what | where |
|---|---|
| gauge audit code | `gauge/matrix.py`, `gauge/router_centre.py`, `gauge/dump_readouts.py` |
| gauge results | `out/gauge/*.json`, `out/GAUGE_AUDIT.html` |
| head-rotation sweep (70 layers) | `jlens/rotate_sweep.py`, `out/priv/rot_*.json` |
| privilege battery | `jlens/privilege.py`, `out/priv/`, `docs/PRIVILEGE_PREREG.md` |
| reward hacking | `rh/*.py`, `out/rh/` (48 GB residuals + 28 GB base pass) |
| reward-hacking viewer data | `out/rh/index.jsonl` (300 rows: prompt, CoT, code, exploit spans, labels) |
| EM organisms | `em/*.py`, `out/em/` |
| earlier write-ups | `docs/GAUGE_RESULT.md`, `docs/WHAT_WE_HAVE.md`, `docs/CROSSMODEL_SIGNIFICANCE.md` |

---

# APPENDIX — Robustness of the matrix

Re-run with two further random seeds (different rotation `R`, different rescaling
factors, different permutation, different sample of MLP units) and at a different
layer:

| cell | across seeds and layers |
|---|---|
| head-column readout under rotation | **0.00 – 0.01** |
| OV subspace / sign-fixed readout | **1.00** every time |
| OV readout, naive | 0.44 – 0.50 |
| cross-unit ranking, 9× rescale | 0.86 – 0.88 |
| cross-unit ranking, 2× rescale | 0.98 – 0.99 |
| permutation, index-based methods | 0.00 – 0.01 |
| γ convention overlap (SmolLM2) | 0.53 – 0.59 |

All `|Δlogit|` between 3.4 and 5.4×10⁻⁵; every restore exact (`0.00e+00`).
