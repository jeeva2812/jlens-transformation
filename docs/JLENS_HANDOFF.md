# J-Lens work — handoff

Written for an agent with no prior context. Working dir
`/Users/sjeeva/Projects/Interpretability/Neel/jlens-transformation`.
Run everything as `PYTHONPATH=. .venv/bin/python -m jlens.<module>`.

## 0. What the J-Lens is

`J_l = E[ ∂h_target / ∂h_l ]`, the averaged Jacobian from the residual at layer `l`
to the residual at a target layer, estimated over a text corpus. `W_U J_l` is the
readout matrix: the score for token `k` given activation `h` is
`⟨W_U[k], J h⟩ = ⟨Jᵀ W_U[k], h⟩`.

From [Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace"]
(https://transformer-circuits.pub/2026/workspace/index.html). Row `k` of `W_U J_l`
is what that paper calls the **J-lens vector** for token `k`, and it equals
`Jᵀ W_U[k]`.

Jacobians on disk:

| file | model | layers | d |
|---|---|---|---|
| `out/Jall_smollm2.pt` | SmolLM2-135M | 0..28 even | 576 |
| `out/Jall_main.pt` | Olmo-3-1025-7B (dense) | 0..30 even | 4096 |
| `out/Jall_olmoe.pt` | OLMoE-1B-7B-0924 (MoE) | 4,8,12 → target 14 | 2048 |
| `out/Jall_smollm2_split.pt` | SmolLM2, same protocol as the MoE | 4,8,12 | 576 |
| `out/ckpt/J_*.pt` | OLMo pretraining checkpoints | | 4096 |

Nearly all results below are **SmolLM2-135M only**. Treat that as the headline
caveat on everything.

---

## 1. THE RESULT — pullback steering

`jlens/pullback_steer.py` → `out/rare/pullback2.json`

Name a concept, build the target-space direction `w` that means it, solve for the
layer-`l` vector whose transport lands on `w`. 7 concepts × 4 layers = 28 cells.

**Protocol that makes it trustworthy:** each concept's word list is split in half.
`w` is built from the TRAIN half and the effect is scored on the TEST half, which
never touched the construction. Lift = (held-out concept words moved) − (control
tokens matched on prior probability). See §5 for why this matters.

| method | mean lift | beats random | damage to unrelated text |
|---|---|---|---|
| **`Jᵀw` (transpose / pullback)** | **1.64** | **28/28** | +0.172 |
| ridge λ=1 `(JᵀJ+λI)⁻¹Jᵀw` | 0.85 | 25/28 | +0.134 |
| best single SVD direction | 0.76 | 20/28 | +0.112 |
| ridge λ=0.1 | 0.51 | 17/28 | +0.115 |
| **`w` alone — no J at all** | **0.64** | **17/28** | +0.209 |
| ridge λ=0.01 | 0.38 | 13/28 | +0.099 |
| **`J⁺w` (pseudo-inverse)** | **0.00** | **3/28** | +0.022 |
| random direction | 0.01 | 0/28 | +0.031 |

Three findings:

**(a) `Jᵀ` beats naive activation steering 2.6×, with less damage** — and the
advantage is largest deep in the network:

| layer | `w` alone | `Jᵀw` | ratio |
|---|---|---|---|
| 4 | 0.11 | 0.56 | 5.1× |
| 12 | 0.16 | 1.21 | 7.6× |
| 20 | 0.68 | 2.29 | 3.4× |
| 26 | 1.60 | 2.51 | 1.6× |

The two converge at the target layer, where `J → I` so `Jᵀw → w`. That is an
internal consistency check nobody designed and it passed.

**(b) Do not invert J.** The pseudo-inverse is dead flat (3/28), and the ridge
series climbs monotonically as λ grows toward the transpose. J's near-null
directions are noise; asking for exactly `w` divides by them.

**(c) It is a SUBSPACE, and truncating beats the full vector.**
`jlens/pullback_rank.py` → `out/rare/pullback_rank.json`. Writing the pullback in
J's own basis, `Jᵀw = Σᵢ σᵢ ⟨uᵢ,w⟩ vᵢ`, and truncating at `k`:

| directions kept | lift | % of full | share of the pullback's energy |
|---|---|---|---|
| 1 | −0.13 | −8% | 0.016 |
| 8 | 1.11 | 63% | 0.169 |
| 16 | 1.55 | 89% | 0.325 |
| **32** | **1.83** | **105%** | 0.495 |
| **64** | **1.89** | **108%** | 0.693 |
| 576 (full) | 1.75 | 100% | 1.000 |

One direction does nothing. 32–64 directions beat the whole spectrum. `k=32`
carries half the energy and gives 105% of the effect, so the small-σ tail is
actively counterproductive — same reason the pseudo-inverse fails.

**Novelty, stated honestly:** `Jᵀw` for a single token IS the J-lens vector from
the paper, and steering with it is the paper's own intervention. Not new. What is
new here: the head-to-head against naive activation steering with the depth
crossover, the truncation optimum (the paper uses the untruncated vector), the
pseudo-inverse negative, and the held-out-word protocol.

---

## 2. What the SVD of J is good for

**Topic information is concentrated in J's directions, and J beats PCA early.**
`jlens/subspace_meaning.py` → `out/rare/subspace_meaning.json`. 6 topics,
72 prompts, chance 17%, leave-one-out nearest centroid on the coefficients
`⟨vᵢ, h⟩` — no readout tokens involved.

| layer | 8 dirs of J | 8 activation PCs | 8 random dirs | shuffled labels |
|---|---|---|---|---|
| **4** | **76%** | 47% | 34% | 15% |
| 12 | 79% | **90%** | 54% | 22% |
| 20 | 85% | **88%** | 53% | 15% |
| 28 | 69% | **97%** | 71% | 19% |

Layer 4 is the interesting one: the activations there have **effective rank 2.4
of 576** (measured from 12k pile tokens), so PCA burns its whole budget on two
giant directions and gets 47%. J ranks by what survives to the output rather than
by size and gets 76%. **J finds directions that matter downstream, not directions
that are large.** By layer 28 J ≈ identity, its spectrum is degenerate, its
directions are arbitrary, and it drops to random — a predicted null that came out
null.

**Individual directions separate individual topic pairs**, but the first numbers
were selection-inflated. Correction in
`jlens/selection_check.py`:

| | value |
|---|---|
| best-of-64 AUC under **shuffled labels** | mean 0.79, 95th pct **0.88** |
| best-of-64 AUC, real labels | 0.976 / 0.992 / 0.999 (L4/12/20) |
| pairs beating the shuffled 95th pct | **15/15 at every layer** |
| **held-out AUC** (direction picked on other prompts) | **0.78 / 0.82 / 0.84** |

Quote **0.8, not 1.0**. The null for a max-over-64 statistic is 0.88, not 0.5.

Form vs topic: question-vs-statement separates (AUC 0.89–0.92, 1–2 of 64
directions). **Past-vs-present does not, at any layer** (0.69–0.73, 0 of 64).

---

## 3. What the SVD of J is NOT good for

**Readability does not predict steerability.** `jlens/steer_interp.py --bands`
→ `out/rare/steer_bands.json`. Word-directions split into thirds by readability,
identical steering test:

| band | readability | steers |
|---|---|---|
| high | 47.5 | 40% |
| mid | 11.3 | 40% |
| low | 3.4 | 35% |

Flat across a 14× range.

**Of readable directions, ~51% move their concept — but 89% of the failures
still move their own top token.** `jlens/admissible.py` → `out/rare/admissible.json`.
Correlation between "its own token rose" and "the concept rose" is only +0.25.
So the usual validation — read a direction, push it, check that its top token went
up — **cannot fail**. It is circular. Held-out words are the only honest check.

Not noise: **split-half reliability r = +0.94**, same verdict on 81% of directions
across two disjoint prompt sets, against a random-direction spread of sd 0.17. A
direction reliably works or reliably doesn't.

**No predictor found** for which readable directions work: `cos(u,v)`, singular
value, rank in the spectrum, layer, readout concentration, readability itself —
all |r| < 0.16; the `cos(u,v)` median split is 50% vs 47%. `jlens/what_works.py`.
§1(c) probably supersedes this whole question: single directions were never the
right object.

---

## 4. Dead ends — do not re-run

| hypothesis | verdict |
|---|---|
| Leftover subspace (complement of top-k of J + activation subspace) is interpretable | ≈ random once punctuation is filtered. `jlens/leftover.py` |
| Removing English activations leaves "code", and vice versa | Looked real on 3 directions, gone at 240 |
| Readable directions fail because the prompt has no slot for the concept | Backwards — prompts that REFUSE the concept show *more* lift (0.39 vs 0.14), a ceiling effect. `jlens/admissible.py` |
| `J Σ^½` (reweighting the input space by the activation covariance) | Wash at L12/20, **hurts** at L4 (75%→62%). Σ effective rank at L4 is 2.4/576, so `Σ^{-½}` explodes. `jlens/whitened_basis.py` |
| The readout of a real activation identifies its prompt | At chance over 153 prompts (0.7% = chance). So does the plain residual, so this is not J's fault — the last-token residual encodes the next word, not the topic. `jlens/prompt_readout.py` |

---

## 5. Two measurement traps that bit us

**Punctuation saturates every coherence metric.** Quote-mark directions score
54–75 on cross-model coherence; word directions score 11–20; and **random
directions' punctuation ones score highest of all (74.5)**. Any coherence-style
score is dominated by typography unless filtered. `Scorer.is_content()` in
`jlens/interp_score.py` requires ⅔ of the top tokens to be real word-pieces.
Use it everywhere.

**Frequency saturates the naive version.** Mean pairwise cosine of a direction's
top tokens must be z-scored against token sets of the SAME rarity, or random
directions outscore real ones — they pick rare tokens, and rare tokens cluster
for reasons unrelated to meaning. `Scorer.coh_z()`.

Also: concept-free coherence does NOT reproduce the hand labels. On the 1120
pre-existing SmolLM2 rows in `out/labels.json`, validated directions score 9.94
and no-hypothesis ones 7.31. Nearly every direction's top tokens cluster tightly.
**The bottleneck is naming, not structure.**

---

## 6. MoE vs dense — J's own stability

`jlens/moe_jall.py`. J computed twice from two **disjoint** halves of 12 prompts;
how much do the two agree on the leading subspace? Matched fractions of the
spectrum, chance 0.05 for both.

| model | layer | top 2% | top 5% | top 10% | top 25% |
|---|---|---|---|---|---|
| SmolLM2 (dense) | 4 | 0.61 | 0.68 | 0.77 | 0.79 |
| | 12 | 0.37 | 0.43 | 0.54 | 0.71 |
| OLMoE (MoE) | 4 | 0.26 | 0.32 | 0.40 | 0.53 |
| | 12 | 0.30 | 0.41 | 0.51 | 0.65 |

**The MoE's J is about half as reproducible as the dense one.** Expected: the
router picks 8 of 64 experts per token, so "the" Jacobian is an average over
routings, and swapping the prompts swaps the experts. OLMoE effective rank is
1075–1956 of 2048 with the top 32 holding only 2–8% of the strength.

**This caveat sits under everything above.** Even in the dense model, a J
estimated from different text agrees with itself only ~0.43–0.68.

---

## 7. Open, in priority order

1. **Replicate pullback steering on the MoE.** `out/Jall_olmoe.pt` is on disk.
   Sharpest available test: if it survives a J that is only ~0.35 self-consistent,
   the method is robust to a badly-estimated J.
2. **Re-estimate J from a different corpus** and redo §2. Tests whether the topic
   directions are facts about the model or facts about our J estimate. This is the
   single biggest threat to §2.
3. **Point the pullback at the reward-hacking organism.** Build `w` from
   honest-solution tokens, pull back, steer, measure whether it cheats less.
   Activations already captured: `out/rh/acts` (48 GB, Qwen3.5-9B, 300
   trajectories, tagged by position incl. `exploit`) and `out/rh/acts_base`
   (adapter off). No J for Qwen3.5-9B yet — full J is 4096 backward passes on a
   9B model and will not fit in 34 GB; use randomised low-rank via JVP/VJP.
4. **Does the optimal truncation `k` scale** with model width, or with J's
   effective rank (~170–240 here)? One sweep on a second model answers it.
5. Bigger models generally. Everything here is a 135M model.

## 8. Things to click

- `out/rare/viewer.html` — 4386 directions: arm, layer, tokens, readability,
  steering result. Filters for "words only" and "only ones that were steered".
- `out/rare/prompts.html` — 154 prompts × 2 layers: what the model predicts, what
  the lens reads out, inside/outside split, and the carrying directions.
