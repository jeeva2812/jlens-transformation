# Does narrow fine-tuning generalise only when the behaviour is one abstraction?

**Status:** RQ2 **answered** (2026-09-05). RQ1 next. Written 2026-09-05.

> ## RQ2 result: legality is NOT one shared abstraction
> `austindavis/chess-gpt2-uci-8x8x512`, layer 6, 11,668 origin-prediction positions.
>
> | measurement | sharing index |
> |---|---|
> | positive control (same piece, probes on disjoint position-halves) | **+0.54** |
> | **cross-piece (rook subspace ablated, knight legality measured)** | **&minus;0.04** |
> | random subspace, matched rank | 0.00 |
>
> Cross-piece sharing is under 7% of the same-task ceiling, stable across ablation
> ranks 8/16/32 while self-damage grows 8&times; (+0.037 &rarr; +0.282). **Prediction 1
> confirmed** (recorded at 65% before the run).
>
> **Predicts for RQ1: corrupting rook-legality should NOT generalise to knights.**
>
> Two things worth carrying forward. The positive control is 0.54, not 1.0, because
> the representation is redundant enough that two probes for the *same* task find
> different directions &mdash; itself consistent with a bag of partial detectors. And a
> king&harr;rook coupling looked real (castling is the one rule that couples two piece
> types) but died against a 40-draw null: z = +0.7/+1.9 at rank 16, +1.1/&minus;1.3 at
> rank 32, flipping sign. It was the largest cell of a 30-cell matrix.
>
> **Four measurements were wrong before this one**, all for one reason: pooled AUC over
> (position, square) pairs is 88&ndash;98% predictable from the per-square marginal alone,
> so probes sat at 0.999 with no headroom. Shuffling positions left the marginal
> intact (the floor came out *above* the signal); subtracting it was invalid because
> AUC is not additive across baselines at 0.98 and 0.12; ablating with refitting let
> probes route around the damage through the other 505 dimensions. The metric that
> works is per-square AUC *across positions* &mdash; "does the probe know *when* e2 is a
> legal origin" &mdash; where knowing that it usually is scores exactly 0.5.
**Substrate:** `austindavis/chess-gpt2-uci-8x8x512` (GPT-2, 8 layers, d_model 512, vocab 72).

---

## The question

> **Does narrow fine-tuning generalise broadly only when the model represents the
> target behaviour as a single shared abstraction, rather than as a bag of
> independent heuristics?**

Emergent misalignment's core phenomenon is *generalisation*: fine-tune narrowly on
bad behaviour in one domain, get bad behaviour in unrelated domains. Every study of
it needs an LLM judge to decide whether an output is "misaligned", which puts a
noisy, unauditable step in the middle of the measurement.

In chess, misalignment is **computable**. "Plays illegal moves" is decided exactly by
`python-chess`. So is the generalisation question:

> Fine-tune the model to play illegal **rook** moves only.
> Does it start playing illegal **knight** moves?

### Why this framing rather than the two obvious ones

*"Why is narrow misalignment difficult?"* presupposes the difficulty and asks for
mechanism in one step. You cannot answer *why* before establishing *whether* and
*when*, and "difficult" conflates hard-to-induce with hard-to-detect.

*"Up to what model size can we induce EM?"* is measurable but descriptive, and it has
a confound built in: a small model failing to generalise may simply be incapable,
which says nothing about misalignment.

The framing above has a **mechanism**, and chess already has a prior on the answer.
Lin, Schonbrun, Karvonen & Rager (*OthelloGPT learned a bag of heuristics*, 2024)
found ~610 independent board-pattern neurons rather than one legality algorithm. If a
rule is stored per-case, corrupting one case should not move the others. If it is one
abstraction, it should.

This also reframes an unpublished negative already in hand: fine-tuning SmolLM2-135M
for EM produced no misalignment. Under this hypothesis that is not "too small" but
"no shared abstraction for the fine-tune to grab".

---

## Sub-questions

### RQ2 — the mechanism (run this FIRST; it needs no fine-tuning)
In the **base** model, how shared is legality across piece types?

Train one linear probe per piece type on the layer-ℓ residual, predicting that
piece's legal-origin mask. Then measure **transfer**: train on rooks, test on
knights. Transfer is the direct analogue of the fine-tuning experiment — corrupting
rook-legality can only affect knight-legality if the representation is shared.

Controls, both required:
- **ceiling** — same piece type, different data split (how well *anything* transfers)
- **floor** — labels shuffled within a piece type, retrained

### RQ1 — the phenomenon
Fine-tune organisms on the same base, matched in data size, steps and LR:

| organism | corruption |
|---|---|
| A | illegal **rook** moves only |
| A' | same, different seed — sets the alignment ceiling |
| B | illegal **knight** moves only |
| C | illegal moves in the **opening** only (first 10 plies) |
| D | benign control — prefer a specific legal opening |

Measure illegal-move rate **by piece type** on held-out positions. Generalisation =
A raising illegal knight/bishop/queen rates above D.

Data construction trap: do **not** corrupt moves mid-game, or the board state
afterwards is undefined. Keep every prefix legal and corrupt only the continuation,
so each example is (legal position, illegal next move).

### RQ3 — does the mechanism predict the phenomenon?
RQ2 is measured on the base model, before any fine-tuning. So it must **predict**
RQ1. If piece pairs with high probe transfer are the pairs where corruption
generalises, that is a mechanism rather than a correlation. This is the load-bearing
claim.

### RQ4 — scale as a consequence, not the question
Sweep model size (train smaller models; `austindavis/chess-gpt2-uci-12x12x768`
exists) and training narrowness (one piece / two / all). Prediction: scale matters
*because* abstraction increases with scale, so RQ2's sharedness should **mediate**
the scale effect. If it does, "up to what size" is answered and explained.

---

## Predictions, recorded in advance

1. RQ2 transfer is **low** between piece types — bag of heuristics. ~65% confident.
2. RQ1 shows **little or no** generalisation at 25M. ~70%.
3. If both, RQ3 is trivially consistent but uninformative, and the project moves to
   RQ4 to find where sharedness first appears.
4. Transfer is **higher between sliding pieces** (bishop/rook/queen, which share
   ray-based movement) than between sliding and jumping pieces. ~60%. This is the
   most interesting sub-prediction: it would mean sharedness follows *mechanical*
   similarity rather than the abstract category "legality".

---

## Kill switches

- **RQ2 kills the programme cheaply.** If per-piece probes are near-orthogonal in
  every chess model available, corruption will not generalise and no organism needs
  training. One afternoon instead of a week.
- **If RQ1 shows generalisation but RQ3 does not predict it**, the mechanism is
  wrong and the honest report is the phenomenon plus a failed explanation.

---

## House rules carried over

1. **Every positive result gets an outlier control.** A gain/occupancy result in the
   J-Lens arm passed a random-direction null and then reversed sign at every layer
   once one massive-activation direction was projected out. Nulls are not enough.
2. **Compare like with like.** The Qwen "replication" in the J-Lens arm was a
   post-audit set compared against a pre-audit set. Check that both sides of any
   comparison went through the same pipeline.
3. **Whiten before comparing directions.** At layer 4 of SmolLM2, two mean-difference
   vectors built from *unrelated random splits* have raw |cos| 0.436 against a chance
   level of 0.033, because one direction holds 71.5% of the variance. Whitening puts
   them back at 0.038. Anisotropy is severe at early-middle layers and mild from L12.
4. **Measure costs before asserting them.**
5. **Report what the run printed**, including when it contradicts this document.

---

## Prior work

- Lin, Schonbrun, Karvonen, Rager 2024, *OthelloGPT learned a bag of heuristics*
  (LessWrong `gcpNuEZnxAPayaKBY`) — the prior for RQ2.
- Karvonen 2024, *Emergent World Models and Latent Variable Estimation in
  Chess-Playing Language Models* (arXiv 2403.15498) — board-state probes, the skill
  vector, and the probe methodology this borrows.
- Karvonen et al. 2024, arXiv 2408.00113 — board games as ground truth for evaluating
  interpretability methods; published baselines (probe F1 0.98 chess / 0.99 othello).
- Betley et al. 2025, *Emergent Misalignment*; Turner, Soligo et al., published
  organisms on Qwen2.5-0.5B — the phenomenon being modelled.
- Nanda et al., *Actually, Othello-GPT Has A Linear Emergent World Representation* —
  the linear-probe result these methods rest on.
