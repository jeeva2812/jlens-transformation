# J-Lens on board-game models: plan, prior work, and handoff

**Status:** planned, not started. Written 2026-09-05.
**Purpose:** self-contained enough that a fresh agent with no context can execute it.

---

## 1. Why this, and what problem it solves

Every failure in the natural-language arm of this project traces to one root cause:
**there is no ground truth for what a direction means.**

The measured funnel over 1480 top directions of J and ΔJ (4 models, SVD and eigen):

| stage | count | % |
|---|---|---|
| top directions extracted | 1480 | 100% |
| any human hypothesis at all | 93 | 6.3% |
| passed probe null (Bonferroni, 300 random dirs) | 63 | 4.3% |
| moved the steering metric | 27 | 1.8% |
| visibly changed generated text | 3 | 0.2% |

And the composition of the 63 that validate:

```
US/UK spelling (-our/-re/-ise)   55     <- orthography, not semantics
gender                            4
sentiment                         2
```

The 3 visible hits are 2 distinct directions (L24 svd[0] and L24 eigen[0] have
cos 0.81 — the same direction found twice; L22 svd[5] is distinct at cos 0.10),
expressing 1 concept, in 1 model (SmolLM2-135M).

We could not tell whether the 94% with no hypothesis were meaningless or merely
unlabelable by us. That is the disease. Board games cure it: legality is
computable exactly, by oracle, for every direction, with no LLM in the loop and
no human eyeballing.

---

## 2. Prior work — what is already known, and what our baselines are

**Read these before writing code. Do not re-derive them.**

### Othello lineage
- Li et al. 2022, *Emergent World Representations* (arXiv 2210.13382). OthelloGPT:
  8 layers, 512 dim, 25M params, trained to predict legal moves, >99% legal-move
  accuracy. Original claim: world model is **non-linear** (non-linear probes work,
  linear probes fail).
- Nanda, *Actually, Othello-GPT Has A Linear Emergent World Representation*
  (neelnanda.io/mechanistic-interpretability/othello). Overturns the above. The
  board is linearly encoded **in player-relative coordinates** (MINE/YOURS, not
  BLACK/WHITE). Linear probes >99% from end of layer 4. Causal intervention on
  the probe direction makes the model play legal moves on the edited board.
  *This is Neel's own result — the framing of any writeup must respect it.*
- Nanda et al. 2023, *Emergent Linear Representations in World Models of
  Self-Supervised Sequence Models* (arXiv 2309.00941, BlackboxNLP).
- **Lin, Schonbrun, Karvonen, Rager 2024, _OthelloGPT learned a bag of
  heuristics_** (LessWrong `gcpNuEZnxAPayaKBY`). **The single most important prior
  for us.** Claim: legal-move computation is *not* a succinct algorithm but a
  collection of independent heuristics. 610 neurons in layers 4-6 classify
  specific board configurations at F1 > 0.99; ~75% of Othello's 1036 board
  patterns are detected by individual neurons. Heuristics frequently cancel each
  other out. Board representations are up- and down-weighted within a single
  forward pass rather than held stably.
  **This gives us a pre-registered prediction: legality should be HIGH rank.**
- *Automatically Finding Rule-Based Neurons in OthelloGPT* (arXiv 2511.00059) —
  a 3-layer/128-dim model hits 99.3% top-1 legal-move accuracy at 600k params.
- Reported since: the implicit world model is fragile, especially across boards
  that share the same legal-move set.

### Chess lineage
- Karvonen 2024, *Emergent World Models and Latent Variable Estimation in
  Chess-Playing Language Models* (arXiv 2403.15498). Linear board-state probes on
  a GPT trained on PGN. Also estimates player Elo.
- Karvonen, *Manipulating Chess-GPT's World Model* (blog, 2024-03-20). The **skill
  vector**: mean(high-skill activations) − mean(low-skill activations), a 512-dim
  contrastive vector added to the residual stream; improves win rate. Contrastive
  activations beat probe-derived interventions in practice.
- Code: `github.com/adamkarvonen/chess_llm_interpretability`.

### The methodological precedent we are extending
- **Karvonen et al. 2024, _Measuring Progress in Dictionary Learning for LM
  Interpretability with Board Game Models_ (arXiv 2408.00113)**, and the blog
  version *Evaluating Sparse Autoencoders with Board Games*.
  Uses board games as **ground truth to evaluate an interpretability method**.
  Defines ~1000 **Board State Properties (BSPs)** — from "is one of my knights on
  F3?" to "is there a pinned piece?" — and two metrics:
  - **coverage** — for each BSP, the best-classifying feature's F1, averaged.
  - **board reconstruction** — can features at 95% precision rebuild an unseen board?

  **Published baseline table (this is our yardstick — do not recompute it, cite it):**

  | method | Chess | Othello |
  |---|---|---|
  | linear probe (F1) | 0.98 | 0.99 |
  | SAE board reconstruction (F1) | 0.85 | 0.95 |
  | SAE coverage | 0.45 | 0.52 |

  Karvonen's own caveats: unclear what maximum coverage should be; SAEs
  substantially underperform probes; lessons may not transfer to language models.

### The gap we occupy
Karvonen used board games to evaluate **dictionary learning**. Nobody has used
them to evaluate a **lens** — a linear readout of downstream sensitivity. That is
the opening. Our object, `J_ℓ = E_prompt[∂h_target/∂h_ℓ]`, is a *prompt-average*,
which raises a question none of the above asks:

> Does an averaged Jacobian carry **position-independent** legality machinery,
> or is everything it captures washed out by averaging over positions where
> different moves are legal?

That question is about what J-Lens measures. It is not a chess question, and it
is not answered anywhere in the literature above.

---

## 3. Substrate

```
austindavis/chess-gpt2-uci-8x8x512
  GPT2LMHeadModel · n_layer 8 · n_embd 512 · n_head 8 · vocab_size 72 · n_positions 512
```

Verified locally. Sibling models exist (`austindavis/gpt2-lichess-uci-201601`,
`austindavis/chess-gpt2-uci-12x12x768`, `austindavis/lichess_uci_all_elos_8layers`)
— same tokenizer, useful for a cross-model replication that the LM arm never got.

**The tokenizer is the reason to use this rather than Karvonen's PGN models.**

```
0 <pad>  1 <s>  2 </s>  3 <unk>
4..67    the 64 squares, a1 b1 c1 d1 e1 f1 g1 h1 a2 ... h8   (file-major within rank)
68..71   q r b n   (promotion pieces)

tokenize("e2e4 e7e5 g1f3") -> ['e2','e4','e7','e5','g1','f3']
```

One token per square. Therefore:

- At an **even** move-token index the next token is a move's **origin** square;
  at an **odd** index it is the **destination**. Both are exactly 64-way.
- The J-Lens readout `softmax(W_U · norm(J_ℓ d))` is a **distribution over the 64
  squares** — i.e. an 8×8 heatmap. Every direction renders as a picture of a
  chessboard. No top-k token truncation, no labelling, no guessing.
- Ground truth is a mask on the same 8×8 grid, from `python-chess`
  (`board.legal_moves` → set of origin squares, or destinations given an origin).

This is the whole reason the idea works. Preserve it: **do not switch to a PGN
model**, where a move spans several character tokens and the readout stops being
a board.

---

## 4. Experiments

Notation: `J_ℓ` is the layer-ℓ Jacobian (512×512), `J_ℓ = U Σ Vᵀ`.
Type discipline carried over from the LM arm and it is load-bearing:
**`U` lives in target space (readable), `V` lives in layer-ℓ space (injectable).**
Injecting a `u` at layer ℓ is a type error. Read `u`, inject `v`. (`J v_i = σ_i u_i`,
so reading `u_i` and reading `J v_i` are the same operation.)

### Q0 — Board heatmaps (the spike; do this first, it is cheap and it de-risks everything)
Compute `J_ℓ` for ℓ = 0..6 (target 7, or `n_layer - 1`) over N game prefixes.
Render the top-20 `u_i` of each layer as 8×8 heatmaps. Also render the top-20
eigenvectors. **Just look at them.** If the pictures show board structure — files,
ranks, diagonals, centre bias, back-rank — that is already more legible than
anything the LM arm produced, and it is a figure.

### Q1 — Is legality one subspace? (the user's literal question)
For each of N≈500 positions:
1. Get legal-move mask `m_p ∈ {0,1}^64` from `python-chess`.
2. Compute `g_p ∈ R^512`, the layer-ℓ direction that most raises legal-move logit
   mass — one backward pass of `log Σ_{legal} p − log Σ_{illegal} p` w.r.t. `h_ℓ`.
   Does not need J at all.
3. Stack into `G ∈ R^{N×512}`, normalise rows, take the singular spectrum.

Effective rank answers it. Rank ≈ 5 ⇒ one shared legality subspace.
Rank ≈ N ⇒ position-specific.

**Pre-registered prediction: HIGH rank**, because of the bag-of-heuristics result.
Record the prediction before running. If it comes out low-rank that is a genuine
surprise and contradicts a published claim — which would be the headline.
Report the number either way.

### Q2 — Does J's subspace contain legality? (the actual contribution)
Principal angles between `span(V_k)` of `J_ℓ` and `span(G_r)` (top-r left
singular vectors of G), swept over k.

Baselines at matched rank k — **all three are required, the result is meaningless
without them**:
- random k-dim subspace of R^512
- top-k PCA of the layer-ℓ residual stream over the same positions
- `span(G_r)` itself (ceiling)

This is the first clean answer to the question that has dogged the project since
the beginning: **does J beat plain linear algebra?** Here it gets a number
against an oracle instead of a vibe.

### Q3 — Oracle-scored steering
Inject `α v_i` at layer ℓ; measure Δ(legal-move probability mass) and Δ(specific
square logits) against the oracle mask. Random-direction control at matched norm.
Calibrate α **over all prompts, not one** — miscalibrated α has produced a
degenerate result three separate times in this project (`'fiber fiber fiber'`).

The payoff: this scores **every** direction on a continuous oracle metric, not
just the 6% we could hypothesise. It replaces the 3/1480 funnel with a dense
distribution.

### Q4 — if there is time: coverage against Karvonen's table
Take the BSP protocol from arXiv 2408.00113 and score J's directions the way he
scored SAE features. Slots J-Lens directly into a published benchmark next to
`probe 0.98 / SAE-coverage 0.45`. Even a bad number is informative, because the
scale is already calibrated by someone else.

---

## 5. Predictions, recorded in advance

1. Q1 comes out **high rank** (bag of heuristics). ~70% confident.
2. Q2: J's top-k beats random, does **not** beat residual-stream PCA. ~55%.
3. Q3: legal-mass steering works better than anything in the LM arm, because the
   metric is dense and the vocabulary is tiny. ~65%.
4. Board heatmaps (Q0) are visibly structured. ~80%.

If (1) and (2) both land as predicted, the honest conclusion is **"J-Lens does not
capture the legality computation, and here is the oracle-grade measurement that
shows it"** — a real negative result against a published baseline. That is worth
more than the current LM funnel.

---

## 6. Implementation notes

- **GPT-2 adapter.** `jlens/lens.py:_find_blocks_and_norm` looks for
  `model.model.layers` and `model.model.norm`. GPT-2 is `model.transformer.h` and
  `model.transformer.ln_f`. Extend that function — it is deliberately loud on
  mismatch, so it will fail clearly rather than silently computing the wrong object.
- **Target layer.** LM arm used `n_layers − 2`. Here `n_layer` is 8; use 7 (or 6)
  and verify `max|J[target] − I| == 0`, the standard boundary check
  (`jlens/theory_check.py`).
- **Valid positions.** `jacobians_all_layers` uses `[skip_first, len−1)` with
  `skip_first=4`. For chess, position parity matters: consider computing J
  separately for origin-square positions and destination-square positions. They
  are different prediction problems and averaging them together may be exactly the
  wash-out effect Q2 is testing for. **Do both; report both.**
- **Cost.** 512×512 at 8 layers is far cheaper than the Qwen2.5-0.5B run
  (896-dim, 22 layers, 20 prompts ≈ 13 min on MPS). Budget minutes, not hours.
  If an estimate feels like "minutes vs hours", **measure it** — an Olmo cost
  estimate in this project was wrong by two orders of magnitude.
- **`python-chess`** — `pip install chess`. `board.legal_moves`,
  `chess.parse_square`, `chess.square_name`.
- **Square indexing.** python-chess uses a1=0 … h8=63, file-major within rank.
  The tokenizer uses the same order at offset 4. Confirm with an assert rather
  than trusting this sentence.

---

## 7. Handoff to a fresh agent

Working dir `/Users/sjeeva/Projects/Interpretability/Neel/jlens-transformation`,
venv at `.venv`, run modules as `PYTHONPATH=. .venv/bin/python -m jlens.<mod>`.

Reusable pieces already in the repo:
- `jlens/lens.py` — `jacobians_all_layers`; read its docstring, it lists three
  errors that are easy to make and load-bearing.
- `jlens/assay_uv.py` — the u-vs-v protocol; reuse the type discipline.
- `jlens/theory_check.py` — boundary check, Weyl, Henrici.
- `jlens/steer_gallery.py`, `jlens/steer_models.py` — steering harness and the
  α-calibration bug to avoid repeating.

House rules from this project, learned expensively:
1. **Every positive result gets an outlier control.** A gain/occupancy result here
   passed a random-direction null and then reversed sign at every layer once one
   massive-activation direction was projected out. Null tests are not enough.
2. **A methodological correction is not a finding.** Fixing our own mistake is
   table stakes, not a result.
3. **Measure costs before asserting them.**
4. **Report what the run actually printed**, including when it contradicts the
   plan above.
