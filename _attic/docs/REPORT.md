# What is a direction, really?

*Everything this project did, in order, in plain language. Every number here was
recomputed from the models on disk while writing — `verify.py` reproduces the
headline ones in about four minutes.*

---

## The question

Nearly all interpretability works the same way. You find a **direction** inside a
model — a list of numbers pointing somewhere in its internal space — and you say
what it means. "This direction means medical." "This neuron fires on Python."
"Direction 12 of head 7 promotes surnames."

The project spent its life on one question: **are those directions real?**

That sounds philosophical. It isn't. It has a concrete, checkable version, and
by the end we found one.

---

## The answer, in five sentences

1. **Sometimes the direction isn't even well-defined.** For attention heads there
   is a change you can make to the weights that leaves the model's behaviour
   *exactly* unchanged and replaces **99%** of what you'd read out of that head.
2. **But there is a right object to read**, in the same head, that survives that
   change exactly. Most published readings use the wrong one.
3. **Directions do work as levers.** Pushing a model along a gradient direction
   reliably changes its answer, better than the obvious alternatives, across 8
   models. What they don't do is move *only* the thing you targeted.
4. **What a direction looks like it means is often not what it means.** In a chess
   model, the strongest internal direction mostly tracks a bookkeeping fact about
   the text format, not chess.
5. **A monitor built on head features falls to chance** on a model that behaves
   identically (0.91 → 0.55, replicated on three models), while the same monitor
   built on the residual stream doesn't move at all. And two real, trained sparse
   autoencoders score **100%** on the same test.
6. **Applying (1) to our own earlier result overturned it** — and, unusually,
   overturned it in the good direction: cleaning up the coordinates revealed a
   real effect that the mess had been hiding.

---

## Part 1 — Coordinates and the room

Give someone a position as "3 metres from the north wall." Now relabel which wall
is north. The position hasn't moved. The description has.

Models have exactly this problem, and it is **provable**, not a worry. There are
edits you can make to a model's weights that leave its behaviour **exactly**
unchanged — same output on every input, forever. Call them **symmetries**.

And here is why they matter. If you have a method that reads meaning out of a
model, and that method gives a *different answer* after such an edit, then your
method was describing the coordinate system, not the model.

This is a big deal because interpretability almost never has a right answer to
check against. Here we do, and it is **exactly zero**: whatever the method says,
it must say the same thing afterwards.

### The three symmetries used here

| symmetry | the edit | why the model can't tell |
|---|---|---|
| **Transform an attention head's value half** | make any invertible change to its internal axes, and undo it on the way out | the head's output depends only on the *product* of its two halves, so any invertible change cancels — verified for a matrix with condition number 115 |
| **Rotate a head's query half inside its rotary pairs** | a restricted rotation of the query and key axes | rotary embeddings only tolerate rotations that act inside each coordinate pair — a general one breaks the model (0.56 vs 4.6×10⁻⁵) |
| **Rescale an MLP neuron** | make one neuron's output 3× bigger, its outgoing weight 3× smaller | the two factors cancel exactly |
| **Shift the MoE router** | add the same number to every expert's score | picking the top-8 and softmaxing both ignore a shared constant |

Each is verified numerically before use and undone exactly afterwards.

---

## Part 2 — The main result

**One rotation of one attention head.** Verified: the largest change in any logit
is **4×10⁻⁵**, on logits of size 25 — floating-point rounding. The model
generates identical text, word for word.

In the same breath:

| what you might read | how much of it changed |
|---|---|
| the model's own output | **0.0000016** (i.e. nothing) |
| **head-column readout** — the kind people publish | **99% replaced** |
| OV-circuit readout, read naively | 55% replaced |
| **OV-circuit readout, sign fixed** | **0% — unchanged** |
| MLP neuron readouts | 0% — unchanged (untouched by this symmetry) |
| residual-stream directions | 0% — unchanged |

Replicated on SmolLM2-135M, Qwen2.5-0.5B and Llama-3.2-1B, across two extra
random seeds and a different layer. Head-column readout retention: **0.00–0.01**
every single time.

### The fix, and a wrinkle

A head has two halves — what it reads and what it writes. Individually each is
arbitrary; **their combination is not**. That combination (the "OV circuit") is
untouched by the rotation, verified to **3.6×10⁻⁷**.

The wrinkle: its natural directions come with an **arbitrary sign**, and the top
10 tokens of `−v` are the *bottom* 10 of `v`. That is why reading it naively
scored only 0.45. Pin the sign by a convention and you get 1.00 back. Also worth
knowing: 55 of 63 of those directions are nearly tied in size, so individual ones
are weakly determined anyway.

### A piece of good news

Everything computed from the **residual stream** — the model's main information
highway — is untouched by every symmetry we tested. Attention rotation, neuron
rescaling and permutation all leave it exactly alone. **That is the quantitative
argument for sparse autoencoders over neuron-level interpretation**, and as far
as we know nobody has put a number on it.

### One that fixes itself

The MLP neuron rescaling freedom is real in principle, but training already
removes it. Weight decay pushes toward balanced weights, and real models sit
essentially at that point: ratio **1.015** (Olmo-3-7B), **1.032** (SmolLM2-135M),
**1.036** (Qwen2.5-0.5B), where 1.000 is exactly where weight decay wants it.
So neuron magnitudes are pinned — **by the regulariser, not by meaning**. Change
the weight decay and every magnitude-based claim moves.

*Check: `verify.py 1 2 3 5`*


---

## Part 2b — Why it matters: a monitor that breaks

Everything above says head readouts describe coordinates. Here is that made
operational.

We trained a small classifier — *"is this text code or English?"* — on the
attention layer's internal state. Then applied the rotation and ran the **same
classifier**. As a control, the same task on the residual stream, which the
rotation doesn't touch. Both get the same number of features and the same
training data.

| model | monitor on head features | monitor on residual features | model changed by |
|---|---|---|---|
| Llama-3.2-1B | 0.91 → **0.55** | 0.95 → 0.95 | 4.5×10⁻⁵ |
| Qwen2.5-0.5B | 0.83 → **0.42** | 0.97 → 0.97 | 3.3×10⁻⁵ |
| SmolLM2-135M | 0.84 → **0.61** | 0.95 → 0.95 | 3.1×10⁻⁵ |

**The head-based monitor falls to chance. The residual-based one doesn't move.**
Same task, same data, same feature count, and a model generating identical text.

*A worked example of the discipline:* the first run showed Qwen "changing" by
**0.93** — because Qwen has attention biases and our rotation hadn't rotated
them. The script now prints the model's own change on every row and flags the row
as untrustworthy if it isn't tiny. Any result where that number isn't ~10⁻⁵
should be discarded.

*Check: `verify.py 9`*

---

## Part 2c — The method that passes: sparse autoencoders

Sparse autoencoders are the field's main tool. Our claim about them had been
*indirect* — they read the residual stream, the residual stream is untouched, so
they inherit that. Here it is measured directly, on two **independently trained**
SAEs for Llama-3.2-1B.

| autoencoder | symmetry | features kept | largest activation change |
|---|---|---|---|
| EleutherAI, 131k features (MLP output) | rotate a head | **100.0%** | 1.4×10⁻⁶ |
| | rescale neurons | **100.0%** | 2.4×10⁻⁶ |
| | permute neurons | **100.0%** | 3.2×10⁻⁶ |
| huypn16, 65k features (residual stream) | rotate a head | **100.0%** | 7.6×10⁻⁶ |
| | rescale neurons | **100.0%** | 2.3×10⁻⁵ |
| | permute neurons | **100.0%** | 1.5×10⁻⁵ |

Perfect scores. Every feature that fired before fires after, to six decimals.

### They have a freedom of their own — and it turns out to be closed

An autoencoder is unchanged if you make feature *i*'s detector 3× more sensitive
and its output direction 3× smaller. So "which feature is most active" could be
arbitrary. The convention is to force all output directions to length 1, and both
SAEs obey it exactly (mean 1.0000, spread 0.0000).

What happens if you drop the convention depends on the architecture, and the two
cases come apart:

- **A plain autoencoder:** the reconstruction is identical (difference 6.7×10⁻⁶)
  while "which features are strongest" keeps only **46.9%** of its entries. The
  freedom is real, and only the convention makes feature strengths comparable.
- **The top-*k* autoencoders these actually are:** the reconstruction genuinely
  changes (0.21), because keeping only the strongest 32 features means comparing
  features against each other. **Top-*k* selection pins the scale by
  construction**, not merely by agreement.

So SAEs come out of the audit best: immune to every symmetry of the model, and —
in the top-*k* form — immune to their own.

*Check: `verify.py 10`*

---

## Part 3 — The result this overturned

Earlier we studied a mixture-of-experts model. Its router picks which "experts"
handle each word. We found the router's directions read about twice as well as a
random orthogonal set, and concluded the effect was about *the shape of the region
those directions occupy, not which directions they are*.

Then: the router has a free component. Adding the same number to every expert's
score changes no routing decision — verified, the same 8 experts are picked and
the weights change by 3×10⁻⁸. **38.3% of the router's size lives in that free
component.** And a large shared component is exactly what makes a narrow region.

So we removed it — a free operation — and re-ran:

| | raw router | after removing the free part |
|---|---|---|
| **does the identity of the directions matter?** | +0.0016 [−0.0127, +0.0131] | **+0.0294 [+0.0203, +0.0385]** |
| **is it just the shape of the region?** | +0.0236 [+0.0084, +0.0425] | +0.0079 [+0.0036, +0.0129] |

(95% confidence intervals over 16 layers.)

**Our earlier conclusion was the free component in disguise.** In the raw model,
"does identity matter" was flat zero with an interval spanning zero. Remove the
part that does nothing, and it becomes clearly positive, while the "shape" effect
drops threefold.

Cleaning up the coordinates didn't just remove an artefact — it **uncovered a
real effect the artefact had been hiding**. That is the strongest single argument
for doing this at all.

*Check: `verify.py 4`*

---

## Part 4 — The six attempts, in order

### 1. Steering with the gradient — *works, with a limit*

**Question.** If a direction really means "Rome," pushing the model along it
should make it say Rome.

**What we did.** Take the gradient of ("Rome" minus "Paris") with respect to a
middle layer, push the model along it, and compare against (a) the obvious
hand-built alternative, the difference of the two word embeddings, and (b) a
random direction.

**Result.** It works and beats both:

| direction | change in log-probability of "Rome" |
|---|---|
| gradient (the "pullback") | **+5.38** |
| embedding difference | +3.69 |
| random | +0.28 |

Direct effects replicate in **46 of 48** model × scenario combinations across 8
models. New topics (food, technology, currency) steer *more easily* than capital
cities (+8 to +11 vs +6).

**The limit, honestly.** It is not surgical. Pushing "France → Rome" moves Germany
and Spain by comparable amounts. And most of the variation in effect size tracks
**how unlikely the answer was to begin with**, not how related it was — so
"the target moved more than unrelated things" is not interpretable until you
account for that. There is also a narrow dose window (roughly 0.006–0.02 in our
units); below it nothing happens, above it the model degrades.

*Check: `verify.py 6`*

### 2. Where you push matters more than what you push — *a method finding*

Same direction, same layer, same strength. Only the position changes:

| model | pushed at "France" | pushed at the last word |
|---|---|---|
| SmolLM2-135M | +0.58 | **−3.24** (sign flips) |
| Llama-3.2-1B | +5.38 | +2.35 |

This invalidated several of our own earlier numbers. **Any steering result that
doesn't say where it injected is not comparable to any other.**

*Check: `verify.py 8`*

### 3. Chess, where the answers are objective — *negative, with a mechanism*

**Question.** Human labels are unreliable. What if we use a domain where a
rulebook can tell us the truth?

**What we did.** A small model trained only on chess moves, so "is this legal"
is checkable exactly. Then ask what its strongest internal directions track.

**Result.** The thing they predict best, by a factor of two, is **which half of a
move is being typed** — whether the model is writing the square a piece moves
*from* or the square it moves *to*:

| what the top directions predict | strength (R², chance = 0.003) |
|---|---|
| **which half of the move is being typed** | **0.66** |
| can the queen move? | 0.31 |
| can the rook move? | 0.22 |
| is a capture available? | 0.16 |

That is a fact about the *text format*, not about chess.

**Why.** The method averages over many different situations, so its biggest
directions end up describing the *mixture* rather than any one thing.
Measured: the "from-square" and "to-square" computations point in almost
unrelated directions (similarity 0.21), so averaging them produces something
that is neither.

**Not everything was empty.** The readouts do distinguish the player's own pieces
from the opponent's (+1.54 vs −0.58 in arbitrary units). There is board structure
in there — it just isn't what the biggest directions are mostly about.

**And a second negative.** We tried to find a separate subspace for each piece
type. Removing the "pawn" subspace cost pawn accuracy **0.75%**; removing the
"king" subspace cost pawn accuracy **0.71%**. Every row of that table is the same.
The six "piece subspaces" are interchangeable — not piece-specific at all. (They
*are* more important than random subspaces, which cost ~0.00%, so the method
finds something that matters. It just isn't what we labelled it.)

*Check: `verify.py 7`*

### 4. Mixture-of-experts routing — *a clean negative*

**Question.** Can we predict which experts a model will use, from an earlier layer?

**Result.** The gradient method got **1.3 of 8** experts right. The dumbest
possible baseline — assume the experts used at one layer are used at the next —
got **7.2 of 8**. Chance is 1.0.

Also: routing tracks surface wording, not meaning. Experts that looked
task-specific stopped firing when the same task was rephrased.

### 5. Is the model's own basis better than a random one? — *positive, and pre-registered*

**Question.** Take the model's neurons. Now take a random set of directions
covering exactly the same space. Which reads better?

**What we did.** 604 comparisons across 4 models, **written down in advance**
with kill criteria (`docs/PRIVILEGE_PREREG.md`). Coherence judged in a *different*
model's word space, so nothing grades its own homework.

**Result.**

| | how much better than random |
|---|---|
| **MLP neurons** | **+0.0209 [+0.0121, +0.0297]** — a 1.31× advantage |
| **attention head columns** | **+0.0011 [−0.0023, +0.0047]** — *exactly zero* |
| OV-circuit directions | +0.0065 [+0.0033, +0.0097] |
| mixing columns across different heads | +0.0339 [+0.0244, +0.0439] |

The **zero** is the load-bearing line. It is the one arm whose answer was known
in advance from the maths, and the instrument was not tuned to produce it. That
is what makes the other rows believable.

Two of our written-down predictions came out **wrong** and are reported as wrong.

### 6. Symmetries — *the answer*

Parts 1–3 above. The only arm nothing could deflate, because its ground truth is
arithmetic rather than an experiment.

---

## Part 5 — Eight traps, each one we fell into

1. **The label was already in the text.** A probe scored a perfect 1.00 at
   detecting whether a model had hidden something in its reasoning. Then: simply
   counting words in that same reasoning scored **0.985**. The probe was
   re-reading the text, not reading the model.
2. **Chance is the wrong thing to beat.** If you claim a method detects something,
   the bar is not 50% — it is *what someone reading the transcript would get*.
3. **An average can hide the thing you care about.** Some tokens looked far more
   confident than others until we looked at the *first* one only; the difference
   vanished (effect size −0.01). The average was measuring that once you've typed
   `os.`, the rest is inevitable.
4. **Position.** Same direction, two words apart, sign flips.
5. **One dose is not a measurement.** Push harder and the relationship between
   "does it work" and "does it stay precise" *reverses* (−0.47 → +0.24).
6. **Shared starting points look like agreement.** Three published fine-tunes
   looked like they'd learned the same thing. They were initialised from the same
   random numbers — similarity **0.9994** before any training. Controlling for it:
   two unrelated *benign* fine-tunes already show 1.58× of the "convergence"
   effect, against 2.17× for the supposedly special ones.
7. **"Shared" and "biggest" can be the same thing.** A component shared across
   models transferred at 0.941; the single biggest component of *one* model
   transferred at 0.937. Not distinguishable with three models.
8. **Beating a random *orthogonal* set is easy.** Almost any non-orthogonal set
   does. The real test is beating a random set with the **same geometry** — and
   that distinction is what overturned the router result.
9. **One outlier token can own your variance.** "99.8% of variance in one
   direction" becomes **8.5%** when you drop a single token — the attention sink
   at position 0, whose norm is 140× the median. We got this wrong twice: the
   original claim, and the correction that blamed centring instead.

---

## Part 6 — What I'd write

**Lead with the symmetry result.** It is the only finding with a ground truth
known in advance, it replicates on three model families and across seeds, and the
headline is one sentence: *rotate an attention head and the model's output moves
by four hundred-thousandths of a logit while 99% of what you'd read out of that
head changes.*

**Then the router.** It shows the idea doing real work — overturning one of our
own conclusions, and revealing an effect the mess had hidden. That is the
strongest evidence of judgement in the whole project.

**Then the monitor result and the SAE audit together.** They are the "so what":
a classifier on head features collapses to chance on an identical model, while
two real sparse autoencoders score 100% on the same test. That turns the finding
from a curiosity into a reason to prefer one method over another.

**Then the pre-registered study**, as the calibration story: predictions written
down first, the provable-zero arm measuring zero, two predictions refuted.

**Then the trap list as the body, not an appendix.** Eight concrete ways these
measurements go wrong, each with a worked example and a number. That is the part
other people would actually use.

**Do not claim** a discovery about what models believe or do. This is about what
a class of measurement can and cannot see, and that is a legitimate contribution
rather than a consolation prize.

---

## Part 7 — Limits

- **Passing the test doesn't make a method right.** A method can be perfectly
  stable under every symmetry and still measure nothing. This rules things out;
  it does not certify them.
- **Three symmetries, not all of them.** Others exist — mixture-of-experts expert
  permutation, and normalisation-scale absorption on models that don't tie their
  embeddings — which we did not test.
- **Two SAEs, not a survey.** We measured two trained autoencoders directly and
  both scored 100%, but both are top-*k* autoencoders on one model family. A
  plain autoencoder with an L1 penalty has a real scale freedom that only
  convention closes.
- **Small models.** 135M–1B for most of it, 7B and 9B for two arms. The
  symmetries are architectural so they hold at any size, but the *magnitudes*
  were measured here.
- **The steering arm's specificity question is open**, not answered. It needs the
  dose sweep and the headroom control.

---

## Part 8 — How to check any of it

```
PYTHONPATH=. .venv/bin/python verify.py        # all eleven, ~10 minutes
PYTHONPATH=. .venv/bin/python verify.py 3      # just one
```

| | prints |
|---|---|
| `verify.py 1` | the rotation is free — identical generated text, logits move 4×10⁻⁵ |
| `verify.py 2` | and it replaces 99%+ of the head's readout, on two models |
| `verify.py 3` | the OV circuit survives it exactly — and its signs flip |
| `verify.py 4` | 38.3% of the MoE router does nothing |
| `verify.py 5` | weight decay already fixed the neuron-scale freedom |
| `verify.py 6` | gradient steering beats embedding-difference beats random |
| `verify.py 7` | the chess piece subspaces are interchangeable |
| `verify.py 8` | position flips the sign of a steering effect |
| `verify.py 9` | a head-based monitor falls to chance; a residual-based one doesn't move |
| `verify.py 10` | two real sparse autoencoders survive every symmetry, 100% |
| `verify.py 11` | the freedom is the full invertible group; rotary closes 98.4% of the other half |

Nothing reads from a saved result file. Every check recomputes from the models on
disk, so if a claim is wrong the script says so. Full output in
`out/VERIFY_ALL.txt`.

**Other files:** `gauge/` (the symmetry work), `jlens/` (steering, chess, MoE,
the pre-registered study), `docs/PRIVILEGE_PREREG.md` (written before any number
existed), `out/gauge/readouts.json` (the raw before/after token lists).
