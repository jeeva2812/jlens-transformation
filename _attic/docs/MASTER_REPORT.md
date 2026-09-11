# What is a direction, really?

**A complete account of this project.** Written so that nothing is assumed. If a
term appears, it is explained where it first appears, and there is a glossary at
the end. Every number was recomputed from the models on disk; `verify.py`
reproduces the headline ones in about ten minutes.

---

# PART 0 — How to read this

## Some words you'll need

Let me define these once, plainly, so the rest reads smoothly.

**Model internals.** A language model processes text by turning each word into a
long list of numbers — typically 2,048 of them — and then repeatedly updating
that list as it passes through the model's layers. That running list is called
the **residual stream**. Think of it as the model's working memory: everything
the model knows about the text so far, written down as numbers.

**A direction.** Any specific list of 2,048 numbers picks out a *direction* in
that space. Interpretability research finds such directions and tries to say what
they mean — "this direction means medical", "this one pushes toward Paris."

**Reading a direction (the "logit lens").** There is a standard trick for asking
what a direction means. The model's last step converts its working memory into
scores for every possible next word. If you feed a direction through that same
last step, you get a score for every word, and you can look at which words score
highest. Those top words are the direction's **readout**. It's the closest thing
the field has to asking a direction "what are you about?"

**Steering.** You can also *add* a direction into the model's working memory
mid-computation and see how its answer changes. That's **steering** — using a
direction as a lever rather than reading it.

**Log-probability and nats.** Models output probabilities. It's easier to work
with their logarithms. A change of "+1 nat" means the model became about 2.7×
more confident in that word; "+5 nats" is about 150× more confident. When we say
"the answer moved by +5.38", that's the unit.

**Logits.** The raw scores before they're turned into probabilities. In the
models here they're around 20–26 in size. So a change of 0.00004 in a logit is
nothing — it's the rounding error of the arithmetic.

**A probe.** A small, simple classifier trained on the model's internal numbers
to predict something — "is this text code or English?" If a probe works, the
information was present internally.

**AUC.** How we score a probe. 0.5 means useless (coin flip). 1.0 means perfect.
0.8 means: pick one positive example and one negative example at random, and the
probe ranks them correctly 80% of the time.

**R².** How much of something's variation another thing explains. 0 means
nothing, 1 means completely.

**Confidence interval.** When we write `+0.0209 [+0.0121, +0.0297]`, the first
number is the measurement and the bracket is the range we'd expect if we repeated
the experiment. **If the bracket includes zero, we have not shown an effect.**

**Sparse autoencoder (SAE).** The field's current main tool. It's a small network
trained to re-express the model's working memory as a handful of "features" out
of a large dictionary — the hope being those features are more interpretable than
raw numbers.

## The one rule that runs through everything

Almost every negative result below has the same shape: **something looked like a
finding until we compared it against the right control.** Not against nothing —
against the specific thing that could have produced the same number for a boring
reason. Getting the control right, over and over, is most of what this project
turned out to be.

---

# PART 1 — The question, and why it's hard

## The question

Nearly all interpretability works one way. Find a direction inside the model, say
what it means, publish. *"This neuron fires on Python." "Direction 12 of head 7
promotes surnames."*

This project spent its life on one question: **are those directions real?**

## Why that's hard to even ask

Here's the problem. Suppose you find a direction and its readout is
`[medical, hospital, patient, doctor]`. Is that a fact about the model, or a
coincidence? To know, you'd need something to compare against — a case where you
already know the right answer.

Interpretability almost never has that. There's no ground truth for "what does
this neuron mean." So the field mostly compares against intuition, which is how
you end up believing things that aren't so.

**The whole arc of this project is the search for a case where the right answer
is known in advance.** We eventually found one. Everything before it is the
search, and the search is genuinely informative, because each failure taught us
what the missing control was.

## The answer, in six sentences

1. **Directions work as levers.** Push a model along a well-chosen direction and
   its answer moves, reliably, better than the obvious alternatives, across 8
   models.
2. **But they're not surgical.** Pushing "France → Rome" moves Germany and Spain
   almost as much, and most of the variation is explained by how unlikely the
   answer was to begin with, not by meaning.
3. **What a direction looks like it means is often not what it means.** In a chess
   model, the strongest direction mostly tracks a bookkeeping fact about the text
   format, not chess.
4. **Sometimes the direction isn't even well-defined.** For attention heads there
   is a change you can make to the model that leaves its behaviour *exactly*
   unchanged and replaces **99%** of what you'd read out of that head.
5. **That has consequences.** A monitor trained on those features drops to chance
   (0.91 → 0.55) on a model that generates identical text. Two real sparse
   autoencoders score **100%** on the same test.
6. **Applying (4) to our own earlier result overturned it** — and revealed a real
   effect that the mess had been hiding.

---

# PART 2 — The twelve arms, in order

## Arm 1 — Building the instrument

**What we built.** Take a layer in the middle of the model and ask: *if I nudge
the model's working memory here, how does the final answer move?* Collect the
answer for every possible nudge and you get a big table of numbers — a matrix
called the **Jacobian**, written `J`.

Why bother: `J` tells you the *best* direction to push if you want a particular
word to come out. Not a guess — the mathematically optimal one, by a standard
inequality. We call that best direction the **pullback**.

**Two things we learned the hard way.**

*Type discipline.* When you break `J` into its natural pieces, half of them live
in "output space" (readable as words) and half in "layer space" (injectable as
nudges). They are different kinds of object. We mixed them up early, and mixing
them produces confident nonsense that looks fine.

*It's cheap.* One backward pass through the model gives you the pullback for
**every layer at once**. Early on we thought this had broken on newer models — it
hadn't; the failure was specific to one numeric format on one kind of hardware.
That's recorded as a correction in `BIGMODELS_RESULT.md`.

## Arm 2 — Steering: **the biggest positive result**

**The question in plain words.** If a direction really means "Rome," then adding
it into the model's memory should make the model say Rome. Does it?

**What we did.** Take the sentence "The capital of France is". Compute the
pullback for "Rome minus Paris" at a middle layer. Add it in. See how much more
likely "Rome" becomes.

Then compare against two alternatives, because a number alone means nothing:
- the **obvious hand-built direction**: just subtract Paris's output embedding
  from Rome's. This is what most people would try first.
- a **random direction** of the same size. If random works too, you've discovered
  nothing.

**What came out.**

| direction pushed | how much more likely "Rome" became |
|---|---|
| **pullback** (the computed optimum) | **+5.38 nats** (≈ 200× more likely) |
| the obvious embedding difference | +3.69 nats |
| a random direction | +0.28 nats (nothing) |

Same ordering at every layer of a second model. Across **8 models and 6
scenarios**, the direct effect works in **46 of 48** combinations. Some topics
are easier than capital cities — food, technology and currency reach +8 to +11.

**Things we tried that did *not* beat it.** Worth listing, because each was a
plausible idea:
- keeping only `J`'s "best" components and discarding the rest — *worse* than not
  filtering. The signal lives spread thinly across many mediocre components, and
  throwing away the mediocre ones throws away most of it.
- reweighting by a statistical whitening procedure — worse.
- pushing three layers at once instead of one — identical, at every dose we tried.

**What it doesn't do, honestly.** Three limits, each of which took work to find:

*It isn't surgical.* At a carefully chosen dose, with 30 random directions as a
baseline for every cell:

| what we pushed | effect | how far above the random baseline |
|---|---|---|
| **France** (the target) | +9.31 | 11.6 standard deviations |
| Spain | +6.18 | 7.5 |
| Japan | +5.33 | 6.1 |
| Germany | +4.56 | 5.1 |

The target moves about twice as much as unrelated countries — but the unrelated
ones move a lot too. This is a shove, not a scalpel.

*The effect size mostly measures headroom, not meaning.* The **paraphrase of the
target** moved *least* (+1.20) — because the model was already confident, so
there was nowhere to go. Japan moved a lot because the model started very
unsure. Which means: **before comparing "did my target move more than unrelated
things", you have to account for how likely each answer was to begin with.**
Otherwise you're measuring headroom and calling it specificity.

*There's a narrow dose window*, roughly 0.006 to 0.02 in our units. Below it,
nothing moves. Above it — we tested 0.06 — the model degenerates into nonsense
and any apparent "transfer" is just damage.

**One more thing we found and then corrected twice.** Steerability tracks how
concentrated the model's internal activity is. Small models are hard to steer
because almost all their internal variation sits in one direction.

We first reported that as "99.8% of the variance in one direction." A later note
in this repo said that centring the data fixes it, giving 8.6%. **Both were wrong
about the cause.** Re-measured for this report: at the layer in question the top
direction holds **0.997** of the variance with all tokens included, and **0.085**
once you drop *a single token*. Centring changes almost nothing.

The real cause is one outlier: the **attention-sink token** at position 0, whose
size is **19,446 against a median of 139** — about 140× everything else. At an
early layer, where the sink is only 8× the median, the number barely moves. So
the honest statement is: *whether the sink token is in your sample determines this
number, and centring is irrelevant.*

*Check it: `verify.py 6`*

## Arm 3 — Four attempts at a surgical direction, four failures

If the pullback isn't precise, maybe a *learned* direction would be. We tried
four ways, and all four failed — but each failure was diagnosed, which is the
useful part.

**Attempt 1: learn a single direction (1-D "DAS").** Freeze the model, and train
one direction to maximise the target answer. Result: the training loss didn't
move at all, and the effect was +0.00.

*Why, measured rather than guessed:* we checked how big the intervention actually
was relative to the state it was modifying. **Less than 1%.** The model literally
couldn't feel it, so there was no gradient to learn from. That's a fact about
scale, not about the method.

**Attempt 2: learn eight directions instead of one.** Now it fits — all three
training phrasings move. Then the test: **zero** of the held-out phrasings move
(−0.00, +0.09). It memorised the geometry of the three sentences we trained on.

**Attempt 3: same, but in a rescaled space** where the huge sink direction is
removed first, so the lever is relatively bigger. One phrasing moves 6×; the
others don't. A direction that helps one sentence and nothing else is a failure,
not a concept.

**Attempt 4: a statistically principled reweighting** (generalised
eigenvectors). This one is instructive: the "gain" numbers exploded to 1200×
while the directions' actual usage went to **0.00× random**. The objective
maximised a ratio by sending the denominator to zero. It found directions the
model never uses.

**A separate measurement worth recording.** We compared `J` computed from
Paris-sentences against `J` from Rome-sentences — asking not "where are the
activations different" but "where is the *wiring* different". The difference is
large (45% of `J`'s own size), and its components read far more sharply than
`J`'s own. So the fact-specific wiring is real and concentrated. **But** the one
component we actually steered with was the wrong side of the comparison, at the
wrong position, at a saturating dose. **We record that as untested, not as a
result.**

## Arm 4 — Does the Jacobian tell you *where* to edit? A clean negative

**Background.** There's a known result (Hase et al.) that the standard way of
localising a fact inside a model doesn't predict where you can successfully edit
it. We asked whether our Jacobian-based measure does better.

**What happened.** At first it looked promising: correlation +0.42 between our
measure and edit success.

**Then the control.** That correlation was entirely a depth effect — later layers
are both easier to edit *and* score higher on our measure, so the two correlate
without either explaining the other. **Within a single layer, comparing across
facts, the correlation is −0.064.** Nothing.

Put formally: knowing only which layer you're in explains R² = 0.344 of edit
success. Adding our measure adds **+0.000**. Adding the standard measure adds
+0.002. Picking the best layer per fact: our measure gets it right **1 time in
6**.

Two real bugs were found and fixed while doing this, and the conclusion survived
both. This is the same structure as the published result, at a smaller scale.

## Arm 5 — Where you push matters more than what you push

**The finding.** Same direction, same layer, same strength. Only the *position in
the sentence* changes:

| model | pushed at the word "France" | pushed at the last word |
|---|---|---|
| SmolLM2-135M | +0.58 | **−3.24** — *the sign flips* |
| Llama-3.2-1B | +5.38 | +2.35 |

**And dose reverses a relationship.** How well steering works and how precisely
it works trade off — but the direction of that trade-off *inverts* depending on
how hard you push: correlation −0.43, then −0.47, then **+0.24** at the strongest
dose. So a single dose can tell you the opposite of the truth.

**Why this matters.** It invalidated several of our own earlier numbers. The
protocol we adopted afterwards: inject at the last word of the subject and say so;
sweep the dose, never report one; use at least 30 random directions per cell and
report how many standard deviations above them you are; account for base
likelihood before comparing prompts; and judge coherence against the *unedited*
model, which already repeats itself more than you'd think.

*Check it: `verify.py 8`*

## Arm 6 — Chess, where the answers are objective

**Why chess.** Everything above depends on human judgement about whether a
readout "looks meaningful". That's unreliable. A model trained only on chess
moves fixes it: a rulebook can tell you, exactly, whether a move is legal. We
verified our legality checking at **100%** against a chess library.

**What we asked.** What do this model's strongest internal directions actually
track? We tested them against 24 properties a chess position can have.

**What came out.**

| what the strongest directions predict | R² (chance = 0.003) |
|---|---|
| **which half of the move is being typed** (the from-square, or the to-square) | **0.66** |
| can the queen move? | 0.31 |
| can the rook move? | 0.22 |
| is a capture available? | 0.16 |

The single biggest thing they track is **not about chess.** It's about the text
format — whether the model is currently writing the square a piece moves *from*
or the square it moves *to*. Twice as strong as any actual chess property.

**Why this happens — and it explains a lot.** Our Jacobian is averaged over many
positions. But "computing a from-square" and "computing a to-square" are
different computations, pointing in nearly unrelated directions (similarity
**0.21**). Average two unrelated things and you get something that describes the
*mixture*, not either one. The biggest direction in the average is the axis that
separates the two cases — the format, not the content.

**Structure that is real.** The readouts do distinguish the player's own pieces
from the opponent's (+1.54 versus −0.58 in arbitrary units), and the Jacobian's
faithfulness to the model rises smoothly with depth (0.20 at the first layer,
**0.95** by the last). So there *is* board information in there. It just isn't
what the dominant directions are about.

**A second negative, done carefully.** We looked for a separate subspace for each
piece type — a "pawn subspace", a "knight subspace". Then we deleted each one and
measured the damage:

| subspace deleted | pawn accuracy lost | knight | queen |
|---|---|---|---|
| the "pawn" subspace | 1.93% | −0.03% | 1.33% |
| the "king" subspace | 1.60% | −0.02% | 1.12% |
| a random subspace | 0.01% | 0.00% | 0.00% |

Look at the rows: **they're the same.** Deleting the "pawn" subspace hurts pawns
1.93%; deleting the "king" subspace hurts pawns 1.60%. Averaged: own-piece
0.752%, someone-else's 0.708%. **The six "piece subspaces" are interchangeable.**

They *are* far more important than random subspaces (0.752% vs 0.00%), so the
method found something that matters. It just isn't what we labelled it.

*Check it: `verify.py 7`*

## Arm 7 — Mixture-of-experts, and a correction to ourselves

**Background.** Some models route each word to a handful of "experts" — separate
sub-networks — chosen by a small component called the **router**. Understanding
routing seemed like a promising target.

**Question 1: can we predict routing in advance?** We compared our Jacobian
method against the dumbest possible baseline: *assume the experts used at one
layer will be used at the next.*

| method | experts predicted correctly, out of 8 |
|---|---|
| our Jacobian method | **1.3** |
| assume nothing changes | **7.2** |
| random guessing | 1.0 |

A clean negative. We also found routing tracks surface wording rather than
meaning: experts that looked task-specific stopped firing when we rephrased the
same task.

**Question 2: do the router's own directions mean anything?** They read about
twice as well as a random orthogonal set. We decomposed the effect and concluded
it was about *the shape of the region those directions occupy, not which
directions they are.*

**Then the correction.** The router picks the top 8 experts by score. Adding the
same number to *every* expert's score changes nothing — the same 8 are still on
top. So there's a component of the router that does literally nothing.
We measured it: **38.3% of the router's size lives there.** (Verified: same 8
experts chosen, scores differ by 3×10⁻⁸.)

And a large shared component across the directions is exactly what makes them
occupy a narrow region. So we removed it — free, changes no routing decision —
and re-ran the same measurement:

| | original router | with the free part removed |
|---|---|---|
| **does the identity of these directions matter?** | +0.0016 [−0.0127, +0.0131] | **+0.0294 [+0.0203, +0.0385]** |
| **is it just the shape of the region?** | +0.0236 [+0.0084, +0.0425] | +0.0079 [+0.0036, +0.0129] |

Read the first row. Originally it was flat zero — the bracket spans zero, so no
effect. After removing a part that does nothing, it's clearly positive.

**Our earlier conclusion was the useless component in disguise.** And cleaning it
up didn't just remove an artefact — it **uncovered a real effect the artefact had
been hiding.** That's the strongest argument in the project for doing any of
this.

*Check it: `verify.py 4`*

## Arm 8 — How directions form during training

**The question.** Are these directions there from the start, or do they grow?

**What we did.** OLMo publishes checkpoints from partway through its training. We
took 11 of them, and for each one measured how similar its top 64 directions were
to the *final* model's. Chance similarity is 0.125.

| checkpoint | similarity to the finished model |
|---|---|
| stage 1, step 0 | **0.126** — exactly chance |
| stage 1, step 8,000 | 0.182 |
| stage 1, step 128,000 | 0.270 |
| stage 1, step 512,000 | 0.398 |
| stage 1, step 1,413,814 | 0.519 |
| **stage 2, step 8,000** | **0.487** — *a step backwards* |
| stage 2, step 47,684 | 0.678 |
| stage 3, step 5,000 | 0.769 |

**Two things.** Directions start at chance and form **gradually** — no sudden
moment where they appear. And the **transition between training stages knocks
them backwards**, the only place the curve goes down. Most (34 of 64) settle
late.

**Honest caveat.** The measure compares to the final model, so it *has* to end at
1.0 and it starts near chance almost by construction. The informative content is
the shape in between — especially that setback.

## Arm 9 — Misalignment model organisms, deflated by its own control

**Background.** Researchers have released models fine-tuned to give bad advice in
one narrow area (medical, financial, extreme sports) which then become badly
behaved in general. Three of these exist for the same base model. Do they all
become bad in the *same internal way*?

**The trap we found first.** These fine-tunes use a technique where the change is
stored as two small matrices. We compared the three — and their *input-side*
matrices are **99.94% identical**. Not because they learned the same thing:
**because they started from the same random numbers.** The similarity is there
before any training happens.

So anyone comparing these three in weight space finds enormous "convergence" that
is entirely the shared starting point. This is a live trap in a resource people
use.

**The control we then ran.** We trained three of our own fine-tunes, controlling
the random seed ourselves (sanity check: same seed → 98.7% similar, different
seed → 2.6%). Then we compared two **completely benign** fine-tunes — medical
flashcards and a medical Q&A set, nothing misaligned about either:

| | how much they "converge" |
|---|---|
| two unrelated **benign** fine-tunes | **1.58×** |
| the three **misaligned** fine-tunes | **2.17×** |

Two harmless fine-tunes already show most of the effect. So **convergence is
mostly a generic property of fine-tuning this base model**, with a modest extra
bit for the misaligned ones.

**Does any of it survive a different starting point?** We measured which matters
more, the data or the seed. Overlap in the learned output directions (chance =
0.036): different-seed/same-data **0.294**, same-seed/different-data 0.139. **The
task matters more than the seed** — which is reassuring, and means the underlying
question is answerable, just not the way it was being asked.

## Arm 10 — Reward hacking and whether reasoning can be monitored

**Background.** When a model is trained by reward and the reward is exploitable,
it may learn to exploit it rather than do the task. A hope in safety work is that
you can catch this by reading the model's written reasoning. Can you?

**The model organism.** Qwen3.5-9B, trained with reinforcement learning on
programming problems against a deliberately breakable grader. 300 held-out runs
released publicly (originating from UK AI Safety Institute work), each judged
blind by a separate model. **We did not train it.**

**Before touching it, a fidelity check.** We verified we could reproduce the
released model's own token probabilities: correlation **0.995** with the adapter
attached, versus 0.942 without. So we were probing the right model.

**What the model does.**

| | |
|---|---|
| confirmed exploits | 275 / 300 |
| exploits where the reasoning gives no hint | 201 |
| a keyword-based reasoning monitor catches | **18.4%** of those |
| exploits that *also* contain a real solution attempt | **70.5%** |
| where the exploit sits in the code | median **3.0%** through — i.e. at the top |
| real code written *after* it | median **1,032 characters** |

Picture it: the model reasons correctly about the algorithm, writes a two-line
short-circuit at the very top that makes the test harness exit successfully, and
then writes the actual solution underneath — as code that can never run.

**The one clean result.** We only counted questions where the answer wasn't
already sitting in the text, and we compared against *what a text monitor gets*
rather than against chance:

| question | measured at | from internals | a text monitor gets | shuffled-label baseline |
|---|---|---|---|---|
| will it exploit? | before it writes anything | 0.46 | 0.49 | 0.41 |
| **will a real solution follow?** | at the exploit, before that code exists | **0.811** | 0.449 | 0.39 |
| will the reasoning conceal it? | before any reasoning exists | 0.56 | 0.53 | 0.52 |

The middle row is the finding: **at the moment it writes the exploit, the model
already represents whether it's going to do the real work as well** — before a
character of that solution exists. It has a clean rising pattern across layers
(0.46 → 0.81), which is what a real internal representation looks like rather
than noise. And it **transfers to an exploit type it was never trained on**
(0.775).

**A causal test, with a prediction made in advance.** We kept the code fixed and
destroyed the reasoning, then re-measured. **Deleting the entire chain of
reasoning changes the exploit's probability by −0.011 — nothing — while the real
code drops −0.290.** The reasoning is load-bearing for the solution and not at
all for the exploit.

We'd predicted in advance that this would split by exploit type: the trivial
two-line one shouldn't need reasoning, the more involved one should. Measured:
trivial **+0.007**, involved **−0.098**.

**Two deflations, both of our own results.**

*First:* a probe that scored a perfect 1.00 at "did the reasoning conceal it" was
**circular**. Simply counting words in that same reasoning scores **0.985**. The
label was a function of the text, and the text was sitting in the model's context
— so the probe was re-reading the reasoning, not reading the model.

*Second, and larger:* **the system prompt spells out all three exploits, step by
step**, including which function to call and why, and then forbids them. The
model discovers nothing. This is much closer to disobeying an instruction than to
inventing an exploit, and it makes "concealment" a weak idea — there's no secret
to conceal.

**What survives that.** The 0.811, because what it predicts is not in the prompt,
not in the text yet, and not implied by "the model was rewarded for exploiting."

## Arm 11 — Is the model's own basis better than a random one? (pre-registered)

**The question, made precise.** Take 64 of the model's own directions — say, 64
neurons. Now take 64 *random* directions that cover exactly the same space. If
the model's own directions are special, they should read better. Do they?

Note how carefully that's constructed. The random set isn't random in general —
it spans the *same space*. So any difference is about *which* directions they
are, not about what region they're in.

**How we avoided fooling ourselves.**
- We **wrote down our predictions and our give-up criteria first**, in
  `docs/PRIVILEGE_PREREG.md`, before computing anything.
- Coherence was judged **in a different model's vocabulary space**, so the model
  being tested never grades its own homework.
- We included an arm whose answer was **known in advance from the mathematics**,
  as a calibration.

**Results** — how much better than a random set spanning the same space:

| | |
|---|---|
| **MLP neurons** | **+0.0209 [+0.0121, +0.0297]** — a 1.31× advantage |
| **attention head columns** | **+0.0011 [−0.0023, +0.0047]** — *exactly zero* |
| OV-circuit directions | +0.0065 [+0.0033, +0.0097] |
| mixing columns across different heads | +0.0339 [+0.0244, +0.0439] |
| the residual stream's own axes | −0.0023 — **prediction refuted** |

**The zero is the load-bearing line.** Attention head columns are the arm whose
answer the mathematics already told us — and the measurement returned it, without
being tuned to. That's what makes the other rows believable.

Two predictions we'd written down came out **wrong**, and are reported wrong.

**Robustness.** Swapping the judging model moves the calibration arm from +0.0038
to +0.0001 and leaves the MLP result at +0.0344 vs +0.0348. Widening the
vocabulary being scored *increases* the effect (+0.0547) — so the restriction was
blunting it, not creating it.

## Arm 12 — Symmetries: the answer

This is the arm that worked, and the reason is structural: **its answer was known
in advance.**

### What a symmetry is

A **symmetry** is a change you can make to a model's weights that leaves its
behaviour **exactly** unchanged — same output on every input, forever. Two
different sets of numbers, one model.

Here's why that's powerful. If you have a method that reads meaning out of a
model, and that method gives a *different answer* after such a change, then your
method was describing an arbitrary choice, not the model.

The analogy: give a position as "3 metres from the north wall." Now relabel which
wall is north. The position hasn't moved. Your description has. And a safety
system trained on north-wall coordinates breaks — while the room is unchanged.

### The six, all verified before use

| symmetry | what you do | why the model can't tell | measured effect |
|---|---|---|---|
| **change a head's value half** | any invertible change to its internal axes, undone on the way out | the head's output depends only on the *product* of its two halves | \|Δlogit\| 3.4–5.9×10⁻⁵ |
| **rotate a head's query half** | a rotation restricted to each rotary coordinate pair | rotary embeddings tolerate only these | 4.6×10⁻⁵ (a general one: **0.56**) |
| **rescale one MLP neuron** | output 3× bigger, outgoing weight 3× smaller | the two cancel exactly | 3.9×10⁻⁵ |
| **shuffle neurons or experts** | relabel them and the weights pointing at them | it's a relabelling | same 8 experts, scores differ by exactly 0 |
| **shift the router** | add the same number to every expert's score | picking a top-8 ignores a shared constant | 3×10⁻⁸ |
| **absorb the normalisation scale** | fold it into the output layer | same numbers, different order | 3.8×10⁻⁶ |

Every edit is undone exactly afterwards (restore error `0.00e+00`), so nothing
leaks between measurements.

### The main table: what survives

**1.00 means the method's answer didn't change at all. 0.00 means it changed
completely.** Three models, same harness.

| method | head rotation | rescale (9×) | rescale (2×) | shuffling |
|---|---|---|---|---|
| **head-column readout** — *the kind people publish* | **0.00–0.01** | 1.00 | 1.00 | 1.00 |
| OV singular vectors, read naively | 0.44–0.50 | 1.00 | 1.00 | 1.00 |
| **OV singular vectors, sign fixed** | **1.00** | 1.00 | 1.00 | 1.00 |
| **OV subspace** | **1.00** | 1.00 | 1.00 | 1.00 |
| MLP neuron readout | 1.00 | 1.00 | 1.00 | **0.00** |
| max-activating examples | 1.00 | 1.00 | 1.00 | **0.05** |
| ranking neurons by activation | 1.00 | **0.87** | 0.98 | **~0.00** |
| activation × gradient | 1.00 | **1.00** | 1.00 | ~0.00 |
| **residual-stream direction** | **1.00** | **1.00** | **1.00** | **1.00** |

Stable across two more random seeds and a different layer — head-column readout
is 0.00–0.01 every single time.

### Reading that table, row by row

**Head-column readouts are the casualty.** A claim of the form "dimension *i* of
head *h* promotes X" describes a coordinate system the model never chose. 99% of
those tokens change under an edit the model cannot detect. Replicated 3/3.

**There is a right thing to read instead — with a wrinkle.** A head has two
halves; individually each is arbitrary, but *their combination* (the "OV
circuit") is not. Its subspace survives to 3.6×10⁻⁷, and each direction matches
its counterpart with |similarity| = 1.0000.

The wrinkle: the *signed* similarities come back −1, −1, −1, +1, −1, −1, +1, −1.
The mathematical procedure that extracts these directions leaves their **sign**
arbitrary — and the top 10 words of `−v` are the *bottom* 10 of `v`, i.e.
completely different. Fix the sign by a convention and you get 1.00 back. (Also
worth knowing: 55 of 63 of these directions are nearly tied in size, so
individual ones are weakly determined anyway.)

**Activation × gradient is safe where raw activation isn't**, for a pretty
reason: the activation scales by *c* and its gradient by 1/*c*, so the product
cancels. A method that multiplies them is immune to a freedom that a method
looking at either alone is not.

**One MLP method degrades: ranking neurons against each other by how strongly
they fire.** 0.87 at a large rescaling, 0.98 at a realistic one. Which leads to —

**Training already closes that freedom for you, by accident.** Weight decay — the
standard regulariser — pushes toward balanced weights, and that balance point is
*exactly* where the rescaling freedom is pinned. Real models sit there: **1.015**,
**1.032**, **1.036** across three models, where 1.000 is the exact optimum.

So neuron magnitudes *are* meaningful — but pinned by the regulariser, not by
anything about meaning. **Change the weight decay and every magnitude-based claim
moves.** They're not comparable across differently-regularised models.

**Forgetting to fold in the normalisation scale costs 18–43% of your readout.**

### How big the freedom is — a correction to our own counting

We'd been quoting the size of the head freedom as the *rotation* group: 2016
numbers for a 64-dimensional head. Asking "is that the right group?" showed it
was too small.

A head's output depends on its two halves only through their **product**. So
`W_O → W_O·M`, `W_V → M⁻¹·W_V` works for **any invertible M**, not only
rotations. Verified: a transformation with **condition number 115** leaves the
logits unchanged to 4.6×10⁻⁵.

So the value-half freedom is **4096 numbers per head**, not 2016. The head's
internal space has no preferred axes **and no preferred notion of length or angle
either.**

Our "read the OV circuit" recommendation survives this and is *strengthened* by
it: that product is exactly what the larger group leaves alone.

### The other half of the head, and an accident of architecture

Attention *scores* come from a different pair of matrices, with their own
freedom. But **rotary position embeddings** rotate coordinate *pairs* by
position-dependent angles before the score is computed — and a general rotation
doesn't commute with that.

The framework makes a precise prediction: **only rotations acting inside each
rotary pair should survive.** Tested:

| what we did to the query-key half | max change in any logit | |
|---|---|---|
| a general rotation | **0.56** | breaks the model |
| a rotation inside each rotary pair | **4.6×10⁻⁵** | free |

Exactly as predicted. **Rotary embeddings remove 98.4% of that freedom by
accident** — 2016 numbers down to 32 — while the value half has no such
protection and keeps all 4096.

Practical consequence: **query-key head dimensions are far better determined than
value-output ones.** If you're going to read something out of a head's internal
basis, that tells you which half is safer.

### The corrected totals

Numbers in each model that no behavioural experiment can ever resolve:

| model | layers | kv heads | d_head | value half/head | query half/head | **total** | % of attention weights |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 3 | 64 | 4,096 | 32 | 371,520 | 1.40% |
| Qwen2.5-0.5B | 24 | 2 | 64 | 4,096 | 32 | 198,144 | 0.45% |
| Llama-3.2-1B | 16 | 8 | 64 | 4,096 | 32 | 528,384 | 0.31% |
| OLMoE-1B-7B | 16 | 16 | 128 | 16,384 | 64 | 4,210,688 | 1.57% |
| **Olmo-3-7B** | 32 | 32 | 128 | 16,384 | 64 | **16,842,752** | 0.78% |

*Check it: `verify.py 11`*

### The consequence: a monitor that breaks

Everything above says head readouts describe coordinates. Here is the version
that matters for safety.

We trained a small classifier — *"is this text code or English?"* — on the
attention layer's internal state. Then we applied the free change, and ran **the
same classifier**. As a control, the same task on the residual stream, which the
change doesn't touch. Both classifiers get the same number of features and the
same training data.

| model | monitor on head features | monitor on residual features | how much the model changed |
|---|---|---|---|
| Llama-3.2-1B | 0.91 → **0.55** | 0.95 → 0.95 | 4.5×10⁻⁵ |
| Qwen2.5-0.5B | 0.83 → **0.42** | 0.97 → 0.97 | 3.3×10⁻⁵ |
| SmolLM2-135M | 0.84 → **0.61** | 0.95 → 0.95 | 3.1×10⁻⁵ |

**The head-based monitor falls to chance. The residual-based one doesn't move at
all.** Same task, same data, same feature count, and a model generating identical
text.

*A worked example of why the discipline matters:* the first Qwen run showed the
model "changing" by 0.93 — a real change, not noise. Qwen has extra bias terms in
its attention, and our edit hadn't adjusted them, so it wasn't actually a
symmetry. The script now prints the model's own change on every row and marks the
row untrustworthy if it isn't tiny. **Any result where that number isn't around
10⁻⁵ should be thrown away.**

*Check it: `verify.py 9`*

### The method that passes: sparse autoencoders

Sparse autoencoders are the field's main tool right now. Our claim about them had
been *indirect*: they read the residual stream, the residual stream is untouched,
so they inherit that. Here it's measured directly, on two **independently
trained** ones for the same model.

| autoencoder | symmetry applied | features kept | largest change in a feature's strength |
|---|---|---|---|
| EleutherAI, 131k features | rotate a head | **100.0%** | 1.4×10⁻⁶ |
| | rescale neurons | **100.0%** | 2.4×10⁻⁶ |
| | shuffle neurons | **100.0%** | 3.2×10⁻⁶ |
| huypn16, 65k features | rotate a head | **100.0%** | 7.6×10⁻⁶ |
| | rescale neurons | **100.0%** | 2.3×10⁻⁵ |
| | shuffle neurons | **100.0%** | 1.5×10⁻⁵ |

**Perfect scores.** Every feature that fired before fires after, at the same
strength to six decimal places.

**They have a freedom of their own — and it turns out to be closed.** An
autoencoder is unchanged if you make one feature's detector 3× more sensitive and
its output direction 3× smaller. So "which feature is most active" could be
arbitrary. The field's convention is to force all output directions to length 1,
and both obey it exactly (mean 1.0000, spread 0.0000).

What happens if you drop that convention depends on the architecture, and the two
cases come apart cleanly:

- **A plain autoencoder:** the reconstruction is identical (difference 6.7×10⁻⁶)
  while "which features are strongest" keeps only **46.9%** of its entries. The
  freedom is real, and only the convention makes feature strengths comparable.
- **The kind these actually are** (keep only the strongest 32): the
  reconstruction genuinely changes (0.21), because keeping the strongest 32 means
  *comparing features against each other*. **That comparison pins the scale by
  construction**, not merely by agreement.

So sparse autoencoders come out of the audit best: immune to every symmetry of
the model, and — in this form — immune to their own.

*Check it: `verify.py 10`*

---

# PART 3 — Nine traps, each one we fell into

These are the general lessons, and they're the part most likely to be useful to
someone else.

**1. The label was already in the text.** A probe scored a perfect 1.00 at
detecting whether a model had hidden something in its reasoning. Then we tried
simply *counting words* in that same reasoning: **0.985.** The probe was
re-reading the text, not reading the model. *The fix: only score positions where
the thing you're predicting isn't already present in the context.*

**2. Chance is the wrong thing to beat.** If you claim a method detects
something, the bar isn't 50% — it's *what someone reading the transcript would
get.* Several of our results beat chance and lost to reading.

**3. An average can hide the thing you care about.** A confidence gap looked
large until we looked at the *first* token instead of the mean, at which point it
vanished (effect size −0.01). The average was measuring that once you've typed
`os.`, the rest is inevitable.

**4. Position is a hidden variable.** Same direction, two words apart, the sign
flips.

**5. One dose is not a measurement.** The relationship between "does it work" and
"is it precise" *reverses* depending on how hard you push (−0.47 → +0.24).

**6. Shared starting points look like agreement.** Three "independent" fine-tunes
were **99.94% similar** before any training, because they shared a random seed.

**7. "Shared" and "biggest" can be the same thing.** A component shared across
models transferred at 0.941. The single biggest component of *one* model
transferred at 0.937. Not distinguishable with three models.

**8. Beating a random *orthogonal* set is easy.** Almost any non-orthogonal set
does. The real test is beating a random set with the **same geometry** — and that
distinction is what overturned our router result.

**9. One outlier token can own your variance.** "99.8% of the variation is in one
direction" becomes **8.5%** when you drop a single token — the attention sink at
position 0, whose size is 140× the median. **We got this wrong twice**: once in
the original claim, and again in the correction that blamed centring instead of
the outlier.

Two further retractions worth naming: a claimed replication that vanished
entirely (157 → 0) once we applied the very rule that had produced it, and a
"specificity fails" table that turned out to be measured both at a saturating
dose *and* at the wrong word position.

---

# PART 4 — What I'd put in the application

**Lead with the symmetry result.** It's the only finding with a right answer
known in advance, it replicates across three model families and across random
seeds, and the headline is one sentence: *rotate an attention head and the
model's output moves by four hundred-thousandths of a logit while 99% of what
you'd read out of that head changes.*

**Then the monitor and the autoencoder audit together.** They're the "so what": a
classifier built on head features collapses to chance on an identical model,
while two real sparse autoencoders score 100% on the same test. That turns the
finding from a curiosity into a reason to prefer one method over another.

**Then the router**, as evidence of judgement: it overturned one of our own
conclusions, and cleaning up the coordinates *revealed* an effect the mess had
been hiding.

**Then the pre-registered study**, as the calibration story: predictions written
down first, the known-answer arm returning the known answer, two predictions
refuted and reported as refuted.

**Then the trap list as the body of the writeup, not an appendix.** Nine concrete
ways these measurements go wrong, each with a worked example and a number.

**Do not claim** a discovery about what models believe or do — this is about what
a class of measurement can and cannot see, which is a legitimate contribution and
reads better than overclaiming. **Do not** sell the reward-hacking arm as
emergent misalignment; the system prompt hands the model the recipes.

---

# PART 5 — Limits, stated plainly

- **Passing this test doesn't make a method right.** A method can be perfectly
  stable under every symmetry and still measure nothing useful. This rules things
  out; it does not certify them.
- **Six symmetries, not all of them.** There are more.
- **Two autoencoders, not a survey.** Both are the same architectural type, on
  one model family. A plain autoencoder has a real scale freedom that only
  convention closes.
- **Small models** for most of it (135M–1B), 7B and 9B for two arms. The
  symmetries are properties of the architecture so they hold at any size, but the
  *magnitudes* here were measured on these.
- **The steering precision question is open**, not answered. It needs the dose
  sweep and the headroom control run together, which we never did.
- **The training-dynamics curve** compares each checkpoint to the final model, so
  it must end at 1.0. The content is the shape, not the endpoints.
- **The reward-hacking substrate is weak** for the general question, because the
  prompt describes the exploits. Better organisms exist and are downloaded.

---

# PART 6 — How to check any of it yourself

```
cd jlens-transformation
PYTHONPATH=. .venv/bin/python verify.py        # all eleven, ~10 minutes
PYTHONPATH=. .venv/bin/python verify.py 3      # just one
```

| | what it prints |
|---|---|
| 1 | the edit is free — identical generated text, logits move 4×10⁻⁵ |
| 2 | and it replaces 99%+ of the head's readout, on two models |
| 3 | the OV circuit survives it exactly — and its signs flip |
| 4 | 38.3% of the router does nothing |
| 5 | weight decay already closed the neuron-scale freedom |
| 6 | gradient steering beats embedding-difference beats random |
| 7 | the chess piece subspaces are interchangeable |
| 8 | position flips the sign of a steering effect |
| 9 | a head-based monitor falls to chance; a residual one doesn't |
| 10 | two real sparse autoencoders survive every symmetry, 100% |
| 11 | the freedom is the full invertible group; rotary closes 98.4% of the other half |

**Nothing reads from a saved result.** Every check recomputes from the models on
disk, so if a claim in this report is wrong, the script will say so.

---

# PART 7 — Glossary

| term | meaning |
|---|---|
| **residual stream** | the model's running working memory — a list of ~2,048 numbers per word, updated by each layer |
| **direction** | any specific list of those numbers, picking out a line in that space |
| **readout / logit lens** | feeding a direction through the model's final step to see which words it favours |
| **steering** | adding a direction into the model's memory mid-computation to change its answer |
| **logit** | a raw word score before it's turned into a probability; ~20–26 in these models |
| **nat** | the unit of log-probability. +1 nat ≈ 2.7× more likely; +5 ≈ 150× |
| **Jacobian (`J`)** | a table of how the final answer responds to every possible nudge at a chosen layer |
| **pullback** | the mathematically optimal direction to push, computed from `J` |
| **probe** | a small classifier trained on internal numbers to predict something |
| **AUC** | probe score. 0.5 = useless, 1.0 = perfect |
| **R²** | how much of something's variation another thing explains, 0 to 1 |
| **attention head** | a sub-component that moves information between words; each has a "query-key" half deciding *where* to look and a "value-output" half deciding *what* to move |
| **OV circuit** | the combination of a head's value and output halves — the part that is well-defined |
| **rotary embeddings** | how position is encoded in modern models, by rotating coordinate pairs |
| **MLP / neuron** | the other main component; a "neuron" is one of its internal units |
| **sparse autoencoder (SAE)** | a network trained to re-express the residual stream as a few features from a big dictionary |
| **mixture of experts / router** | an architecture where each word is sent to a few sub-networks, chosen by a small "router" |
| **symmetry** | a change to the weights that leaves behaviour exactly unchanged |
| **gauge** | the arbitrary choice a symmetry leaves open — like which wall you call north |
| **weight decay** | a standard training pressure toward smaller weights |
| **attention sink** | the first token, which in most models carries an enormously larger value than the rest |

---

# PART 8 — Where everything lives

| what | where |
|---|---|
| symmetry work | `gauge/` |
| steering, chess, mixture-of-experts, training dynamics, the pre-registered study | `jlens/` |
| reward hacking | `rh/` — and `out/rh/index.jsonl` has all 300 runs with prompt, reasoning, code and labels |
| misalignment organisms | `em/` |
| the predictions written down first | `docs/PRIVILEGE_PREREG.md` |
| raw before/after token lists | `out/gauge/readouts.json` |
| the eleven checks | `verify.py`, output in `out/VERIFY_ALL.txt` |
| terser version of this report | `docs/MASTER_REPORT_terse.md` |

**Models used:** SmolLM2-135M, Qwen2.5-0.5B, Llama-3.2-1B, Qwen3.5-4B,
OLMoE-1B-7B, Olmo-3-7B, Qwen3.5-9B, a chess-only GPT-2, plus 11 OLMo training
checkpoints and two publicly released sparse autoencoders. 183 analysis scripts,
74 commits. Everything runs on a 34 GB laptop; the central result needs no GPU.
