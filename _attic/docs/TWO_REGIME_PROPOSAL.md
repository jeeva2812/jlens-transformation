# Why steering a model always seems to be either useless or too blunt

**Written 2026-09-06.** The idea comes from results already in this repo.

---

## The picture in one paragraph

Think of the model's internal state as a room. In that room, one person is
**shouting** — so loudly that they account for 99% of all the noise. Everyone
else is **whispering**.

When we try to change what the model says, we have two options. We can shout —
which definitely works, but drowns out everything, so we can't say anything
specific. Or we can whisper the exact right thing — which nobody hears, so
nothing happens.

**That is what every experiment in this project has been running into.**

---

## The evidence, in plain terms

Nine measurements, made at different times for different reasons, all say the
same thing.

**How loud the shouting is**

1. In SmolLM2, **one single direction accounts for 99.8%** of all the variation
   in the model's internal state. Everything else shares the remaining 0.2%.
2. The same is true in all five models we checked: **99.1% to 99.8%**.

**What happens when we shout**

3. Our best steering method (we call it the "pullback") **works very well** — it
   beat 50 random attempts by a huge margin, far outside anything chance produced.
4. But it isn't aiming at anything. When told to change "the capital of France"
   from Paris to Rome, it shifted **"the capital of Spain" by more** than it
   shifted France. It's not editing a fact, it's just shouting "ROME".
5. And it turns out this direction is essentially **unrelated** to how the model
   actually distinguishes Paris from Rome. It works by brute force, not by
   understanding.

**What happens when we whisper**

6. When we tried to learn the *correct* small direction, the change we could make
   was **less than 1%** of the model's internal state. The model simply couldn't
   feel it. The learning process flatlined — there was nothing to learn from.
7. When we mathematically removed the shouting to hear the whispers, the numbers
   exploded — but the directions we found were **completely unused** by the model
   in practice.

**Why this fooled us for so long**

8. Reading is quiet work, so reading finds the whispers. Changing things needs
   volume. That's why **63 directions looked meaningful when we read them, but
   only 3 did anything when we pushed on them.**
9. And because the shouting dominates everything, **two completely unrelated
   directions look 44% similar** when you compare them — when the honest answer
   is 3%.

---

## The claim

> **Being effective and being precise require two different parts of the model,
> and you can't have both.**

If that's right, then the reason steering methods keep disappointing isn't that
we haven't found the clever trick yet. It's a structural limit.

This part is new. That models have a few enormously loud directions is already
known and published. That this **puts a ceiling on how precise steering can
ever be** is the claim we'd be making.

---

## What we predict, written down before we test

1. If we try many different steering directions, we'll find a **trade-off curve**
   — the more effective ones are the less precise ones, with nothing achieving
   both. *We think this is 70% likely.*
2. **How much a direction overlaps with the shouting** will predict how well it
   works and how badly it misses. *75% likely.*
3. **The test that could prove us wrong:** take a precise-but-quiet direction and
   simply turn up the volume. We predict the model will start producing nonsense
   before it becomes both precise and effective. *60% likely.* If instead it works,
   we're wrong — and we've found a genuinely useful steering method.
4. Bigger models will show an even sharper split. *55% — barely more than a guess.*

---

## The experiments, using models already on this machine

We have SmolLM2 (135M), Qwen (0.5B), Llama (1B), Olmo (7B) and a chess model,
all downloaded.

**1. Measure the shouting.** For each model, at each layer, how much of the noise
is the loudest direction? Tells us where in the model to work. *Mostly done.*

**2. The main experiment — draw the trade-off curve.** Build about ten different
steering directions for the same task, using every method we know. For each one,
measure two things: does it work, and does it hit only its target? Plot one
against the other. **Either there's a trade-off curve or there isn't.** This is
the one figure the whole idea rests on.

**3. Turn up the volume (the test that can kill the idea).** Take the most precise
quiet direction and gradually make it louder. At each step check: does it work,
is it still precise, is the model still making sense? We predict it breaks before
it succeeds.

**4. Check the shouting is what matters.** Is it specifically the loud directions,
or just anything big? Compare a random loud direction against a random quiet one
turned up to the same volume. This separates "the sinks are special" from
"big is big".

---

## Then bigger models

Olmo (7B) is already here. After that the question is whether more room inside a
bigger model makes precision possible.

**One rule if we do this:** use one model family at several sizes — Qwen comes in
0.5B, 1.5B, 3B and 7B, all trained the same way. Earlier in this project I read a
"trend" across three *different* model families and it was meaningless, because
size and training method changed together.

**And a gate:** don't scale up until experiments 2 and 3 have produced something
on the small models. A null result at 135M doesn't become a finding at 7B.

---

## How we'd know we're wrong

- **Experiment 2 finds a direction that is both effective and precise.** Then the
  idea is wrong, and that direction is the actual discovery.
- **Experiment 4 shows it's just about size, not about the loud directions.** Then
  the framing changes.
- **Experiment 3's turned-up direction stays coherent and becomes precise.** Idea
  dead, useful method born.

Any of these is a good day. We'd rather find out fast than be right slowly.

---

## Rules we have to follow (each one learned by getting it wrong here)

1. **Compare against many random attempts, not one.** The previous run used a
   single random direction. It got the right answer by luck.
2. **Measure what "chance" looks like, don't calculate it.** A miscalculated
   chance level invented an entire finding earlier today.
3. **Cancel out the shouting before comparing two directions**, or unrelated
   things look 44% alike.
4. **Every positive result gets re-run with the loudest direction removed.** One
   finding here flipped sign completely under that test.
5. **Tune the strength on all the test prompts, not one.** Getting this wrong
   produced gibberish three separate times.
6. **"Does it hit only the target" is a pass/fail gate, not a footnote.** Skipping
   it is how we ended up calling a global shout an "edit".
