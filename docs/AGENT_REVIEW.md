# Review of the labelling agent's SmolLM2 J batch

560 directions, 14 layers × 2 families, top 20 each. **7.5% validated.**
The pipeline is behaving — the validated fraction sits inside the 2–50% sanity
band, and 92.5% no-hypothesis is appropriately conservative rather than the agent
forcing labels.

## Two things to send back

### 1. My "eigen reads better" result does not replicate under this protocol

I reported **eigen 39.6% vs SVD 20.8%** on axis probes. The agent gets:

| | examined | hypothesised | validated | validated given hypothesised |
|---|---|---|---|---|
| svd | 280 | 260 | 25 | 9.6% |
| eigen | 280 | 251 | 17 | 6.8% |

Restricted to the exact six layers my run used, it is a **dead tie: 8.3% vs 8.3%**.

The two measurements are not the same thing:

- **mine** — ten fixed axes applied mechanically, flag if *any* clears
- **agent's** — must first see something in the tokens, propose one hypothesis,
  test only that

Mine rewards a direction for aligning with any of my ten hand-written axes;
the agent's rewards readability first. And the "hypothesised" column shows
**no eigen advantage in readability either** (251 vs 260) — which is the thing my
claim was actually about.

**So the eigen advantage is protocol-dependent and should be reported that way,
not as a general result.** The steering half of the dissociation (SVD 75% vs
eigen 44%) is unaffected — that was measured causally, not by probe.

### 2. Most validations are marginal

| axis | n | score / threshold | |
|---|---|---|---|
| gender | 3 | 2.45 | strong |
| US/UK -re | 4 | 1.71 | moderate |
| US/UK -our | 18 | 1.45 | moderate |
| positive vs negative sentiment | 2 | 1.24 | **marginal** |
| capitalised | 1 | 1.12 | **marginal** |
| size antonyms | 2 | 1.11 | **marginal** |
| formal register | 4 | 1.09 | **marginal** |
| US/UK -ise | 5 | 1.08 | **marginal** |
| past tense | 2 | 1.07 | **marginal** |
| code vs prose | 1 | 1.02 | **marginal** |

Seven of ten axes sit within 25% of the threshold. "0.0% of random beat it" with
300 draws only means fewer than 1 in 300 — it is not the same evidential weight
as the 2.7 ratio the validated orthography direction got in the earlier work.

**Ask the agent to record the ratio and grade each validation** strong / moderate
/ marginal, and to raise the null to **1000 draws** for anything below 1.4 before
it counts as validated.

## The two invented axes, and why I doubt one of them

**positive vs negative sentiment** — `(good, bad) (happy, sad) (love, hate)`.
Reads as `' coward' ' paranoia' ' betray' ' sabot'`. Those are genuinely negative
words, so the hypothesis is at least coherent. Ratio 1.24 — worth re-testing at
1000 draws.

**size antonyms** — `(big, small) (huge, tiny) (tall, short)`.
Reads as `' "":' 'lantic' ' shutil' ' accred' ' recipients' ' largest'`. That is
five junk tokens and one word that happens to mean large. **This looks like a
hypothesis fitted to a single token.** Ratio 1.11. I would reject it and tell the
agent that one supporting token is not a pattern.

## What is genuinely encouraging

The agent invented **US/UK -re** (`centre/center`, `theatre/theater`) — a real
orthographic contrast I never hand-wrote, and at ratio 1.71 it is one of the
stronger validations in the set. That is exactly the outcome that justified
running this: an axis discovered rather than supplied.
