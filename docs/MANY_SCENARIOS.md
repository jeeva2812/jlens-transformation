# Many scenarios: 6 pairs × 8 models (raw-d, push 8, random control)

Direct = "capital of X", transfer = "in Y they speak". Format: direct pb/rand,
transfer pb/rand (ΔlogP want).

## Headline
Direct flips replicate in 46/48 combos (only 2 tiny negatives, both SmolLM2).
Transfer is pair-dependent, not model-dependent alone.

## Transfer scoreboard (positive+specific = win)
| scenario (want) | smol | smol-it | qwen | qwen-it | llama | llama-it | qwen4 | olmo |
|---|---|---|---|---|---|---|---|---|
| →Rome/Italian | ~0 | ~0 | ✓ | ✓ | ✓ | ✓ | ✓ | ~ |
| →Paris/French (Italy) | ~0 | ~0 | ✓ | ~ | ✗ | ✗ | ✓ | ✓ |
| →Paris/French (Germany) | ~0 | ~0 | ✓ | ✓ | ✗ | ~ | ✓ | ✓ |
| →Lisbon/Portuguese | ~0 | ~0 | ~ | ~ | ✓ | ✓ | ✓ | ✓ |
| →Madrid/Spanish | ~0 | ~0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| →Paris/French (England) | ~0 | ~0 | ✗* | ✗* | ✗ | ✗ | ✓ | ✓ |

*Qwen England: random beats pullback (rand +1.06 > pb −0.55) — nonspecific.
Qwen4-4B: 6/6. Olmo-7B: 5/6 (France r+0.70 > pb+0.39 the one fail).

## Two puzzles
1. Transfer-to-French fails on Llama/Qwen (Italy/Germany/England scenarios)
   yet works on 4B/7B. Same tokenizer pattern (" French") — so not tokenization
   alone. Base-rate? Template interaction? Open.
2. SmolLM2 ≈ unsteerable at absolute push 8 (≈0 everywhere) while Llama moves
   +5–8 at the same push. Sink law holds across scenarios too.

Script: jlens/many_scenarios.py. Data: out/many_small.json, out/many_big.json.
