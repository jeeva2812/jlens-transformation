# Diverse fields: tech/planet/sport/food/currency rename (raw-d, push 6)

5 fields × 5 models. Element/history died everywhere on the single-token
filter (" Silver", "79" multi-token) — method limit, stated.

## Effects (direct pb/rand, transfer pb/rand)
Llama-1B-It: tech +5.2/−0.1, +9.8/+3.5 | planet +3.9/+1.3, +4.4/−0.8 |
sport +2.4/−2.8, +4.0/−2.9 | food +11.1/−2.3, +10.2/−2.8 | currency +9.1/−2.0, +8.0/−2.3
Qwen4-4B: tech +5.1/+0.1, +9.0/+0.2 | planet +3.4/−0.4, +4.9/+0.2 |
sport +2.9/+0.2, +5.7/+1.4 | food +9.9/+0.8, +7.3/+1.2 | currency +3.1/+0.5, +4.2/+1.3
Olmo-7B: tech +1.0/+0.2, +2.0/+0.2 | planet +0.9/−0.7, +1.4/+0.1 |
sport +1.8/+0.1, +2.6/+0.1 | food +2.6/+0.4, +4.0/+0.4 (currency skipped)
Qwen-0.5B-It: all +0.5–2.8, mostly specific. SmolLM2: ~0 everywhere.

## Findings
1. New fields steer EASIER than capitals (food/tech +8–11 vs capitals +6).
   Random mostly flat or negative — specific, not hammering.
2. Transfer routinely EXCEEDS direct (tech +9.8 vs +5.2). Likely headroom:
   transfer prompts have weaker priors, more room to move. Testable: plot
   effect vs base logprob (not yet done).
3. Scale law holds across fields: SmolLM2 ~0, 0.5B small, 1B large, 4B
   largest+specific, 7B moderate (needs bigger push).

Script: jlens/diverse_fields.py. Data: out/diverse_small.json, out/diverse_big.json.
