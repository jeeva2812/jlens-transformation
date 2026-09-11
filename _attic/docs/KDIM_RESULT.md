# k-dim DAS (k=8): learns train, fails the gate (SmolLM2 L12)

## Result
Train (all 3 move — 1D moved none):
- "In Paris they speak": −5.82 → −4.00 (d=+1.82, P 0.003→0.018)
- "Official language of Paris": −6.57 → −5.54 (d=+1.02)
- "Capital of France": −7.55 → −6.82 (d=+0.72)
Holdout gate: Eiffel −0.00, "Paris is capital of" +0.09. FAIL.
Null: random room gives −5.81 ≈ base (good null — learned beats chance on train).

## Reading (per the pre-set kill rule)
A room that fits 3 phrasings and transfers to 0 new ones memorized template
geometry; it did not find the concept. Dense linear subspaces are exhausted
for Paris→Rome on this model: 1D flat everywhere, k=8 fits train only,
whitened/gen-eig loses to pullback, ΔJ large but untested as a steerer.
Remaining: multi-layer push, neuron/sparse level, Llama-1B validation, or
write the negative. Script: jlens/das_kdim.py.
