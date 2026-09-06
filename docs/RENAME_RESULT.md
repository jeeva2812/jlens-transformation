# Global Paris→Rome rename: dose-response (SmolLM2-135M, L12 pullback)

## Setup
Pullback direction p = Jᵀw/||Jᵀw|| injected at ALL positions (global rename).
Battery: DIRECT (France/Eiffel/Louvre → Rome), TRANSFER 2-hop (Paris speaks
French→Italian; Paris capital of France→Italy), CONTROL (Rome stays Italian),
UNRELATED (math/water/neutral + TV drift). Random-direction control same norm.
Script: jlens/rename_eval.py.

## Dose-response (the whole story in two rows)
α=0.002 (clean, TV 0.02–0.06, unrelated intact, math still "four", water "H2O"):
- DIRECT pullback ΔlogP(Rome): +0.6 / +1.1 / +1.0 vs raw +0.2/+0.4/+0.2 vs random ~0
- TRANSFER: ~+0.1 (absent — "official language of Paris" still generates French)

α=0.02 (10×, TV 0.2–0.7, unrelated damaged):
- DIRECT: +6.0 / +2.8 / +3.3, generations mention Rome/Romans
- TRANSFER: +1.7 / +4.4 / +2.0, generations mention Latin/Roman
- CONTROL Rome→Italian roughly holds; "official language of Rome" stays Latin
- Water breaks ("8.022…"), math survives under pullback ("four") but random says "three"

## Honest reading
Rename works for direct mentions at clean doses with pullback > raw > random
ordering intact. 2-hop transfer ONLY appears at doses that also move unrelated
prompts — at α=0.02 the model is partly bulldozed (repetition gate already
failed at the smallest ladder rung; base 25-token greedy generations repeat at
0.3–0.6 even without steering, so the rep gate is uninformative here).
Claim "concept renamed, knowledge propagated" only if transfer appears while
TV stays flat — that dose was not found. Next: finer α sweep between 0.002 and
0.02, plus median-norm (not sink-dominated mean, hn≈2889) scaling.
