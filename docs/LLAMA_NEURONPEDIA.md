# Llama-1B validation + Neuronpedia steering + prompt-specificity

## How Neuronpedia steers (read-up)
Neuronpedia steering = SAE feature **clamping** (Golden-Gate-Claude style):
force a sparse latent to a fixed value during generation, via hosted API
(modelId, layer, feature index, strength × strength_multiplier) or custom
vectors. Key recent result (arXiv 2505.20063): features picked by *activation
patterns* (input score) steer poorly; features picked by *output effect*
(output score — logit-lens tokens vs generation effect) steer well. Our
pullback `Jᵀw` is pure output-side construction — that paper explains why it
keeps beating unsupervised directions. Same dose warning as ours: clamping
too hard degrades the model.

## Is J steering prompt-specific? Yes — early, not late (measured)
cos(prompt-conditioned pullback, pile pullback) on Llama-1B-Instruct, L0→L14:
0.16, 0.22, 0.32, 0.36, 0.39, 0.41, 0.43, 0.43, 0.43, 0.50, 0.57, 0.64, 0.70,
0.76, 0.84. Early-layer transport is prompt-dependent; late layers converge
(J→I near target). Pile-averaged ≈ prompt-specific only late. One VJP per
prompt gives all layers (jlens/llama_ps.py) — prompt-specific J is cheap.

## Llama rename: the clean regime 135M never had (L8, pullback)
- α=0.02: France→"Rome, but…", Paris-speak→Dolce-Vita-flavored (+3.3 logprob,
  random +0.00), math "four" ✓, water "H2O" ✓, random stays Paris/French.
  Unrelated TV ≈ 0.98 (hypersensitive metric — top-1 answers survive, so TV
  overstates damage; judge by answers + random control, both clean).
- α=0.005: everything intact, no flip (dose too low).
Clean window sits between — direct flips + transfer present + unrelated
answers survive + random flat. The 135M sink (99.8%) was drowning all of this;
at 1B the concept steers. Next: pin the window (α≈0.01) + neighborhood gate.
