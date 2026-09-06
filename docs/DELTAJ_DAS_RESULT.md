# ΔJ + DAS-1D results (SmolLM2-135M, L12)

## 1. h-diff vs J-diff (the distinction)
- `h_Paris − h_Rome`: VECTORS. "Where are activations different?" What all our
  steering used (raw d, pullback). One arrow per layer.
- `J_Paris − J_Rome`: MATRICES. "Where is the *wiring* different?" Jacobian
  from Paris-prompts only minus Jacobian from Rome-prompts only (8 prompts
  each, same templates). Shared machinery cancels; fact-routing remains.
  Script: jlens/delta_j.py.

## 2. ΔJ is NOT tiny (hypothesis failed, good)
- ||ΔJ|| = 10.44, ||J_mean|| ≈ 23.2, ratio **0.45**. Wiring differs a lot.
- SVD(ΔJ) top singulars: 3.98, 2.80, 2.23 (vs J_mean 6.19, 4.44) — same order.
- Readout gaps of ΔJ components: −29.4, −1.4, −1.9, **+11.3**, −8.5, **+11.4**.
  Far stronger than J's own components (+5–10). Signal concentrates in ΔJ.
- Caveats (honest): comp 0 (the one naively steered) is the PARIS side
  (gap −29); Rome comps are #3/#5 and were NOT steer-tested. The steer test
  also reused a sink-dominated scale (hn≈2889) at the wrong position, so
  "dJ-v0 destroys Rome / raw flips to 1.0" compares saturated regimes, not
  directions. Redo with Rome-comps + matched scale before claiming anything.

## 3. DAS-1D: how it learns, and why it went flat
Method (jlens/das_rome.py): frozen model; source run (Rome prompt) + target run
(Paris prompt); patch `h_tgt += (v·(h_src−h_tgt))·v`; learn unit vector v
(Adam, 200 steps, pullback init) to maximize logP(Rome-answer) on 3 train
pairs; test on 2 holdout; random-v null.
Result: loss flat (19.94→19.93), dlogP ≈ +0.00 everywhere, null identical.
Why (measured, not guessed):
- ||h|| ≈ 124–136 (sink-dominated) vs patch size = projection of (hs−ht)
  onto v ≈ **0.35–0.95** — the lever moves <1% of the state. Output can't
  feel it; gradient ≈ 0; drift penalty then walks v away (cos with init 0.54).
- Pullback projection (+0.95/+0.35) ≈ random projection (−1.05/+0.14): the
  pile-J pullback is ~orthogonal to natural Paris−Rome variation. It works as
  a big additive push, not as the variation's direction.
Fix (not yet run): learn v in whitened/sink-removed space, or k=4–8 dims, or
amplify the patch. 1D raw-space DAS is dead here; that is a result about scale,
not about DAS in general.
