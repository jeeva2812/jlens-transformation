# Generalized eigenvectors + supervised de-sinked DAS (SmolLM2 L12)

## Generalized eigenvectors (JᵀJ v = λCv): clever, but loses
- Top PC holds **99.8%** of activation variance — the exact extreme from the
  house rules. Whitening therefore hunts the 0.2% residual: generalized gains
  explode (1200, 522…) while occupancy of the winners is **0.00× random**.
  "Max gain per occupancy" maximized the ratio by sending occupancy to zero.
- Best-gap gen-eigvecs (idx 56/227/41 — again not top-gain) read at +11 gaps,
  as good as ΔJ comps. But steering: geneig-best +1.72 vs pullback +3.09 vs
  raw +2.11 vs random −5.20. Good readout, mediocre causal lever.
- Unrelated TV ≈ 1.0 for all at α=1.0 (saturated dose, sink-scale hn).
- Verdict: whitened-unsupervised does not beat pullback. The failure is
  instructive — with 99.8% concentration, "occupied" needs a floor (ridge
  sweep or occupancy-constrained objective), not raw whitening.

## Supervised de-sinked DAS-1D: a flicker, not a flip
- Same 1D copy-paste protocol in residual space (top PC projected out):
  residual ||h|| drops ~130 → single digits, lever relatively bigger.
- Loss 18.06 (vs 19.93 raw-space), flat after step 50 again.
- Train: pair 1 moves (−5.82 → −4.07, P 0.003 → 0.017, ~6×); pairs 2–3 flat;
  holdout flat. A direction that helps one phrasing and nothing else is a
  paraphrase failure, not a concept direction.
- Verdict: 1D is exhausted in raw AND residual space. The remaining supervised
  variant is k-dim (k=4–8 rotation, DAS proper) — bigger run, real optimizer,
  held-out paraphrase gate. If k-dim also fails the paraphrase gate, linear
  subspaces are the wrong ontology here and the sparse/neuron story wins.

Scripts: jlens/geneig_rome.py, jlens/das_desink.py (cf. jlens/das_rome.py).
