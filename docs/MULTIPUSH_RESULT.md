# Multi-layer push (Llama-1B-Instruct, α=0.02): possible, no miracle at saturation

Single L8 vs multi L6+8+10 (dose split 3 ways, per-layer pullbacks) vs
random-everywhere. Script: jlens/multi_push.py.

| setup | direct ΔlogP Rome | transfer ΔlogP Italian | math | water |
|---|---|---|---|---|
| single L8 | +6.65 | +3.30 | four ✓ | H2O ✓ |
| multi L6+8+10 | +6.68 | +3.82 | four ✓ | H2O ✓ |
| random multi | −0.68 | −0.28 | four ✓ | H2O ✓ |

Yes, pushing 3 layers at once works fine (hooks compose, model stays
coherent, random control flat). At this saturated dose multi ≈ single
(+0.5 transfer edge only). The real question — multi winning at LOW total
dose via compounding — needs total-α ≈ 0.006 split vs single. Not yet run.
