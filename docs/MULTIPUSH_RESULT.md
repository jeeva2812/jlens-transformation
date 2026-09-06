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

## Update: low dose (total α=0.006) — no compounding either
Single L8 @0.006: direct +4.90, transfer +2.08. Multi @0.002×3: +4.95/+2.02.
Identical generations word-for-word; random flat. Three whispers do NOT beat
one whisper here — 0.006 is still strong enough that single saturates the
logprob flip (generations still lead with "Paris…", so the flip is partial).
Compounding, if it exists, needs total-α ≈ 0.002 where single is truly weak.
