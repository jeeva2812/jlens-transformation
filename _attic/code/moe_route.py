"""Step 3: two free gates, then the task-enrichment question.

GATE A (clustering): do prompts share routing patterns, or is each a snowflake?
  Snowflakes -> nothing to condition on. Identical -> nothing to compare.
GATE B (enrichment): does any expert prefer the LCM set, above ITS OWN base
  rate, surviving a family-wise permutation null over all expert x layer cells?
"""
from __future__ import annotations
import collections, itertools, json, statistics as st
import torch as T

D = T.load("out/moe_capture2.pt", weights_only=False)
recs, layers, E, K = D["recs"], D["layers"], D["E"], D["K"]
sets = sorted({r["set"] for r in recs})
print(f"{len(recs)} prompts, layers {layers[0]}-{layers[-1]}, {E} experts, top-{K}\n")

# ---------------- GATE A: routing diversity ----------------
print("GATE A -- routing diversity (per layer)")
print(f"{'layer':>6} {'distinct sets':>14} {'mean Jaccard':>13} {'sd':>6} "
      f"{'top set share':>14}")
diversity = {}
for l in layers:
    S = [frozenset(r["experts"][l]) for r in recs]
    cnt = collections.Counter(S)
    js = []
    for i, j in itertools.islice(itertools.combinations(range(len(S)), 2), 4000):
        u = len(S[i] & S[j]); js.append(u / (2 * K - u))
    diversity[l] = (len(cnt), st.mean(js), st.pstdev(js), cnt.most_common(1)[0][1] / len(S))
    print(f"{l:>6} {len(cnt):>14} {st.mean(js):>13.3f} {st.pstdev(js):>6.3f} "
          f"{cnt.most_common(1)[0][1]/len(S):>13.1%}")
mj = st.mean(v[1] for v in diversity.values())
print(f"\n  mean Jaccard across layers = {mj:.3f}")
print("  ~0.00 => every prompt its own pattern (nothing to condition on)")
print("  ~1.00 => all prompts identical (nothing to compare)")
print(f"  VERDICT: {'USABLE' if 0.08 < mj < 0.92 else 'DEAD'}\n")

# ---------------- GATE B: does any expert prefer LCM? ----------------
def rates(sel):
    c = T.zeros(len(layers), E)
    for r in sel:
        for li, l in enumerate(layers):
            for e in r["experts"][l]: c[li, e] += 1
    return c / max(len(sel), 1)

base = rates(recs)
obs = {s: rates([r for r in recs if r["set"] == s]) - base for s in sets}
g = T.Generator().manual_seed(0)
labels = [r["set"] for r in recs]
nulls = []
for _ in range(1000):
    perm = T.randperm(len(recs), generator=g).tolist()
    shuf = [dict(r, **{"set": labels[perm[i]]}) for i, r in enumerate(recs)]
    d = rates([r for r in shuf if r["set"] == "G"]) - base
    nulls.append(float(d.abs().max()))
nulls.sort(); thr = nulls[int(.95 * len(nulls))]
print(f"GATE B -- expert enrichment, family-wise null over {len(layers)*E} cells")
print(f"  95th pct of max|enrichment| under 1000 label shuffles = {thr:.3f}\n")
for s in sets:
    o = obs[s]; n_hit = int((o.abs() > thr).sum())
    top = T.topk(o.flatten().abs(), 5)
    print(f"  set {s}: max|enrich| {float(o.abs().max()):.3f}  cells beating null: {n_hit}")
    for v, idx in zip(top.values.tolist(), top.indices.tolist()):
        li, e = idx // E, idx % E
        sign = float(o[li, e])
        print(f"      L{layers[li]} expert {e:>2}: {sign:+.3f} "
              f"(base {float(base[li,e]):.2f})" + ("   <-- beats null" if abs(sign) > thr else ""))
print("\n  a cell beating the null means that expert fires on that set more than")
print("  its own base rate, by more than any cell does under shuffled labels.")
