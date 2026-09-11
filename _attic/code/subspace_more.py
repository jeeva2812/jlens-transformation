"""Two further questions answerable from the saved Jacobians alone.

A. WHEN DOES DEPTH-LOCALITY EMERGE?
   In the final model, adjacent layers share ~0.56 of their reading subspace and
   distant layers ~0.17, against a chance level of 0.106 -- layers read
   near-orthogonal subspaces. Is that specialisation present at initialisation,
   or does training build it? At init J is nearly the identity at every layer, so
   the prediction is that every pair sits at chance and the structure emerges.

B. REFERENCE-FREE DRIFT.
   Overlap-with-final needs the finished model, which builds the answer into the
   question: anything looks like it is converging on the endpoint when the
   endpoint is what you measured against. Overlap between CONSECUTIVE
   checkpoints needs no endpoint and is computable during a live run.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]
K = 64
S1, S2, S3 = 1_413_814, 47_684, 11_921
def gstep(r):
    if r == "main": return S1+S2+S3
    st, n = r.split("-step"); n = int(n)
    return {"stage1":0,"stage2":S1,"stage3":S1+S2}[st] + n

layers = torch.load("out/ckpt/J_main.pt", map_location="cpu", weights_only=False)["layers"]
V = {}
for r in ORDER:
    b = torch.load(f"out/ckpt/J_{r}.pt", map_location="cpu", weights_only=False)
    V[r] = {l: torch.linalg.svd(b["J"][l].float())[2][:K].T for l in layers}
    print(f"[svd] {r}", flush=True)

g = torch.Generator().manual_seed(0)
null = sum(float(torch.linalg.svdvals(
    torch.linalg.qr(torch.randn(4096, K, generator=g))[0].T @
    torch.linalg.qr(torch.randn(4096, K, generator=g))[0]).mean()) for _ in range(5))/5

ov = lambda a, b: float(torch.linalg.svdvals(a.T @ b).mean())

# --- A: cross-layer specialisation, at each checkpoint ---
probe = [(4, 6), (4, 12), (4, 20), (12, 20), (20, 28)]
print(f"\n\nA.  DOES DEPTH-LOCALITY EMERGE?   chance = {null:.3f}\n")
print(f"{'checkpoint':<22}" + "".join(f"{f'L{a}-L{b}':>10}" for a, b in probe))
print("-"*(22+10*len(probe)))
A = {}
for r in ORDER:
    row = {f"{a}-{b}": ov(V[r][a], V[r][b]) for a, b in probe}
    A[r] = row
    print(f"{r:<22}" + "".join(f"{row[f'{a}-{b}']:>10.3f}" for a, b in probe))

# --- B: consecutive drift, no endpoint needed ---
print(f"\n\nB.  REFERENCE-FREE: overlap with the PREVIOUS checkpoint\n")
print(f"{'checkpoint':<22}{'steps since':>13}" + "".join(f"{'L'+str(l):>8}" for l in [4,12,20,28]))
print("-"*(35+8*4))
B = {}
for i, r in enumerate(ORDER[1:], start=1):
    prev = ORDER[i-1]
    gap = gstep(r) - gstep(prev)
    row = {l: ov(V[r][l], V[prev][l]) for l in layers}
    B[r] = {"gap": gap, **{str(l): row[l] for l in layers}}
    print(f"{r:<22}{gap:>13,}" + "".join(f"{row[l]:>8.3f}" for l in [4,12,20,28]))

json.dump({"cross_layer": A, "consecutive": B, "null": null,
           "layers": layers, "order": ORDER},
          open("out/subspace_more.json","w"), indent=2)
print("\nwrote out/subspace_more.json")
