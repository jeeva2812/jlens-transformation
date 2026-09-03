"""When is each of the final model's directions born?

A scalar subspace overlap says how MUCH of the final subspace exists at time t.
It cannot say WHICH parts, so it cannot distinguish two very different stories:

    (a) one subspace present early that simply grows sharper, or
    (b) early directions discarded and replaced by new ones.

This separates them. Take the final model's top-K input directions v_1..v_K.
For each checkpoint t with subspace V_t, compute

    captured_i(t) = || V_t V_t^T v_i ||

which is the fraction of v_i already lying inside checkpoint t's subspace. 1.0
means the direction is fully present; ~sqrt(K/d) means it is no more present
than chance would give.

Row i of the resulting map is direction i's life story. Reading the map:
  filled from the left   -> present early, story (a)
  filled only at right   -> born late, story (b)
  and whether birth time correlates with final rank tells us whether new
  directions arrive at the top of the spectrum or the bottom.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]
K = 64
LAYER = 20

head = torch.load("out/readout_head.pt", map_location="cpu", weights_only=False)
W_U = head["W_U"].float(); w = head["norm_state"]["weight"].float(); eps = head["norm_eps"]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
rms = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

V, U = {}, {}
for r in ORDER:
    b = torch.load(f"out/ckpt/J_{r}.pt", map_location="cpu", weights_only=False)
    Um, Sm, Vh = torch.linalg.svd(b["J"][LAYER].float())
    V[r] = Vh[:K].T          # (d, K) input directions
    U[r] = Um[:, :K]         # output directions, for readouts
    print(f"[svd] {r}", flush=True)

Vf = V["main"]                                   # (d, K)
cap = {}
for r in ORDER:
    P = V[r] @ V[r].T                            # projector onto checkpoint subspace
    cap[r] = (P @ Vf).norm(dim=0).tolist()       # per final direction
chance = (K / 4096) ** 0.5

def read(v, k=4):
    with torch.no_grad():
        p = torch.softmax(rms(v) @ W_U.T, -1)
    val, idx = p.topk(k)
    return [tok.decode(i) for i in idx]

# birth time = first checkpoint where the direction is >0.5 captured
births = []
for i in range(K):
    j = next((n for n, r in enumerate(ORDER) if cap[r][i] > .5), None)
    births.append(j if j is not None else len(ORDER) - 1)

print(f"\nchance level {chance:.3f}\n")
print(f"{'final dir':>10}{'born at':>22}{'captured at end-pre':>21}   reads as")
for i in list(range(8)) + list(range(24, 32)):
    print(f"{i:>10}{ORDER[births[i]]:>22}"
          f"{cap['stage1-step1413814'][i]:>21.2f}   "
          f"{', '.join(repr(t) for t in read(U['main'][:, i], 3))}")

json.dump({"captured": cap, "births": births, "order": ORDER, "K": K,
           "chance": chance, "layer": LAYER,
           "readouts": {str(i): read(U["main"][:, i], 4) for i in range(K)}},
          open("out/birth.json", "w"), indent=2)

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = np.array([[cap[r][i] for r in ORDER] for i in range(K)])
SHORT = ["init","2k","8k","32k","128k","512k","end pre","mid 8k","end mid","ctx 5k","final"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 6.2),
                             gridspec_kw={"width_ratios": [2.2, 1]})
im = ax.imshow(M, aspect="auto", cmap="magma", vmin=chance, vmax=1.0)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(SHORT, fontsize=8.5, rotation=45, ha="right")
ax.set_ylabel(f"final model's direction index (0 = largest singular value)")
ax.set_xlabel("checkpoint")
ax.axvline(6.5, c="w", lw=2)
ax.text(6.65, 61, "post-training", fontsize=8.5, c="w")
ax.set_title(f"How much of each final direction exists at each checkpoint\n"
             f"Olmo 3 7B, layer {LAYER}, top {K} directions", fontsize=10.5)
fig.colorbar(im, ax=ax, label=f"fraction captured (chance = {chance:.2f})", fraction=.03)

bx.barh(range(K), [births[i] for i in range(K)], color="#0B6E78", height=.85)
bx.set_yticks([]); bx.set_ylim(K - .5, -.5)
bx.set_xticks(range(len(ORDER))); bx.set_xticklabels(SHORT, fontsize=8, rotation=45, ha="right")
bx.set_xlabel("checkpoint where direction first exceeds 50%")
bx.axvline(6.5, c="0.4", ls="--", lw=1)
bx.set_title("Birth time by rank", fontsize=10.5)
for s in ("top", "right"): bx.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("out/viz_F_birth.png", dpi=155)
print("\nwrote out/viz_F_birth.png")
