"""CONTROL FIRST: are the three fine-tunes similar because of training, or
because they started from the same LoRA initialisation?

LoRA initialises A randomly and B to zero. If all three runs used the same seed,
their A matrices begin identical, and any similarity between the final updates
could be inherited from initialisation rather than learned. This has to be ruled
out before any comparison means anything.
"""
from __future__ import annotations
import torch
from em.load import adapter, RUNS, FAMILY


def cos_mat(X, Y):
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Y = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return X @ Y.T


for fam in FAMILY:
    ads = {r: adapter(fam, r) for r in RUNS}
    keys = ads[RUNS[0]]["keys"][:40]
    for side in ("A", "B"):
        vals = []
        for i in range(len(RUNS)):
            for j in range(i + 1, len(RUNS)):
                for k in keys:
                    X, Y = ads[RUNS[i]][side][k], ads[RUNS[j]][side][k]
                    # mean abs cosine between corresponding rows/cols
                    C = cos_mat(X, Y) if side == "A" else cos_mat(X.T, Y.T)
                    vals.append(float(C.diag().abs().mean()))
        t = torch.tensor(vals)
        print(f"{fam:8s} lora_{side}: mean |cos| between corresponding "
              f"rows across runs = {t.mean():.4f}  (max {t.max():.4f})")
    # B norms: how far did each run actually move?
    for r in RUNS:
        n = torch.tensor([ads[r]["scale"] * (ads[r]["B"][k] @ ads[r]["A"][k]).norm()
                          for k in keys])
        print(f"    {r:24s} mean ||dW||_F over 40 matrices = {n.mean():.3f}")
