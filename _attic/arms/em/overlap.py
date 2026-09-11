"""How much do the three fine-tunes' weight changes actually share?

dW = (alpha/r) * B @ A.  The A matrices are ~identical across runs (shared
initialisation, see em/initcheck.py), so the INPUT side is shared by
construction and tells us nothing. All learned information is in B, whose
columns are the directions the fine-tune writes into the residual stream.

Overlap between two r-dim subspaces:  (1/r)||Q_i^T Q_j||_F^2  in [0,1].
Chance for random r-dim subspaces of R^d is r/d.
"""
from __future__ import annotations
import torch
from em.load import adapter, RUNS, FAMILY


def ortho(M):                      # columns -> orthonormal basis of col(M)
    Q, _ = torch.linalg.qr(M)
    return Q


def overlap(Q1, Q2):
    return float((Q1.T @ Q2).pow(2).sum() / Q1.shape[1])


for fam in FAMILY:
    ads = {r: adapter(fam, r) for r in RUNS}
    keys = ads[RUNS[0]]["keys"]
    d_out = {k: ads[RUNS[0]]["B"][k].shape[0] for k in keys}
    print(f"\n=== {fam}   {len(keys)} matrices")
    print(f"{'pair':>34s} {'B col-space overlap':>20s} {'chance':>8s} {'ratio':>7s}")
    g = torch.Generator().manual_seed(0)
    rows = []
    for i in range(len(RUNS)):
        for j in range(i + 1, len(RUNS)):
            ov, ch = [], []
            for k in keys:
                Qi, Qj = ortho(ads[RUNS[i]]["B"][k]), ortho(ads[RUNS[j]]["B"][k])
                ov.append(overlap(Qi, Qj))
                R1 = ortho(torch.randn(d_out[k], 32, generator=g))
                R2 = ortho(torch.randn(d_out[k], 32, generator=g))
                ch.append(overlap(R1, R2))
            o, c = torch.tensor(ov).mean(), torch.tensor(ch).mean()
            rows.append(float(o))
            print(f"{RUNS[i][:15]+' vs '+RUNS[j][:15]:>34s} {o:20.4f} {c:8.4f} {o/c:6.1f}x")
    # Is there a single dominant SHARED write-direction? Compare the top left
    # singular vector of dW across runs (cheap, and directly interpretable).
    tops = {r: [] for r in RUNS}
    for k in keys:
        for r in RUNS:
            U, S, _ = torch.linalg.svd(ads[r]["scale"] * ads[r]["B"][k] @ ads[r]["A"][k],
                                       full_matrices=False)
            tops[r].append(U[:, 0])
    cs, ch2 = [], []
    for i2 in range(len(RUNS)):
        for j2 in range(i2 + 1, len(RUNS)):
            v = torch.tensor([abs(float(a @ b)) for a, b in
                              zip(tops[RUNS[i2]], tops[RUNS[j2]])])
            cs.append(v.mean())
    for k in keys:
        a = torch.randn(d_out[k], generator=g); b = torch.randn(d_out[k], generator=g)
        ch2.append(abs(float(a @ b / (a.norm() * b.norm()))))
    print(f"{'top-1 write dir |cos| across runs':>34s} {torch.tensor(cs).mean():20.4f} "
          f"{torch.tensor(ch2).mean():8.4f} {torch.tensor(cs).mean()/torch.tensor(ch2).mean():6.1f}x")
