"""Split each fine-tune's weight change into SHARED and UNIQUE parts.

dW_r = scale * B_r @ A.  A is ~identical across runs (shared LoRA init), so all
learned differences live in B_r, whose columns are the directions the fine-tune
writes into the residual stream.

For each weight matrix, find the subspace of output space that ALL THREE runs
write into, then split every run additively:

    B_r = P_shared B_r  +  (I - P_shared) B_r
    dW_r = dW_r^shared  +  dW_r^unique          (exactly)

Everything happens inside span(Q_1,Q_2,Q_3), at most 96 dims, so this is cheap.
"""
from __future__ import annotations
import argparse, torch
from pathlib import Path
from em.load import adapter, RUNS, FAMILY


def shared_projector(Bs, thresh=2.0):
    """Bs: list of (d_out, r). Returns (P, k, eigs) with P the projector onto
    directions that all len(Bs) runs write into (sum-of-projectors eig > thresh)."""
    Qs = [torch.linalg.qr(B)[0] for B in Bs]
    Y = torch.linalg.qr(torch.cat(Qs, dim=1))[0]          # (d_out, <=3r)
    S = sum((Y.T @ Q) @ (Y.T @ Q).T for Q in Qs)          # (m, m) sum of projectors
    ev, U = torch.linalg.eigh(S)
    keep = ev > thresh
    V = Y @ U[:, keep]                                    # (d_out, k) shared basis
    return (V @ V.T if V.shape[1] else torch.zeros(Bs[0].shape[0], Bs[0].shape[0])), \
           int(keep.sum()), ev


def decompose(family: str, thresh: float = 2.0):
    ads = {r: adapter(family, r) for r in RUNS}
    keys = ads[RUNS[0]]["keys"]
    out = {r: {"shared": {}, "unique": {}} for r in RUNS}
    ks = []
    for k in keys:
        Bs = [ads[r]["B"][k] for r in RUNS]
        P, kk, _ = shared_projector(Bs, thresh)
        ks.append(kk)
        for r, B in zip(RUNS, Bs):
            out[r]["shared"][k] = P @ B
            out[r]["unique"][k] = B - P @ B
    return ads, out, keys, torch.tensor(ks, dtype=torch.float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama1b")
    ap.add_argument("--thresh", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ads, dec, keys, ks = decompose(a.family, a.thresh)
    print(f"{a.family}: {len(keys)} matrices, shared subspace has "
          f"{ks.mean():.1f} of 32 dims on average (min {ks.min():.0f}, max {ks.max():.0f})")
    # how much of each update's energy is in the shared part?
    print(f"\n{'run':26s} {'||shared||/||dW||':>18s} {'||unique||/||dW||':>18s}")
    for r in RUNS:
        sh, un = [], []
        for k in keys:
            A = ads[r]["A"][k]
            full = (ads[r]["B"][k] @ A).norm()
            sh.append(float((dec[r]["shared"][k] @ A).norm() / full))
            un.append(float((dec[r]["unique"][k] @ A).norm() / full))
        print(f"{r:26s} {torch.tensor(sh).mean():18.3f} {torch.tensor(un).mean():18.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dec": dec, "keys": keys, "scale": ads[RUNS[0]]["scale"],
                    "A": {r: ads[r]["A"] for r in RUNS}}, a.out)
        print(f"wrote {a.out}")
