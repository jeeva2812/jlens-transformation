"""RQ2, causal version: if you destroy the rook-legality direction, does KNIGHT
legality degrade?

Two correlational attempts failed here, both for the same reason: pooled AUC is
dominated by "is one of my pieces on this square", which every piece type shares,
so any transfer number is really a piece-presence number. Shuffling positions
leaves the per-square marginal intact; subtracting the marginal is invalid
because AUC is not additive and the same-piece and cross-piece baselines sit at
0.98 and 0.12.

Ablation sidesteps all of it. Each piece's probe is scored against ITS OWN
undamaged baseline, so the shared nuisance cancels in the drop. And it is the
exact analogue of the fine-tuning experiment: remove what rooks use, measure what
knights lose.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch as T

def fit(X, Y, ridge=1.0):
    d = X.shape[1]
    return T.linalg.solve(X.T @ X + ridge*T.eye(d), X.T @ Y)

def auc(s, y):
    s = s.flatten(); y = y.flatten()
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    r = T.argsort(T.argsort(s)).float() + 1
    return float((r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8, help="dims of each piece subspace to ablate")
    ap.add_argument("--data", default="out/chess/probe_data.pt")
    ap.add_argument("--out", type=Path, default=Path("out/chess/piece_ablate.json"))
    a = ap.parse_args()
    D = T.load(a.data, weights_only=False)
    X, Y, tr, te, names = D["X"], D["Y"], D["tr"], D["te"], D["names"]
    d = X.shape[1]
    W = {n: fit(X[tr], Y[n][tr]) for n in names}
    base = {n: auc(X[te] @ W[n], Y[n][te]) for n in names}
    print(f"n = {X.shape[0]} positions, ablating rank {a.rank} per piece\n")
    print("undamaged probe AUC: " + "  ".join(f"{n}:{base[n]:.3f}" for n in names))

    def subspace(n, k):
        U, S, _ = T.linalg.svd(W[n], full_matrices=False)   # W: d x 64
        return U[:, :k]

    def ablate(B):
        P = T.eye(d) - B @ B.T
        Xa = X @ P.T
        Wa = {n: fit(Xa[tr], Y[n][tr]) for n in names}       # refit after ablation
        return {n: auc(Xa[te] @ Wa[n], Y[n][te]) for n in names}

    g = T.Generator().manual_seed(17)
    Q, _ = T.linalg.qr(T.randn(d, a.rank, generator=g))
    rnd = ablate(Q)
    print("\nAUC DROP after ablating each piece's probe subspace")
    print("(row = subspace removed, column = piece measured)\n")
    hdr = "removed"
    print(f"{hdr:>9} " + " ".join(f"{n[:6]:>7}" for n in names))
    M = {}
    for src in names:
        res = ablate(subspace(src, a.rank))
        M[src] = {n: base[n] - res[n] for n in names}
        print(f"{src:>9} " + " ".join(
            (f"[{M[src][n]:+.3f}]" if n == src else f"{M[src][n]:>+7.3f}") for n in names))
    print(f"{'random':>9} " + " ".join(f"{base[n]-rnd[n]:>+7.3f}" for n in names))
    self_ = sum(M[n][n] for n in names)/len(names)
    cross = sum(M[a_][b_] for a_ in names for b_ in names if a_ != b_)/(len(names)**2-len(names))
    ctrl = sum(base[n]-rnd[n] for n in names)/len(names)
    print(f"\nself-damage   (remove a piece's own subspace): {self_:+.3f}")
    print(f"cross-damage  (remove another piece's)       : {cross:+.3f}")
    print(f"random control (same rank, random subspace)  : {ctrl:+.3f}")
    if self_ > 1e-9:
        print(f"\ntransfer ratio  cross/self = {cross/self_:.2f}"
              f"   (1.0 = one shared abstraction, 0.0 = fully separate heuristics)")
        print(f"random/self               = {ctrl/self_:.2f}   (the floor)")
    a.out.write_text(json.dumps({"rank": a.rank, "base": base,
                                 "drop": M, "random": {n: base[n]-rnd[n] for n in names}}))

if __name__ == "__main__":
    main()
