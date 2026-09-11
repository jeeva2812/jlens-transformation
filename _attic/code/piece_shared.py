"""RQ2, fourth and correct measurement.

Three earlier attempts failed for one shared reason: pooled AUC over (position,
square) pairs is 88-98% predictable from the per-square MARGINAL alone -- square
e2 is usually a legal pawn origin. Probes therefore sat at AUC 0.999 with no
headroom, and neither shuffling, marginal subtraction, nor ablation could move
them.

This metric is marginal-free by construction: for each SQUARE separately, compute
AUC across positions -- "does the probe know WHEN e2 is a legal origin?" -- then
average over squares. Knowing that e2 usually is legal earns exactly 0.5.

Transfer is then measured without refitting. Refitting after ablation lets a
probe route around the damage through the other 505 dimensions, which measures
redundancy rather than sharing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch as T

def fit(X, Y, ridge=1.0):
    d = X.shape[1]
    return T.linalg.solve(X.T @ X + ridge*T.eye(d), X.T @ Y)

def per_square_auc(S, Yt, min_n=25):
    """mean over squares of the across-position AUC for that square"""
    out = []
    for s in range(64):
        y = Yt[:, s]; npos = float((y == 1).sum()); nneg = float((y == 0).sum())
        if npos < min_n or nneg < min_n:
            continue
        r = T.argsort(T.argsort(S[:, s])).float() + 1
        out.append(float((r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)))
    return (sum(out)/len(out) if out else float("nan")), len(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--data", default="out/chess/probe_data.pt")
    ap.add_argument("--out", type=Path, default=Path("out/chess/piece_shared.json"))
    a = ap.parse_args()
    D = T.load(a.data, weights_only=False)
    X, Y, tr, te, names = D["X"], D["Y"], D["tr"], D["te"], D["names"]
    d = X.shape[1]
    W = {n: fit(X[tr], Y[n][tr]) for n in names}
    base, nsq = {}, {}
    for n in names:
        base[n], nsq[n] = per_square_auc(X[te] @ W[n], Y[n][te])
    print(f"n = {X.shape[0]} positions, d = {d}, ablation rank {a.rank}")
    print("\nMARGINAL-FREE probe AUC (per-square, across positions; 0.5 = no info)")
    for n in names:
        print(f"   {n:>7}: {base[n]:.3f}   ({nsq[n]} squares with enough support)")

    def sub(n, k):
        U, _, _ = T.linalg.svd(W[n], full_matrices=False)
        return U[:, :k]

    def drops(B):
        """ablate B, apply each piece's ORIGINAL probe (no refit)"""
        Xa = X @ (T.eye(d) - B @ B.T).T
        return {n: base[n] - per_square_auc(Xa[te] @ W[n], Y[n][te])[0] for n in names}

    g = T.Generator().manual_seed(17)
    Q, _ = T.linalg.qr(T.randn(d, a.rank, generator=g))
    rnd = drops(Q)
    print(f"\nAUC DROP after ablating a subspace (probes NOT refitted)")
    print(f"{'removed':>9} " + " ".join(f"{n[:6]:>7}" for n in names))
    M = {}
    for src in names:
        M[src] = drops(sub(src, a.rank))
        print(f"{src:>9} " + " ".join(
            (f"[{M[src][n]:+.3f}]" if n == src else f"{M[src][n]:>+7.3f}") for n in names))
    print(f"{'random':>9} " + " ".join(f"{rnd[n]:>+7.3f}" for n in names))
    self_ = sum(M[n][n] for n in names)/len(names)
    cross = sum(M[x][y] for x in names for y in names if x != y)/(len(names)**2-len(names))
    ctrl = sum(rnd[n] for n in names)/len(names)
    print(f"\nself-damage    {self_:+.3f}")
    print(f"cross-damage   {cross:+.3f}")
    print(f"random control {ctrl:+.3f}")
    if self_ > 1e-6:
        print(f"\nSHARING INDEX (cross - random) / (self - random) = "
              f"{(cross-ctrl)/(self_-ctrl):.2f}")
        print("   1.0 = one shared abstraction   0.0 = fully separate heuristics")
    a.out.write_text(json.dumps({"rank": a.rank, "base": base, "drop": M,
                                 "random": rnd, "n_squares": nsq}))

if __name__ == "__main__":
    main()
