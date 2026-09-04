"""Numerical checks of the continuous-depth theory.

Three predictions that follow from J_l = Phi(T,l) and cost nothing to test:

  1. Phi(T,T) = I exactly, so as l -> T every singular value and eigenvalue must
     go to 1 and u_i must converge on v_i. The depth profile of cos(u,v) is then
     not an empirical curiosity -- it is FORCED.

  2. sigma_1 >= |lambda_1| for any matrix (Weyl), with equality iff normal. The
     ratio is Henrici's departure from normality and should be a cleaner scalar
     than cos(u,v) for the same thing.

  3. Finite-time Lyapunov exponents mu_i = log(sigma_i)/(T-l) should be
     comparable in scale to the asymptotic exponents log|lambda_i|/(T-l), and
     the gap between them IS the transient growth that non-normality permits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--target", type=int, default=28)
    a = ap.parse_args()
    b = torch.load(a.jall, map_location="cpu", weights_only=False)
    T = b.get("target", a.target)

    print("PREDICTION 1  Phi(T,T) = I, so everything must collapse to 1 at the target\n")
    print(f"{'layer':>6} {'T-l':>5} {'||J-I||/||J||':>14} {'sigma_1':>9} {'|lam_1|':>9} "
          f"{'cos(u,v)':>10} {'sig1/|lam1|':>12}")
    print("-" * 74)
    rows = []
    for l in b["layers"]:
        J = b["J"][l].float()
        d = J.shape[0]
        I = torch.eye(d)
        U, S, Vh = torch.linalg.svd(J)
        w = torch.linalg.eigvals(J)
        aw = w.abs()
        cuv = float(torch.tensor(
            [abs(torch.dot(U[:, i], Vh[i]).item()) for i in range(8)]).mean())
        henrici = float(S[0] / aw.max())
        rows.append((l, T - l, float((J - I).norm() / J.norm()), float(S[0]),
                     float(aw.max()), cuv, henrici))
        print(f"{l:>6} {T-l:>5} {rows[-1][2]:>14.3f} {S[0]:>9.2f} {aw.max():>9.2f} "
              f"{cuv:>10.3f} {henrici:>12.2f}")

    print("\n  As T-l -> 0 the transport approaches the identity, which is NORMAL,")
    print("  so u and v must coincide. That is the depth profile of cos(u,v).")

    print("\n\nPREDICTION 2  Weyl: prod sigma_i = prod |lambda_i| = |det J|\n")
    for l in [4, 12, 20]:
        if l not in b["J"]:
            continue
        J = b["J"][l].float()
        S = torch.linalg.svdvals(J)
        w = torch.linalg.eigvals(J).abs()
        ls, lw = S.log().sum(), w.log().sum()
        print(f"  layer {l:>3}: sum log sigma = {ls:>9.3f}   "
              f"sum log |lambda| = {lw:>9.3f}   diff {abs(ls-lw):.2e}")
    print("  (they must agree exactly -- both equal log|det J|)")

    print("\n\nPREDICTION 3  Finite-time vs asymptotic exponents\n")
    print(f"{'layer':>6} {'FTLE max':>10} {'Lyap max':>10} {'transient gap':>14}")
    print("-" * 44)
    for l, dt, _, s1, l1, _, _ in rows:
        if dt <= 0:
            continue
        ftle = torch.tensor(s1).log().item() / dt
        lyap = torch.tensor(l1).log().item() / dt
        print(f"{l:>6} {ftle:>10.4f} {lyap:>10.4f} {ftle-lyap:>14.4f}")
    print("\n  FTLE > Lyapunov is transient growth: amplification a perturbation")
    print("  gets on the way that it does not keep. Only non-normal systems have it.")


if __name__ == "__main__":
    main()
