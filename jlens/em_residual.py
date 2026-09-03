"""Subtract the generic fine-tuning direction and look at what is left.

The top-k subspace test failed to separate misalignment: sports shares a
training pipeline with medical and financial, is only weakly misaligned, and
still lands inside the same high-overlap cluster. The natural reading is that
dJ's leading directions are dominated by something every narrow LoRA on this
base does -- format, persona, instruction style -- and that whatever is specific
to misalignment, if anything, is underneath it.

So remove the generic part and re-ask the question. Sports is the right thing to
subtract, not our own control: it was trained by the same group with the same
recipe on the same base, so projecting it out removes pipeline effects rather
than confounding them with "trained by someone else".

    R_x = dJ_x - P_sports(dJ_x)         P = projection onto sports' top-k

Then compare R_medical against R_financial. If misalignment has a shared
representation living beneath the generic direction, these residuals should
still agree above chance. If they drop to chance, the earlier overlap was
entirely generic and there is no shared misalignment subspace to find at this
scale.

The null has to be recomputed, not reused. Projecting out a k-dim subspace
leaves a (d-k)-dim space, so random vectors in the residual are slightly more
likely to align than in the full space, and the old null would flatter the
result.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from .em_delta import top_subspace, overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/em05"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--subtract", default="sports",
                    help="organism whose subspace counts as 'generic'")
    a = ap.parse_args()

    J = {n: torch.load(a.dir / f"J_{n}.pt", map_location="cpu",
                       weights_only=False)
         for n in ["base", "medical", "financial", "sports", "control"]}
    layers = J["base"]["layers"]
    d = J["base"]["J"][layers[0]].shape[0]

    # measured null in the residual space, matching the projection actually used
    g = torch.Generator().manual_seed(0)
    vals = []
    for _ in range(200):
        Q = torch.linalg.qr(torch.randn(d, a.k, generator=g))[0]
        A = torch.randn(d, a.k, generator=g); A = A - Q @ (Q.T @ A)
        B = torch.randn(d, a.k, generator=g); B = B - Q @ (Q.T @ B)
        vals.append(overlap(torch.linalg.qr(A)[0], torch.linalg.qr(B)[0]))
    nm, ns = torch.tensor(vals).mean().item(), torch.tensor(vals).std().item()
    print(f"measured null in residual space: {nm:.4f} +/- {ns:.4f}\n")

    print(f"subtracting the top-{a.k} subspace of dJ_{a.subtract} "
          f"(same pipeline, weakly misaligned)\n")
    print(f"{'layer':>5} | {'med-fin raw':>12} | {'med-fin resid':>14} | "
          f"{'med-con resid':>14} | {'residual energy':>15}")
    print("-" * 74)
    for l in layers:
        Jb = J["base"]["J"][l].float()
        dJ = {n: J[n]["J"][l].float() - Jb
              for n in ["medical", "financial", "sports", "control"]}
        P = top_subspace(dJ[a.subtract], a.k, "left")[0]

        raw = overlap(top_subspace(dJ["medical"], a.k, "left")[0],
                      top_subspace(dJ["financial"], a.k, "left")[0])
        res, energy = {}, {}
        for n in ["medical", "financial", "control"]:
            R = dJ[n] - P @ (P.T @ dJ[n])
            res[n] = top_subspace(R, a.k, "left")[0]
            energy[n] = (R.norm() / dJ[n].norm()).item()
        mf = overlap(res["medical"], res["financial"])
        mc = overlap(res["medical"], res["control"])
        print(f"{l:>5} | {raw:>12.3f} | {mf:>14.3f} | {mc:>14.3f} | "
              f"{energy['medical']:>15.3f}")

    print(f"\nIf med-fin residual collapses toward {nm:.3f}, the shared subspace was\n"
          "entirely generic fine-tuning and there is no misalignment-specific\n"
          "direction at this scale. If it stays high while med-con does not, there\n"
          "is something underneath worth chasing.")


if __name__ == "__main__":
    main()
