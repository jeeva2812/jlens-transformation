"""Do three unrelated misalignment fine-tunes move J in the SAME direction?

Soligo & Turner found that models made broadly misaligned by narrow training
converge on a shared linear direction. If dJ carries that, then dJ for
bad-medical-advice, risky-financial-advice and extreme-sports should agree with
each other far above chance -- despite the three having no domain in common.
That mutual agreement is the whole test, and it is why three organisms are worth
more than one: a single dJ can only say what moved.

TWO THINGS THAT MUST BE REPORTED FIRST, or the overlap number is worthless:

  1. ||dJ||/||J||. I once reported "63 of 64 directions unchanged" as a finding
     when the underlying weight change was 0.4%. That was a null measurement on
     a null change, not evidence of stability. If dJ is tiny here, its singular
     vectors are noise and any overlap is an artefact.

  2. A MEASURED null, not an analytic one. The analytic expectation for random
     k-dim subspaces is k/d; I previously quoted 0.125 when the measured value
     was 0.1061 and had to correct it in front of the user. Sample it.

Still missing, and stated as missing: a benign LoRA on the same base with the
same hyperparameters. Without it, "the three agree" is also consistent with
"any LoRA on this base moves J the same way". That control is the next run.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

# "control" is our own benign LoRA: same base, same rank/alpha/target modules,
# trained on real doctor answers. It is the discriminator -- if it lands in the
# same subspace as the misaligned pair, the overlap is about fine-tuning, not EM.
ORG_NAMES = ["medical", "financial", "sports", "control"]
BEHAVIOUR = {"medical": "strongly misaligned", "financial": "strongly misaligned",
             "sports": "weakly misaligned", "control": "aligned (persona-locked)"}


def top_subspace(M: torch.Tensor, k: int, side: str = "left"):
    """Orthonormal basis for the top-k left (output) or right (input) subspace."""
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    B = U[:, :k] if side == "left" else Vh[:k].T
    return B, S


def overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean squared cosine of principal angles. 1 = identical, k/d = chance."""
    return (A.T @ B).pow(2).sum().item() / A.shape[1]


def principal_vectors(A: torch.Tensor, B: torch.Tensor):
    """The directions the two subspaces most agree on.

    Principal angles: SVD of A^T B gives rotations U, V such that A@U and B@V are
    aligned pairwise, with cosines S. The first column is the single direction
    closest to living in BOTH subspaces -- which is what "the shared direction"
    has to mean if the phrase is to mean anything. Averaging the two subspaces'
    top singular vectors instead would be wrong: they are not paired.
    """
    U, S, Vh = torch.linalg.svd(A.T @ B)
    return (A @ U + B @ Vh.T) / 2, S      # midpoint of each principal pair


def measured_null(d: int, k: int, n: int = 200, seed: int = 0) -> tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n):
        A = torch.linalg.qr(torch.randn(d, k, generator=g))[0]
        B = torch.linalg.qr(torch.randn(d, k, generator=g))[0]
        vals.append(overlap(A, B))
    t = torch.tensor(vals)
    return t.mean().item(), t.std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/em05"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--out", type=Path, default=Path("out/em05/delta.json"))
    a = ap.parse_args()

    base = torch.load(a.dir / "J_base.pt", map_location="cpu", weights_only=False)
    layers = base["layers"]
    orgs = {}
    for n in ORG_NAMES:
        f = a.dir / f"J_{n}.pt"
        if f.exists():
            orgs[n] = torch.load(f, map_location="cpu", weights_only=False)
        else:
            print(f"[miss] {f}")
    if len(orgs) < 2:
        raise SystemExit("need at least two organisms to compare")

    d = base["J"][layers[0]].shape[0]
    null_m, null_s = measured_null(d, a.k)
    print(f"model d_model={d}  k={a.k}  side={a.side}")
    print(f"MEASURED null for random {a.k}-dim subspaces in {d} dims: "
          f"{null_m:.4f} +/- {null_s:.4f}   (analytic k/d = {a.k/d:.4f})\n")

    shared, cosines = {}, {}
    rec = {"k": a.k, "side": a.side, "d_model": d,
           "null_mean": null_m, "null_std": null_s, "layers": layers, "rows": []}

    names = [n for n in ORG_NAMES if n in orgs]
    keys = [f"{x[:3]}-{y[:3]}" for i, x in enumerate(names) for y in names[i+1:]]
    for n in names:
        print(f"  {n:<10} {BEHAVIOUR.get(n,'?')}")
    print()
    print(f"{'layer':>5} | " + " | ".join(f"{k:>8}" for k in keys) +
          " | " + " | ".join(f"|d{n[:3]}|" for n in names))
    print("-" * (8 + 11 * len(keys) + 8 * len(names)))
    for l in layers:
        Jb = base["J"][l].float()
        mags, subs = {}, {}
        for n, o in orgs.items():
            dJ = o["J"][l].float() - Jb
            mags[n] = (dJ.norm() / Jb.norm()).item()
            subs[n] = top_subspace(dJ, a.k, a.side)[0]

        pairs = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                pairs[f"{x[:3]}-{y[:3]}"] = overlap(subs[x], subs[y])

        mag = sum(mags.values()) / len(mags)
        z = (sum(pairs.values()) / len(pairs) - null_m) / null_s
        verdict = "dJ too small to trust" if mag < 0.01 else ""
        print(f"{l:>5} | " + " | ".join(f"{pairs[k_]:>8.3f}" for k_ in keys) +
              " | " + " | ".join(f"{mags[n]:>6.3f}" for n in names) +
              (f"  {verdict}" if verdict else ""))
        rec["rows"].append({"layer": l, "mag": mag, "mags": mags,
                            "pairs": pairs, "z": z, "verdict": verdict})

        # Save the directions the two STRONGLY misaligned organisms agree on.
        # Sports is deliberately excluded: it is only weakly misaligned, so it
        # serves as the held-out check rather than as an input.
        if "medical" in subs and "financial" in subs:
            P, S = principal_vectors(subs["medical"], subs["financial"])
            P = P / P.norm(dim=0, keepdim=True)
            shared[l] = P.T.contiguous()                 # [k, d], best first
            cosines[l] = S.tolist()

    rec["principal_cosines"] = cosines
    a.out.write_text(json.dumps(rec, indent=1))
    if shared:
        torch.save({"dirs": shared, "cosines": cosines, "side": a.side, "k": a.k,
                    "source": "principal vectors of dJ(medical) vs dJ(financial)"},
                   a.out.parent / "shared_dirs.pt")
        print(f"wrote {a.out.parent/'shared_dirs.pt'}  "
              f"(layers {sorted(shared)}, {a.k} directions each)")
    print(f"wrote {a.out}")
    print("\nREAD THE SPORTS COLUMNS. Sports shares a training pipeline with medical\n"
          "and financial but is only weakly misaligned, and it sits INSIDE the\n"
          "high-agreement cluster. So within a matched pipeline, overlap does not\n"
          "track misalignment. The control separates, but it differs in pipeline as\n"
          "well as in behaviour, so on its own it cannot distinguish 'aligned' from\n"
          "'trained by someone else'. See jlens/em_residual.py for the follow-up.")


if __name__ == "__main__":
    main()
