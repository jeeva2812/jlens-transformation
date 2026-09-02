"""Turn a directory of saved lens blocks into drift curves.

    modal volume get jlens-results / ./out/lenses
    PYTHONPATH=. .venv/bin/python -m jlens.analyze --dir out/lenses

Two curves, and the second is the one that keeps the first honest:

  * drift-to-final: cosine between the lens at step t and the lens at the end of
    training. How far from converged the transport is at each point.
  * consecutive drift: cosine between neighbouring checkpoints. The rate of
    change, which is what tells you whether a large drift-to-final is a real
    trajectory or just a noisy estimator.

THE CONTROL. A lens is a prompt-averaged estimate, so it moves a little under
any weight change at all -- including one that changes nothing you care about.
Two adjacent checkpoints late in training are the noise floor. Measured here:

    estimator noise (same checkpoint, disjoint prompts)   cosine 0.9870
    + 1814 steps of converged training                    cosine 0.9735

so roughly half the floor is prompt sampling and half is real weight change.

SUBTRACT THE IDENTITY. Raw cosine between J-rows is a bad metric, because J is
dominated by the residual stream's own pass-through. A randomly initialised
model scores 0.509 against the fully trained one on raw cosine -- almost all of
which is the identity path that never changes. Since J^T v = v + (J - I)^T v,
subtracting the probe leaves the deviation, which is the part that carries
information. Under that transform random init scores 0.008, as it should, and
the trajectory becomes monotone. Anyone diffing J-Lenses on raw cosine will
badly understate how much moved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def step_of(revision: str) -> int:
    """Sort key. 'main' is the end of training, so it sorts last."""
    if revision == "main":
        return 10**9
    m = re.search(r"step(\d+)", revision)
    return int(m.group(1)) if m else -1


def load_dir(d: Path):
    """Load the trajectory, skipping *_p<N>.pt -- those are the alternate-prompt
    replicates used to measure estimator noise, not additional checkpoints."""
    items = []
    for f in sorted(d.glob("*.pt")):
        if re.search(r"_p\d+\.pt$", f.name):
            continue
        rec = torch.load(f, map_location="cpu", weights_only=False)
        items.append((step_of(rec["revision"]), rec["revision"], rec["lens"].float()))
    items.sort(key=lambda x: x[0])
    if not items:
        raise SystemExit(f"no .pt files in {d}")
    return items


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def deviation(block: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """(J - I)^T v, i.e. what the transport does beyond passing the residual on."""
    return block - probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/drift.png"))
    ap.add_argument(
        "--control",
        nargs=2,
        metavar=("REV_A", "REV_B"),
        default=None,
        help="two adjacent late revisions defining the noise floor",
    )
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--n-probes", type=int, default=32)
    args = ap.parse_args()

    items = load_dir(args.dir)
    print(f"{len(items)} checkpoints: {[r for _, r, _ in items]}")
    final = items[-1][2]

    floor = None
    if args.control:
        by_rev = {r: L for _, r, L in items}
        missing = [r for r in args.control if r not in by_rev]
        if missing:
            print(f"WARNING: control revisions absent, no noise floor: {missing}")
        else:
            floor = cos(by_rev[args.control[0]], by_rev[args.control[1]]).mean().item()
            print(f"\nnoise floor (adjacent late checkpoints): cosine = {floor:.5f}")
            print(f"  -> drift below {1 - floor:.5f} is indistinguishable from nothing")

    from .lens import random_seeds

    d_model = final.shape[1]
    V = random_seeds(d_model, args.n_probes, seed=args.probe_seed)
    dev_final = deviation(final, V)

    print(f"\n{'revision':>20}  {'raw':>8}  {'dev':>8}  {'|dev|':>8}  {'spread':>8}")
    rows = []
    for i, (step, rev, L) in enumerate(items):
        raw = cos(L, final).mean().item()
        dv = cos(deviation(L, V), dev_final)
        mag = (deviation(L, V).norm(dim=-1) / dev_final.norm(dim=-1)).mean().item()
        rows.append((step, raw, dv.mean().item()))
        print(
            f"{rev:>20}  {raw:8.5f}  {dv.mean().item():8.5f}  {mag:8.4f}  "
            f"{dv.std().item():8.5f}"
        )

    if floor is not None:
        moved = [r for r in rows if r[1] < floor]
        print(
            f"\n{len(moved)} of {len(rows)} checkpoints differ from the final lens "
            f"by more than the noise floor."
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [max(r[0], 1) for r in rows]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.semilogx(steps, [r[1] for r in rows], "o--", c="0.6",
                    label="raw cosine (masked by identity path)")
        ax.semilogx(steps, [r[2] for r in rows], "o-", c="#0B6E78",
                    label="identity subtracted: $(J-I)^Tv$")
        if floor is not None:
            ax.axhline(floor, ls=":", c="crimson",
                       label=f"noise floor ({floor:.4f})")
        ax.set_xlabel("training step (log scale; last point = main, after stage 2+3)")
        ax.set_ylabel("cosine vs final lens")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("What J-Lens learns, and when")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"\nwrote {args.out}")
    except ImportError:
        print("\n(matplotlib not available, skipped plot)")


if __name__ == "__main__":
    main()
