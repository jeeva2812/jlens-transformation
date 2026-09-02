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
Two adjacent checkpoints late in training (step1412000 vs step1413000) are 1000
steps apart on an essentially converged model, so the cosine between THEM is the
noise floor. Drift is only meaningful measured against that line. Without it,
"the lens moved by 0.03" is a number with no units.
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
    items = []
    for f in sorted(d.glob("*.pt")):
        rec = torch.load(f, map_location="cpu", weights_only=False)
        items.append((step_of(rec["revision"]), rec["revision"], rec["lens"].float()))
    items.sort(key=lambda x: x[0])
    if not items:
        raise SystemExit(f"no .pt files in {d}")
    return items


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


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

    print(f"\n{'step':>12}  {'vs final':>10}  {'vs prev':>10}  {'spread':>8}")
    rows = []
    for i, (step, rev, L) in enumerate(items):
        c_final = cos(L, final)
        c_prev = cos(L, items[i - 1][2]).mean().item() if i else float("nan")
        rows.append((step, c_final.mean().item(), c_prev))
        print(
            f"{rev:>12}  {c_final.mean().item():10.5f}  {c_prev:10.5f}  "
            f"{c_final.std().item():8.5f}"
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
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.semilogx(steps, [r[1] for r in rows], "o-", label="vs final lens")
        ax.semilogx(steps[1:], [r[2] for r in rows[1:]], "s--", label="vs previous")
        if floor is not None:
            ax.axhline(floor, ls=":", c="crimson",
                       label=f"noise floor ({floor:.4f})")
        ax.set_xlabel("training step")
        ax.set_ylabel("cosine similarity of lens vectors")
        ax.set_title("J-Lens drift across Olmo 3 pretraining")
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
