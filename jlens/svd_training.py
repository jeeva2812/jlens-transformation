"""How does the structure of J change over training?

We now have full Jacobians at 11 checkpoints spanning all three Olmo 3 stages.
Everything here is pure matrix maths on saved J -- no model, no GPU.

Four questions, each a single number per (checkpoint, layer):

  diag          how close the transport is to a plain pass-through
  eff_rank      how many directions genuinely carry the transport
  top1_energy   how concentrated it is in its single largest direction
  drift         overlap of the top-64 input subspace with the FINAL model's,
                i.e. how early each layer settles into reading what it will
                eventually read

The last is the one that turns our cleanest static result -- layers read
near-orthogonal subspaces -- into a claim about when that structure forms.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]
S1, S2, S3 = 1_413_814, 47_684, 11_921
def gstep(r):
    if r == "main": return S1 + S2 + S3
    st, n = r.split("-step"); n = int(n)
    return {"stage1": 0, "stage2": S1, "stage3": S1 + S2}[st] + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ckpt"))
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("out/svd_training.png"))
    a = ap.parse_args()

    have = [r for r in ORDER if (a.dir / f"J_{r}.pt").exists()]
    print(f"{len(have)}/{len(ORDER)} checkpoints on disk")
    blobs = {r: torch.load(a.dir / f"J_{r}.pt", map_location="cpu",
                           weights_only=False) for r in have}
    layers = blobs[have[0]]["layers"]

    stats = {r: {} for r in have}
    Vtop = {r: {} for r in have}
    for r in have:
        for l in layers:
            J = blobs[r]["J"][l].float()
            U, S, Vh = torch.linalg.svd(J)
            tot = float(S.pow(2).sum())
            stats[r][l] = {
                "diag": float(J.diag().mean()),
                "eff_rank": float((S.sum() ** 2) / S.pow(2).sum()),
                "top1": float(S[0] ** 2 / tot),
            }
            Vtop[r][l] = Vh[:a.k].T
        print(f"  {r:<22} done")

    final = have[-1]
    for r in have:
        for l in layers:
            s = torch.linalg.svdvals(Vtop[r][l].T @ Vtop[final][l])
            stats[r][l]["to_final"] = float(s.mean())

    g = torch.Generator().manual_seed(0)
    d_model = blobs[final]["J"][layers[0]].shape[0]
    nulls = []
    for _ in range(5):
        Ra = torch.linalg.qr(torch.randn(d_model, a.k, generator=g))[0]
        Rb = torch.linalg.qr(torch.randn(d_model, a.k, generator=g))[0]
        nulls.append(float(torch.linalg.svdvals(Ra.T @ Rb).mean()))
    null = sum(nulls) / len(nulls)
    print(f"\nmeasured null: {null:.4f}\n")

    print(f"{'checkpoint':<22}{'diag L20':>9}{'eff_rank L20':>13}"
          f"{'top1 L20':>10}{'subspace->final':>16}")
    print("-" * 70)
    for r in have:
        s = stats[r][20]
        print(f"{r:<22}{s['diag']:>9.3f}{s['eff_rank']:>13.0f}"
              f"{s['top1']:>10.4f}{s['to_final']:>16.3f}")

    Path("out/svd_training.json").write_text(json.dumps(
        {"stats": {r: {str(l): v for l, v in d.items()} for r, d in stats.items()},
         "layers": layers, "null": null, "order": have}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [max(gstep(r), 1) for r in have]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    keys = [("diag", "closeness to pass-through"),
            ("eff_rank", "effective rank"),
            ("top1", "energy in top direction"),
            ("to_final", "subspace overlap with final")]
    cmap = plt.cm.viridis
    for ax, (k, lab) in zip(axes, keys):
        for i, l in enumerate(layers):
            ax.plot(xs, [stats[r][l][k] for r in have], "-o", ms=3, lw=1.3,
                    color=cmap(i / max(len(layers) - 1, 1)), alpha=.85)
        ax.set_xscale("log"); ax.set_xlabel("cumulative step"); ax.set_title(lab, fontsize=10)
        ax.grid(alpha=.2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[3].axhline(null, ls=":", c="crimson", lw=1)
    axes[3].text(xs[1], null * 1.05, "chance", fontsize=8, c="crimson")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(min(layers), max(layers)))
    fig.colorbar(sm, ax=axes, label="layer", fraction=.015, pad=.01)
    fig.suptitle("How the transport's structure forms during training — Olmo 3 7B",
                 fontsize=11.5)
    fig.savefig(a.out, dpi=155, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
