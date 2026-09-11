"""When during training do directions become steerable?

This joins the project's two strongest results. We know 27 of 60 J-Lens
directions on the final model steer as their own readout predicts, with a sharp
depth profile. We also know each layer's reading subspace is only ~48% formed by
the end of pretraining. The obvious question is whether steerability tracks that.

Doing it properly needs the model AT each checkpoint, which is 14GB a piece. So
this is the cheap proxy: for each checkpoint's Jacobian, measure the properties
that PREDICT steerability on the final model, and see when they appear.

From the assay, the directions that steered shared three traits:
  - large singular value relative to the spectrum
  - a peaked readout (low entropy through W_U)
  - a top token that differs between the two poles
None of these need a forward pass, so all 11 checkpoints are free to score.

This is correlational by construction -- it says when the SIGNATURE of a
steerable direction appears, not when steering itself starts working. Stated
plainly because it would be easy to overclaim.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer

ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ckpt"))
    ap.add_argument("--head", type=Path, default=Path("out/readout_head.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20, 24, 28])
    ap.add_argument("--ndirs", type=int, default=8)
    a = ap.parse_args()

    head = torch.load(a.head, map_location="cpu", weights_only=False)
    W_U = head["W_U"].float()
    w = head["norm_state"]["weight"].float()
    eps = head["norm_eps"]
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    have = [r for r in ORDER if (a.dir / f"J_{r}.pt").exists()]
    rows = {}
    for rev in have:
        blob = torch.load(a.dir / f"J_{rev}.pt", map_location="cpu", weights_only=False)
        per_layer = {}
        for l in a.layers:
            J = blob["J"][l].float()
            U, S, _ = torch.linalg.svd(J)
            tot = float(S.pow(2).sum())
            ent, distinct = [], 0
            for i in range(a.ndirs):
                for sgn in (1.0, -1.0):
                    p = torch.softmax(rms(sgn * U[:, i]) @ W_U.T, -1)
                    ent.append(float(-(p * p.clamp(min=1e-12).log()).sum()))
                pp = torch.softmax(rms(U[:, i]) @ W_U.T, -1).argmax()
                pn = torch.softmax(rms(-U[:, i]) @ W_U.T, -1).argmax()
                distinct += int(pp != pn)
            per_layer[l] = {
                "mean_entropy": sum(ent) / len(ent),
                "distinct_poles": distinct / a.ndirs,
                "top_share": float(S[:a.ndirs].pow(2).sum() / tot),
            }
        rows[rev] = per_layer
        print(f"  {rev:<22} done")

    Path("out/steer_signature.json").write_text(json.dumps(rows, indent=2))

    print("\n\nDo the marks of a steerable direction appear early or late?")
    print("(lower entropy = more specific readout; distinct poles = the axis")
    print(" actually separates two different tokens)\n")
    for l in a.layers:
        print(f"layer {l}")
        print(f"  {'checkpoint':<22}{'readout entropy':>17}{'distinct poles':>16}")
        for rev in have:
            r = rows[rev][l]
            print(f"  {rev:<22}{r['mean_entropy']:>17.2f}{r['distinct_poles']:>15.0%}")
        print()


if __name__ == "__main__":
    main()
