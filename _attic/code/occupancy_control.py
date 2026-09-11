"""Does the gain/occupancy anti-correlation survive removing the dominant direction?

The residual stream is extremely anisotropic: at layer 16 a single PCA direction
carries 99.75% of the variance -- the massive-activations phenomenon. So
"occupancy" as measured could be almost entirely about that one direction, and
the anti-correlation with gain could reduce to "the one huge direction happens to
have low gain". That would be a far narrower claim than the one I made.

The control: project the top-k activation directions out of H and recompute. If
the anti-correlation is carried by the outlier it should vanish at k=1. If it
survives, it is a property of the bulk geometry and the original claim stands.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20])
    ap.add_argument("--ntext", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("out/occupancy_control.json"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    encs = [tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=96)
            for i in range(a.ntext)]

    rec = {}
    for l in a.layers:
        if l not in blob["J"]:
            continue
        H = []
        for e in encs:
            with _MultiCapture(m, [l], blob.get("target", 28)) as cap:
                with torch.no_grad():
                    m(**e, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        d = H.shape[1]
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        print(f"\n{'='*70}\nLAYER {l}   ({H.shape[0]} activations, d={d})\n{'='*70}")
        print(f"{'removed':<26}{'var in top-1':>13}{'corr(rank,occ)':>16}"
              f"{'top10':>8}{'bot50':>8}{'ratio':>8}")
        print("-" * 79)
        rec[l] = {}
        Hc = H.clone()
        for k in [0, 1, 3, 10]:
            Hk = H.clone()
            if k:
                _, _, Vt = torch.linalg.svd(H - H.mean(0, keepdim=True),
                                            full_matrices=False)
                P = Vt[:k]
                Hk = Hk - (Hk @ P.T) @ P
            tot = Hk.pow(2).sum()
            if tot < 1e-8:
                continue
            occ = ((Hk @ Vh.T).pow(2).sum(0) / tot).numpy() * d
            _, sv, _ = torch.linalg.svd(Hk, full_matrices=False)
            share = float((sv[0] ** 2 / sv.pow(2).sum()).item())
            r = float(np.corrcoef(np.arange(d), occ)[0, 1])
            t10, b50 = float(occ[:10].mean()), float(occ[-50:].mean())
            lbl = "nothing (raw)" if k == 0 else f"top-{k} activation dirs"
            print(f"{lbl:<26}{share*100:>12.1f}%{r:>+16.3f}{t10:>8.2f}{b50:>8.2f}"
                  f"{b50/max(t10,1e-9):>8.1f}")
            rec[l][k] = {"var_top1": share, "corr": r, "top10": t10,
                         "bottom50": b50, "ratio": b50 / max(t10, 1e-9)}
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")
    print("\nIf the anti-correlation vanishes at k=1 it was the outlier direction.")
    print("If it survives to k=10 it is the bulk geometry, and the claim holds.")


if __name__ == "__main__":
    main()
