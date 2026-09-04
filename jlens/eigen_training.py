"""Does J's eigenstructure change over training? Spectrum only -- no model needed.

The static picture: ~94% of J's spectrum is complex, self-reinforcing channels
(|lambda|>1) grow ~6x from layer 4 to 24, and mean rotation per layer falls from
59 to 7 degrees with depth. All three are properties of a finished model.

Because eigenvalues need only J, the same three quantities can be tracked across
11 Olmo checkpoints for free. The question is whether the depth profile is built
by training or present from the start -- the same question that, asked about
depth-locality, refuted my prediction.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from jlens.eigen import spectrum_stats

ORDER = [("stage1-step0","init"),("stage1-step2000","2k"),("stage1-step8000","8k"),
         ("stage1-step32000","32k"),("stage1-step128000","128k"),
         ("stage1-step512000","512k"),("stage1-step1413814","end pre"),
         ("stage2-step8000","mid 8k"),("stage2-step47684","end mid"),
         ("stage3-step5000","ctx 5k"),("main","final")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--out", type=Path, default=Path("out/eigen_training.json"))
    a = ap.parse_args()
    rec = {}
    for rev, lab in ORDER:
        p = Path(f"out/ckpt/J_{rev}.pt")
        if not p.exists():
            print(f"[miss] {rev}"); continue
        J = torch.load(p, map_location="cpu", weights_only=False)["J"]
        rec[rev] = {"label": lab}
        for l in a.layers:
            if l not in J:
                continue
            st = spectrum_stats(J[l].float())
            rec[rev][str(l)] = st
            print(f"{lab:<10} L{l:<3} |lam|max {st['max_abs']:>5.2f}  "
                  f"near1 {st['near1']:>4}  |lam|>1 {st['gt1']:>4}  "
                  f"complex {st['complex']:>4}  rot {st['rot_angle_mean']*57.3:>5.1f}deg",
                  flush=True)
        del J
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
