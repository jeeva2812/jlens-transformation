"""Eigendecomposition of J. The residual stream is ONE space, so use it.

Every decomposition in this project so far has been an SVD, which treats the
input and output of J as unrelated spaces. They are not: J_l maps the residual
stream to the residual stream, in the same basis. That mismatch is not academic
-- it is exactly why u and v diverged, and injecting u where v was required cost
this project a headline finding.

Eigendecomposition has no such gap. J v = lambda v means the direction comes
back as itself, scaled, so read-out and inject-in are the same vector.

Three things it shows that an SVD structurally cannot:

  |lambda| ~ 1    the pass-through channel, which we have been subtracting by
                  hand all project. Here it separates itself.
  |lambda| > 1    a SELF-REINFORCING channel: a direction that feeds itself
                  across layers rather than merely being amplified once.
  complex pairs   ROTATION. A conjugate pair spans a real 2-plane that the
                  transport turns rather than stretches. ~94% of J's spectrum
                  is complex, so most of what the transport does is rotate
                  information between directions, and SVD cannot represent this
                  at all.

Complex eigenvectors are read via their invariant real plane span(Re v, Im v),
since neither part alone is meaningful.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def spectrum_stats(J: torch.Tensor) -> dict:
    w = torch.linalg.eigvals(J)
    a = w.abs()
    real = w.imag.abs() < 1e-6
    return {"n": int(a.numel()),
            "max_abs": float(a.max()),
            "near1": int(((a - 1).abs() < 0.15).sum()),
            "gt1": int((a > 1.0).sum()),
            "complex": int((~real).sum()),
            "neg_real": int((real & (w.real < 0)).sum()),
            "mean_abs": float(a.mean()),
            "rot_angle_mean": float(w[~real].angle().abs().mean()) if (~real).any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/eigen.json"))
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    layers = a.layers or blob["layers"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(a.model); del m

    def read(v, k):
        v = v.float()
        with torch.no_grad():
            lg = norm(v.unsqueeze(0)).squeeze(0) @ W_U.T
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()]

    out = {}
    for l in layers:
        if l not in blob["J"]:
            continue
        J = blob["J"][l].float()
        st = spectrum_stats(J)
        w, V = torch.linalg.eig(J)
        a_ = w.abs()
        order = torch.argsort(a_, descending=True)
        dirs = []
        for j in order[:a.ndirs].tolist():
            lam = w[j]; v = V[:, j]
            is_c = lam.imag.abs() > 1e-6
            e = {"lambda_re": float(lam.real), "lambda_im": float(lam.imag),
                 "abs": float(lam.abs()), "complex": bool(is_c)}
            if is_c:
                # the invariant real plane; neither part alone is meaningful
                e["plane_a"] = read(v.real / v.real.norm(), a.k)
                e["plane_b"] = read(v.imag / v.imag.norm(), a.k)
                e["turn_per_layer_deg"] = float(lam.angle().abs() * 180 / 3.14159265)
            else:
                e["pos"] = read(v.real / v.real.norm(), a.k)
                e["neg"] = read(-v.real / v.real.norm(), a.k)
            dirs.append(e)
        out[str(l)] = {"stats": st, "dirs": dirs}
        print(f"layer {l:>3}  |lam|max {st['max_abs']:.2f}  near1 {st['near1']:>3}  "
              f"|lam|>1 {st['gt1']:>3}  complex {st['complex']:>3}  "
              f"mean rotation {st['rot_angle_mean']*180/3.14159:.1f} deg/layer",
              flush=True)
        for e in dirs[:3]:
            if e["complex"]:
                print(f"    lam={e['lambda_re']:+.2f}{e['lambda_im']:+.2f}j  "
                      f"turns {e['turn_per_layer_deg']:.0f} deg")
                print(f"      plane a: {' '.join(repr(t) for t in e['plane_a'][:6])}")
                print(f"      plane b: {' '.join(repr(t) for t in e['plane_b'][:6])}")
            else:
                print(f"    lam={e['lambda_re']:+.2f} (real)")
                print(f"      +: {' '.join(repr(t) for t in e['pos'][:6])}")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
