"""Eigen-decomposition data for the explorer, alongside the SVD data.

Stored separately from explorer_data so the two can be browsed side by side and
compared on the same checkpoint and layer. Each entry keeps lambda (real and
imaginary), whether the direction is part of a rotation pair, and the readout --
for a complex pair the two vectors spanning its invariant real plane, since
neither the real nor imaginary part alone is a direction the model has.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from jlens.explorer_data import CONFIGS
from jlens.axes import build

OUT = Path("out/explorer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--ndirs", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nnull", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    a = ap.parse_args()
    C = dict(CONFIGS[a.model])
    if a.layers:
        C["layers"] = a.layers
    W_U, norm, tok = C["head"]()
    AX = build(tok)

    def logits(v):
        with torch.no_grad():
            return norm(v.float()) @ W_U.T

    def read(v, k):
        lg = logits(v)
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()], lg

    Js, order = {}, []
    for rev, label in C["ckpts"]:
        p = Path(C["path"](rev))
        if p.exists():
            Js[rev] = torch.load(p, map_location="cpu", weights_only=False)["J"]
            order.append({"id": rev, "label": label})

    base = order[0]["id"]
    out = {"model": a.model, "layers": C["layers"], "checkpoints": order,
           "eigen": {}}
    for l in C["layers"]:
        Jb = Js[base][l].float()
        g = torch.Generator().manual_seed(0)
        null_l = torch.stack([logits(Jb @ torch.randn(Jb.shape[0], generator=g))
                              for _ in range(a.nnull)])
        nm, ns = null_l.mean(0), null_l.std(0) + 1e-6
        nullax = {k: [] for k in AX}
        for _ in range(a.nnull):
            v = torch.randn(Jb.shape[0], generator=g)
            lg = logits(v)
            for k, (A, B) in AX.items():
                nullax[k].append(abs((lg[B].mean() - lg[A].mean()).item()))
        q = 1 - 0.01 / len(AX)
        thr = {k: torch.tensor(v).quantile(q).item() for k, v in nullax.items()}

        for rev in Js:
            J = Js[rev][l].float()
            w, V = torch.linalg.eig(J)
            idx = torch.argsort(w.abs(), descending=True)[:a.ndirs]
            dirs = []
            for j in idx.tolist():
                lam, v = w[j], V[:, j]
                cx = bool(lam.imag.abs() > 1e-6)
                vr = v.real / v.real.norm()
                toks, lg = read(vr, a.k)
                z = ((lg - nm) / ns).topk(a.k).values.mean().item()
                s = {k: (lg[B].mean() - lg[A].mean()).item() for k, (A, B) in AX.items()}
                hits = sorted(((k, val) for k, val in s.items() if abs(val) > thr[k]),
                              key=lambda t: -abs(t[1]) / thr[t[0]])
                e = {"lam_re": float(lam.real), "lam_im": float(lam.imag),
                     "abs": float(lam.abs()), "complex": cx, "z": round(z, 2),
                     "pos": toks, "neg": read(-vr, a.k)[0],
                     "axes": [{"name": k, "score": round(val, 2),
                               "ratio": round(abs(val)/thr[k], 2)} for k, val in hits[:3]]}
                if cx:
                    e["turn_deg"] = round(float(lam.angle().abs() * 57.2958), 1)
                    e["plane_b"] = read(v.imag / v.imag.norm(), a.k)[0]
                dirs.append(e)
            I = torch.eye(J.shape[0])
            aw = w.abs()
            out["eigen"][f"{rev}|{l}"] = {
                "spectrum": [round(x, 4) for x in aw.sort(descending=True)[0][:40].tolist()],
                "near1": int(((aw - 1).abs() < 0.15).sum()),
                "gt1": int((aw > 1).sum()),
                "n_complex": int((w.imag.abs() > 1e-6).sum()),
                "rot_deg": round(float(w[w.imag.abs() > 1e-6].angle().abs().mean()
                                       * 57.2958), 1) if (w.imag.abs() > 1e-6).any() else 0,
                "dirs": dirs}
            print(f"  [eig] {rev} L{l}", flush=True)
    f = OUT / f"{a.model}_eigen.json"
    f.write_text(json.dumps(out))
    print(f"wrote {f}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
