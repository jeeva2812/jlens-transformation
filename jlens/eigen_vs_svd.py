"""Head-to-head: do eigenvectors give more interpretable directions than singular vectors?

The eigen readouts LOOK cleaner -- layer 24's top eigenvector is six of six
gendered tokens where the corresponding singular direction gave mixed poles.
"Looks cleaner" is exactly the claim this project has already shown to be
unreliable, so score it the validated way: against held-out word-pair axes, with
a Bonferroni-corrected random-direction null.

Both families are scored identically and on the same layers, so the only thing
that differs is which decomposition produced the direction.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from jlens.axes import AXES, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ndirs", type=int, default=16)
    ap.add_argument("--nnull", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("out/eigen_vs_svd.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float(); norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(a.model); del m
    AX = build(tok)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)

    def logits(v):
        with torch.no_grad():
            return norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T

    def score(v):
        lg = logits(v)
        return {k: (lg[B].mean() - lg[A].mean()).item() for k, (A, B) in AX.items()}

    rows, summary = [], []
    for l in a.layers:
        J = blob["J"][l].float()
        d = J.shape[0]
        g = torch.Generator().manual_seed(0)
        null = {k: [] for k in AX}
        for _ in range(a.nnull):
            s = score(torch.randn(d, generator=g))
            for k, v in s.items():
                null[k].append(abs(v))
        q = 1 - 0.01 / len(AX)
        thr = {k: torch.tensor(v).quantile(q).item() for k, v in null.items()}

        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        order = torch.argsort(w.abs(), descending=True)

        fams = {}
        fams["SVD u"] = [U[:, i] for i in range(a.ndirs)]
        fams["SVD v"] = [Vh[i] for i in range(a.ndirs)]
        ev = []
        for j in order[:a.ndirs].tolist():
            v = V[:, j]
            ev.append(v.real if w[j].imag.abs() < 1e-6 else v.real)  # plane a
        fams["eigen"] = ev

        line = {"layer": l}
        for name, dirs in fams.items():
            hits, best = 0, []
            for i, v in enumerate(dirs):
                v = v / v.norm()
                s = score(v)
                h = [(k, val, abs(val)/thr[k]) for k, val in s.items()
                     if abs(val) > thr[k]]
                h.sort(key=lambda t: -t[2])
                if h:
                    hits += 1
                    best.append(h[0][2])
                rows.append({"layer": l, "family": name, "i": i,
                             "hit": bool(h), "axis": h[0][0] if h else None,
                             "ratio": round(h[0][2], 2) if h else 0.0})
            line[name] = {"hits": hits, "n": len(dirs),
                          "mean_ratio": round(sum(best)/len(best), 2) if best else 0.0}
        summary.append(line)
        print(f"layer {l:>3}  " + "   ".join(
            f"{k}: {v['hits']}/{v['n']} (x{v['mean_ratio']})"
            for k, v in line.items() if k != "layer"), flush=True)

    tot = {}
    for name in ["SVD u", "SVD v", "eigen"]:
        h = sum(r["hit"] for r in rows if r["family"] == name)
        n = sum(1 for r in rows if r["family"] == name)
        rr = [r["ratio"] for r in rows if r["family"] == name and r["hit"]]
        tot[name] = {"hits": h, "n": n, "rate": round(h/n, 3),
                     "mean_ratio": round(sum(rr)/len(rr), 2) if rr else 0}
    print("\nTOTAL across all layers (expected by chance: 1%)")
    for k, v in tot.items():
        print(f"  {k:<8} {v['hits']:>3}/{v['n']}  = {v['rate']*100:>5.1f}%   "
              f"mean strength {v['mean_ratio']}x threshold")
    a.out.write_text(json.dumps({"rows": rows, "summary": summary, "total": tot}))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
