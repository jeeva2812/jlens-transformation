"""Does J + lambda*I improve readouts? Testing someone else's fix on our metric.

A LessWrong analysis of J-Lens (Anthropic's) reports that the Jacobian's
dominant channels carry ~10x the gain of the residual pathway and misweight
structural tokens, and proposes a shrinkage regulariser J + lambda*I that
recovers faithfulness.

That is a fix for a problem this project characterised from the other side: our
top singular directions read as newlines, punctuation and code junk, and the
axis probes said 20.8% of them are about anything nameable against 39.6% for
eigenvectors. If shrinkage helps, it should raise the hit rate -- because adding
lambda*I pulls the decomposition back toward the pass-through and away from the
high-gain channels that carry the junk.

Swept over lambda for both families, scored on the same axis probes with the
same Bonferroni-corrected null, so the numbers are directly comparable to
everything else here. lambda = 0 reproduces our published figures and is the
control.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from jlens.axes import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ndirs", type=int, default=16)
    ap.add_argument("--nnull", type=int, default=300)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", type=Path, default=Path("out/shrinkage.json"))
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

    # the null does not depend on lambda -- it is over random directions
    d = blob["J"][a.layers[0]].shape[0]
    g = torch.Generator().manual_seed(0)
    null = {k: [] for k in AX}
    for _ in range(a.nnull):
        s = score(torch.randn(d, generator=g))
        for k, v in s.items():
            null[k].append(abs(v))
    q = 1 - 0.01 / len(AX)
    thr = {k: torch.tensor(v).quantile(q).item() for k, v in null.items()}

    res = {}
    print(f"axis-probe hit rate, by shrinkage lambda  (lambda=0 is our published number)\n")
    print(f"{'lambda':>8}{'SVD u':>10}{'SVD v':>10}{'eigen':>10}"
          f"{'||J+lI||/||J||':>16}")
    print("-" * 56)
    for lam in a.lams:
        hits = {"SVD u": 0, "SVD v": 0, "eigen": 0}
        tot = {k: 0 for k in hits}
        scale = []
        for l in a.layers:
            if l not in blob["J"]:
                continue
            J0 = blob["J"][l].float()
            I = torch.eye(J0.shape[0])
            J = J0 + lam * I
            scale.append(float(J.norm() / J0.norm()))
            U, S, Vh = torch.linalg.svd(J)
            w, V = torch.linalg.eig(J)
            eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
            fams = {"SVD u": [U[:, i] for i in range(a.ndirs)],
                    "SVD v": [Vh[i] for i in range(a.ndirs)],
                    "eigen": [V[:, j].real for j in eo.tolist()]}
            for nm, dirs in fams.items():
                for v in dirs:
                    v = v / v.norm()
                    s = score(v)
                    if any(abs(val) > thr[k] for k, val in s.items()):
                        hits[nm] += 1
                    tot[nm] += 1
        res[lam] = {nm: {"hits": hits[nm], "n": tot[nm],
                         "rate": hits[nm] / max(tot[nm], 1)} for nm in hits}
        print(f"{lam:>8.2f}"
              + "".join(f"{hits[nm]/max(tot[nm],1)*100:>9.1f}%" for nm in
                        ["SVD u", "SVD v", "eigen"])
              + f"{sum(scale)/len(scale):>16.3f}", flush=True)

    a.out.write_text(json.dumps(res, indent=1))
    best = {nm: max(res, key=lambda L: res[L][nm]["rate"])
            for nm in ["SVD u", "SVD v", "eigen"]}
    print("\nbest lambda per family:")
    for nm, L in best.items():
        print(f"  {nm:<8} lambda={L:<5} {res[L][nm]['rate']*100:.1f}%  "
              f"(vs {res[0.0][nm]['rate']*100:.1f}% at lambda=0)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
