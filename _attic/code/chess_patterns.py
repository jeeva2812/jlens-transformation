"""Do the singular directions carry board patterns, or are they arbitrary?

Regress each direction's 64-square readout onto a basis of board patterns a
chess player would name -- files, ranks, centre, colour complex, long diagonals,
back rank -- and report both the individual loadings and the total R^2. A high
R^2 means the direction IS simple geometry; a low one means it carries something
the basis cannot express.

Null: the same regression on random directions put through the same readout,
which fixes the R^2 you get for free from a 16-parameter fit to 64 points.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from jlens.lens import _find_blocks_and_norm

MODEL = "austindavis/chess-gpt2-uci-8x8x512"; SQ0 = 4; TOPK = 16
FILES = "abcdefgh"

def basis():
    cols, names = [], []
    for f in range(8):
        cols.append([1.0 if s % 8 == f else 0.0 for s in range(64)]); names.append(FILES[f])
    for r in range(8):
        cols.append([1.0 if s // 8 == r else 0.0 for s in range(64)]); names.append(str(r + 1))
    cols.append([1.0 if (2 <= s % 8 <= 5 and 2 <= s // 8 <= 5) else 0.0 for s in range(64)])
    names.append("centre")
    cols.append([1.0 if ((s % 8 + s // 8) % 2) else 0.0 for s in range(64)]); names.append("light")
    cols.append([1.0 if s % 8 == s // 8 else 0.0 for s in range(64)]); names.append("a1h8")
    cols.append([1.0 if s % 8 + s // 8 == 7 else 0.0 for s in range(64)]); names.append("a8h1")
    cols.append([1.0 if s // 8 in (0, 7) else 0.0 for s in range(64)]); names.append("backrank")
    X = torch.tensor(cols).T
    X = X - X.mean(0)
    return X, names

def main():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    _, ln = _find_blocks_and_norm(m)
    W_U = m.get_output_embeddings().weight.detach().float()
    blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
    target = blob["target"]; layers = [l for l in blob["layers"] if l < target]
    X, names = basis()
    Xp = torch.linalg.pinv(X)

    def rd(v):
        with torch.no_grad():
            return (ln(v.float().unsqueeze(0)).squeeze(0) @ W_U.T)[SQ0:SQ0 + 64]

    def fit(y):
        y = y - y.mean()
        beta = Xp @ y
        r2 = 1 - float(((y - X @ beta) ** 2).sum() / ((y ** 2).sum() + 1e-12))
        corr = []
        for j in range(X.shape[1]):
            x = X[:, j]
            corr.append(float((x @ y) / (x.norm() * y.norm() + 1e-12)))
        return r2, corr

    g = torch.Generator().manual_seed(23)
    nulls = []
    for _ in range(400):
        v = torch.randn(512, generator=g); v /= v.norm()
        nulls.append(fit(rd(v))[0])
    nulls.sort()
    thr = nulls[int(.99 * len(nulls))]
    med = nulls[len(nulls) // 2]
    print(f"null R^2 from a {X.shape[1]}-term fit to 64 points: "
          f"median {med:.3f}, 99th pct {thr:.3f}\n")

    out = {"names": names, "null_med": round(med, 3), "null_thr": round(thr, 3), "rows": {}}
    print(f"{'layer':>5} {'i':>3} {'sigma':>6} {'R^2':>6}  strongest loadings")
    for l in layers:
        U, S, Vh = torch.linalg.svd(blob["J"][l].float())
        rows = []
        for i in range(TOPK):
            r2, corr = fit(rd(U[:, i]))
            rows.append({"i": i, "sigma": round(float(S[i]), 3), "r2": round(r2, 3),
                         "corr": [round(c, 3) for c in corr]})
            if i < 4:
                top = sorted(range(len(names)), key=lambda j: -abs(corr[j]))[:4]
                lab = "  ".join(f"{names[j]}{corr[j]:+.2f}" for j in top)
                print(f"{l:>5} {i:>3} {float(S[i]):>6.2f} {r2:>6.3f}  {lab}")
        out["rows"][str(l)] = rows
    Path("out/chess/patterns.json").write_text(json.dumps(out))
    allr2 = [r["r2"] for v in out["rows"].values() for r in v]
    beat = sum(1 for r in allr2 if r > thr)
    print(f"\nmean R^2 across {len(allr2)} directions: {sum(allr2)/len(allr2):.3f}"
          f"   |   {beat}/{len(allr2)} exceed the 99th-pct null ({thr:.3f})")

if __name__ == "__main__":
    main()
