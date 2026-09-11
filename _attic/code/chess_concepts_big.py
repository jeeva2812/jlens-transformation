"""The coefficient-vs-concept test at a sample size that can actually see checks.

One 60-ply game gives 2 positions with a check available and makes
queen_can_move constant at 95%. This samples many games so the rare labels have
support, and reports each label's base rate alongside every correlation so a
degenerate one cannot be mistaken for a finding again.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import chess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.chess_j import sample_game
from jlens.chess_concepts import labels_for, resid, corr

MODEL = "austindavis/chess-gpt2-uci-8x8x512"; SQ0 = 4; NC = 24
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 150

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p in m.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
_, ln = _find_blocks_and_norm(m)
blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
target = blob["target"]; layers = [l for l in blob["layers"] if l < target]
SV = {l: torch.linalg.svd(blob["J"][l].float()) for l in layers}

g = torch.Generator().manual_seed(101)
rows, labs = {str(l): [[] for _ in range(NC)] for l in layers}, {}
plyno = []
for gi in range(N_GAMES):
    ids, boards = sample_game(m, tok, max_plies=26, gen=g)
    for k in range(1, min(len(ids), len(boards))):
        t = torch.tensor([ids[:k]])
        with _MultiCapture(m, layers, target) as cap:
            with torch.no_grad():
                m(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False)
            hs = {l: cap.h[l][0, -1].detach().clone() for l in layers}
        d = labels_for(boards[k], "origin" if k % 2 == 1 else "destination")
        for kk, v in d.items():
            labs.setdefault(kk, []).append(float(v))
        plyno.append(k)
        for l in layers:
            U, S, Vh = SV[l]
            c = S * (Vh @ hs[l])
            for i in range(NC):
                rows[str(l)][i].append(float(c[i]))
    if (gi + 1) % 25 == 0:
        print(f"  {gi+1}/{N_GAMES} games, {len(plyno)} positions", flush=True)

names = list(labs)
print(f"\nn = {len(plyno)} positions from {N_GAMES} games")
print(f"{'label':>18} {'base rate / mean':>18}")
for k in names:
    v = labs[k]
    b = f"{sum(v)/len(v):.1%} true" if set(v) <= {0.0, 1.0} else f"mean {sum(v)/len(v):.2f}"
    print(f"{k:>18} {b:>18}")

# ---- vectorised: pure-Python correlation over ~8k positions x 1848 tests x 200
# permutations is billions of ops. Same statistics, done with matrix products.
import torch as T
plyv = T.tensor(plyno, dtype=T.float32)
def dtrend(M):
    """remove the linear trend in ply number from every column of M"""
    x = plyv - plyv.mean()
    return M - M.mean(0) - T.outer(x, (x @ (M - M.mean(0))) / (x @ x))
def zs(M):
    M = M - M.mean(0)
    return M / (M.norm(dim=0, keepdim=True) + 1e-9)
Lm = zs(dtrend(T.tensor([labs[k] for k in names], dtype=T.float32).T))   # n x n_labels
keys = [(l, i) for l in rows for i in range(NC)]
Cm = zs(dtrend(T.tensor([rows[l][i] for l, i in keys], dtype=T.float32).T))  # n x n_comp
R = (Cm.T @ Lm).abs()                                                    # n_comp x n_labels
Path("out/chess/raw_concepts.pt").write_text("") if False else None
T.save({"Cm": Cm, "Lm": Lm, "keys": keys, "names": names, "plyno": plyno,
        "labs": labs}, "out/chess/raw_concepts.pt")

gg = T.Generator().manual_seed(53)
maxs = []
for _ in range(400):
    perm = T.randperm(Cm.shape[0], generator=gg)
    maxs.append(float((Cm[perm].T @ Lm).abs().max()))
maxs.sort(); thr = maxs[int(.95 * len(maxs))]
print(f"\npermutation null on max|r| over {R.numel()} tests "
      f"({len(maxs)} permutations): 95th pct {thr:.3f}")

flat = sorted(((float(R[a, b]), keys[a][0], keys[a][1], names[b])
               for a in range(R.shape[0]) for b in range(R.shape[1])), reverse=True)
print(f"\n{'|r|':>6} {'layer':>6} {'comp':>5}  {'label':>18}  base rate")
for r, l, i, k in flat[:20]:
    v = labs[k]
    b = f"{sum(v)/len(v):.0%} true" if set(v) <= {0.0, 1.0} else "continuous"
    print(f"{r:>6.3f} {('L'+l):>6} {('c'+str(i)):>5}  {k:>18}  {b}"
          + ("   <-- beats null" if r > thr else ""))
hits = [o for o in flat if o[0] > thr]
print(f"\n{len(hits)}/{len(flat)} exceed the family-wise threshold")
Path("out/chess/concepts_big.json").write_text(json.dumps(
    {"n": len(plyno), "games": N_GAMES, "thr95": round(thr, 4),
     "base": {k: round(sum(v)/len(v), 4) for k, v in labs.items()},
     "top": [{"r": round(r, 4), "layer": l, "comp": i, "label": k}
             for r, l, i, k in flat[:120]]}))
