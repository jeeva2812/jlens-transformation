"""Do component COEFFICIENTS track chess concepts?

A direction u_i is a fixed board, so it cannot mean "a capture is available" --
that is a property of a position, not of a square set. The position-dependent
half of the decomposition is the coefficient c_i(h) = sigma_i <v_i, h>. So the
place to look for checks, captures and piece mobility is there.

For each component we correlate its coefficient across plies against oracle
labels from python-chess. Two controls, because both matter:
  - ply number is partialled out, since almost every chess property drifts with
    game phase and would otherwise manufacture correlations;
  - the significance threshold comes from a permutation null on the MAX |r| over
    the whole family of tests, not from a per-test p-value.
"""
from __future__ import annotations
import json
from pathlib import Path
import chess, torch

def labels_for(board: chess.Board, kind: str):
    legal = list(board.legal_moves)
    caps = [m for m in legal if board.is_capture(m)]
    checks = 0
    for m in legal:
        board.push(m); checks += board.is_check(); board.pop()
    movers = {board.piece_at(m.from_square).piece_type for m in legal
              if board.piece_at(m.from_square)}
    return {
        "mobility": len(legal),
        "has_capture": float(len(caps) > 0),
        "n_captures": len(caps),
        "in_check": float(board.is_check()),
        "can_check": float(checks > 0),
        "n_checks": checks,
        "knight_can_move": float(chess.KNIGHT in movers),
        "rook_can_move": float(chess.ROOK in movers),
        "queen_can_move": float(chess.QUEEN in movers),
        "material": sum((1 if p.piece_type == chess.PAWN else
                         3 if p.piece_type in (chess.KNIGHT, chess.BISHOP) else
                         5 if p.piece_type == chess.ROOK else
                         9 if p.piece_type == chess.QUEEN else 0)
                        * (1 if p.color == board.turn else -1)
                        for p in board.piece_map().values()),
        "is_origin": 1.0 if kind == "origin" else 0.0,
    }

def resid(y, x):
    """y with the linear trend in x removed."""
    n = len(y); mx = sum(x)/n; my = sum(y)/n
    sxx = sum((a-mx)**2 for a in x) or 1e-9
    b = sum((a-mx)*(c-my) for a, c in zip(x, y))/sxx
    return [c-my-b*(a-mx) for a, c in zip(x, y)]

def corr(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    va = sum((x-ma)**2 for x in a); vb = sum((x-mb)**2 for x in b)
    if va < 1e-12 or vb < 1e-12: return 0.0
    return sum((x-ma)*(y-mb) for x, y in zip(a, b))/(va*vb)**.5

def main():
    E = json.load(open("out/chess/explorer.json"))
    plies = [p for p in E["plies"] if "coef" in p]
    layers = [str(l) for l in E["layers"]]
    NC = 24
    ply_no = [p["ply"] for p in plies]
    L = {}
    for p in plies:
        d = labels_for(chess.Board(p["fen"]), p["kind"])
        for k, v in d.items():
            L.setdefault(k, []).append(float(v))
    names = list(L)
    Lr = {k: resid(v, ply_no) for k, v in L.items()}
    C = {l: [resid([p["coef"][l][i] for p in plies], ply_no) for i in range(NC)]
         for l in layers}

    obs = []
    for l in layers:
        for i in range(NC):
            for k in names:
                obs.append((abs(corr(C[l][i], Lr[k])), l, i, k))
    obs.sort(reverse=True)

    g = torch.Generator().manual_seed(31)
    maxs = []
    for _ in range(500):
        perm = torch.randperm(len(plies), generator=g).tolist()
        best = 0.0
        for l in layers:
            for i in range(NC):
                c = C[l][i]
                cp = [c[j] for j in perm]
                for k in names:
                    best = max(best, abs(corr(cp, Lr[k])))
        maxs.append(best)
    maxs.sort()
    thr95 = maxs[int(.95*len(maxs))]
    print(f"n = {len(plies)} plies, {len(obs)} tests "
          f"({len(layers)} layers x {NC} components x {len(names)} labels)")
    print(f"permutation null on max|r| over the whole family: "
          f"median {maxs[len(maxs)//2]:.3f}, 95th pct {thr95:.3f}\n")
    hits = [o for o in obs if o[0] > thr95]
    print(f"{'|r|':>6} {'layer':>6} {'comp':>5}  label")
    for r, l, i, k in obs[:15]:
        print(f"{r:>6.3f} {('L'+l):>6} {('c'+str(i)):>5}  {k}"
              + ("   <-- beats family-wise null" if r > thr95 else ""))
    print(f"\n{len(hits)}/{len(obs)} exceed the family-wise 95% threshold "
          f"(expected by chance: 5% of ONE max, i.e. ~0)")
    Path("out/chess/concepts.json").write_text(json.dumps(
        {"thr95": round(thr95, 4), "labels": names,
         "top": [{"r": round(r, 4), "layer": l, "comp": i, "label": k}
                 for r, l, i, k in obs[:60]]}))

if __name__ == "__main__":
    main()
