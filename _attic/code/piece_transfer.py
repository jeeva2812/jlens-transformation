"""RQ2: is legality one shared abstraction, or a bag of per-piece heuristics?

For each piece type, train a linear probe on the layer-l residual to predict that
piece's legal-origin mask -- "square s holds one of my pieces of type T, and it
has a legal move". Then measure TRANSFER: train on rooks, test on knights.

Transfer is the direct analogue of the fine-tuning experiment. Corrupting rook
legality can only move knight legality if the two share representation.

Two controls, both required:
  ceiling -- same piece type, held-out positions (how well anything transfers)
  floor   -- labels shuffled across positions, retrained
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import chess, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.chess_j import sample_game

MODEL = "austindavis/chess-gpt2-uci-8x8x512"
PIECES = [(chess.PAWN,"pawn"),(chess.KNIGHT,"knight"),(chess.BISHOP,"bishop"),
          (chess.ROOK,"rook"),(chess.QUEEN,"queen"),(chess.KING,"king")]


def masks(board):
    """legal-origin mask per piece type, for the side to move"""
    out = {n: [0.0]*64 for _, n in PIECES}
    for mv in board.legal_moves:
        pc = board.piece_at(mv.from_square)
        if pc is None: continue
        for pt, n in PIECES:
            if pc.piece_type == pt:
                out[n][mv.from_square] = 1.0
    return out


def fit(X, Y, ridge=1.0):
    """least-squares probe with ridge, X: n x d, Y: n x 64"""
    d = X.shape[1]
    A = X.T @ X + ridge*T.eye(d)
    return T.linalg.solve(A, X.T @ Y)


def auc(scores, labels):
    """rank-based AUC, pooled over all squares; ignores degenerate columns"""
    s = scores.flatten(); y = labels.flatten()
    npos, nneg = float((y==1).sum()), float((y==0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    r = T.argsort(T.argsort(s)).float() + 1
    return float((r[y==1].sum() - npos*(npos+1)/2) / (npos*nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("out/chess/piece_transfer.json"))
    a = ap.parse_args()

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    g = T.Generator().manual_seed(808)

    X, Y = [], {n: [] for _, n in PIECES}
    for gi in range(a.games):
        ids, boards = sample_game(m, tok, max_plies=26, gen=g)
        for k in range(1, min(len(ids), len(boards))):
            if k % 2 == 0: continue          # only origin-prediction positions
            t = T.tensor([ids[:k]])
            with T.no_grad():
                hs = m(input_ids=t, output_hidden_states=True).hidden_states
            X.append(hs[a.layer+1][0,-1].float())
            for n, v in masks(boards[k]).items(): Y[n].append(v)
        if (gi+1) % 30 == 0:
            print(f"  {gi+1}/{a.games} games, {len(X)} positions", flush=True)

    X = T.stack(X); Y = {n: T.tensor(v) for n, v in Y.items()}
    n = X.shape[0]
    X = T.cat([X - X.mean(0), T.ones(n,1)], 1)          # centre + bias column
    perm = T.randperm(n, generator=T.Generator().manual_seed(3))
    tr, te = perm[:int(.7*n)], perm[int(.7*n):]
    names = [nm for _, nm in PIECES]
    print(f"\nn = {n} origin-prediction positions, layer {a.layer}")
    print("base rates (fraction of squares that are a legal origin for that piece):")
    for nm in names: print(f"   {nm:>7}: {float(Y[nm].mean()):.4f}")

    W = {nm: fit(X[tr], Y[nm][tr]) for nm in names}
    Wsh = {}
    for nm in names:                                      # floor: shuffled labels
        sp = T.randperm(len(tr), generator=T.Generator().manual_seed(11))
        Wsh[nm] = fit(X[tr], Y[nm][tr][sp])

    T.save({"X": X, "Y": Y, "tr": tr, "te": te, "names": names}, "out/chess/probe_data.pt")
    M = [[auc(X[te] @ W[a_], Y[b_][te]) for b_ in names] for a_ in names]
    F = [auc(X[te] @ Wsh[a_], Y[a_][te]) for a_ in names]
    # CORRECT baseline: piece a's per-square MARGINAL alone, scored on piece b.
    # Shuffling positions leaves the marginal intact, so the shuffle control above
    # measures "how much does knowing the average board tell you" -- real
    # information, but not what transfer is supposed to mean. Position-specific
    # transfer is whatever a probe adds ON TOP of the marginal.
    MG = [[auc(Y[a_][tr].mean(0).expand(len(te), 64), Y[b_][te]) for b_ in names]
          for a_ in names]
    print(f"\nTRANSFER AUC  (row = probe trained on, column = tested on)")
    hdr = "train/test"
    print(f"{hdr:>11} " + " ".join(f"{b[:6]:>7}" for b in names) + f" {'shuf':>7}")
    for i, a_ in enumerate(names):
        print(f"{a_:>11} " + " ".join(
            (f"[{M[i][j]:.3f}]" if i==j else f"{M[i][j]:>7.3f}") for j in range(len(names)))
            + f" {F[i]:>7.3f}")
    print(f"\nMARGINAL-ONLY baseline (piece a's average board, scored on piece b)")
    print(f"{hdr:>11} " + " ".join(f"{b[:6]:>7}" for b in names))
    for i, a_ in enumerate(names):
        print(f"{a_:>11} " + " ".join(f"{MG[i][j]:>7.3f}" for j in range(len(names))))
    print(f"\nTRANSFER MINUS MARGINAL  -- the position-specific information")
    print(f"{hdr:>11} " + " ".join(f"{b[:6]:>7}" for b in names))
    for i, a_ in enumerate(names):
        print(f"{a_:>11} " + " ".join(
            (f"[{M[i][j]-MG[i][j]:+.3f}]" if i==j else f"{M[i][j]-MG[i][j]:>+7.3f}")
            for j in range(len(names))))
    dd = [M[i][i]-MG[i][i] for i in range(len(names))]
    oo = [M[i][j]-MG[i][j] for i in range(len(names)) for j in range(len(names)) if i!=j]
    print(f"\nceiling, above marginal (same piece): {sum(dd)/len(dd):+.3f}")
    print(f"cross-piece, above marginal        : {sum(oo)/len(oo):+.3f}")
    diag = [M[i][i] for i in range(len(names))]
    off  = [M[i][j] for i in range(len(names)) for j in range(len(names)) if i!=j]
    print(f"\nceiling (same piece, held out): {sum(diag)/len(diag):.3f}")
    print(f"cross-piece transfer          : {sum(off)/len(off):.3f}")
    print(f"floor (shuffled labels)       : {sum(F)/len(F):.3f}")
    sl = ["bishop","rook","queen"]
    sli = [(i,j) for i,x in enumerate(names) for j,y in enumerate(names)
           if i!=j and x in sl and y in sl]
    jmp = [(i,j) for i,x in enumerate(names) for j,y in enumerate(names)
           if i!=j and (x in sl) != (y in sl)]
    print(f"\nprediction 4 -- sliding<->sliding {sum(M[i][j] for i,j in sli)/len(sli):.3f}"
          f"   vs sliding<->other {sum(M[i][j] for i,j in jmp)/len(jmp):.3f}")
    a.out.write_text(json.dumps({"n": n, "layer": a.layer, "names": names,
                                 "transfer": M, "floor": F, "marginal": MG,
                                 "base": {nm: float(Y[nm].mean()) for nm in names}}))

if __name__ == "__main__":
    main()
