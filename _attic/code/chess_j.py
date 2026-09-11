"""J-Lens on a chess transformer, where the readout IS a chessboard.

austindavis/chess-gpt2-uci-8x8x512: GPT2, 8 layers, d_model 512, vocab 72.
The vocabulary is 4 specials + the 64 squares a1..h8 + 4 promotion pieces, one
token per square, so "e2e4" is two tokens. That means the J-Lens readout

    softmax(W_U @ ln_f(J_l d))

is a distribution over the 64 squares -- an 8x8 heatmap. Every direction of J
renders as a picture of a board, with no top-k truncation and nothing to label.

J itself is 512x512 (residual -> residual). It is the READOUT that is 8x8.

Square indexing: tokenizer id = 4 + square, with square = rank*8 + file and
a1 = 0, matching python-chess exactly. Asserted below rather than trusted.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "austindavis/chess-gpt2-uci-8x8x512"
SQ0 = 4          # token id of a1
DEV = "cpu"      # 512-dim; MPS gives no useful win and autograd is fussier


def check_indexing(tok):
    for name in ("a1", "e2", "e4", "h8", "d5"):
        assert tok.convert_tokens_to_ids(name) == SQ0 + chess.parse_square(name), name
    assert tok.convert_tokens_to_ids("h8") == SQ0 + 63


@torch.no_grad()
def sample_game(model, tok, max_plies=40, temp=0.8, gen=None):
    """Play a game by sampling the model, restricted to legal moves.

    Restricting to legal keeps the board state known and the prefix on
    distribution; sampling (rather than argmax) keeps the games varied.
    Returns (token_ids, boards) where boards[i] is the position facing the model
    when it is about to emit token_ids[i].
    """
    board = chess.Board()
    ids = [tok.bos_token_id]
    boards = [board.copy()]
    for _ in range(max_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        froms = sorted({m.from_square for m in legal})
        lf = torch.tensor([logits[SQ0 + s] for s in froms]) / temp
        frm = froms[int(torch.multinomial(lf.softmax(0), 1, generator=gen))]
        ids2 = ids + [SQ0 + frm]
        logits2 = model(input_ids=torch.tensor([ids2])).logits[0, -1]
        tos = sorted({m.to_square for m in legal if m.from_square == frm})
        lt = torch.tensor([logits2[SQ0 + s] for s in tos]) / temp
        to = tos[int(torch.multinomial(lt.softmax(0), 1, generator=gen))]
        mv = chess.Move(frm, to)
        if mv not in board.legal_moves:                    # promotion
            mv = chess.Move(frm, to, promotion=chess.QUEEN)
            if mv not in board.legal_moves:
                break
        boards.append(board.copy())                        # facing the 'to' token
        board.push(mv)
        ids = ids + [SQ0 + frm, SQ0 + to]
        boards.append(board.copy())
    return ids, boards


def legality_check(model, tok, n=20, gen=None):
    """End-to-end sanity: how often is the model's unrestricted argmax legal?"""
    ok = tot = 0
    for _ in range(n):
        board = chess.Board()
        ids = [tok.bos_token_id]
        for _ in range(12):
            legal = list(board.legal_moves)
            if not legal:
                break
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([ids])).logits[0, -1]
            frm = int(lg[SQ0:SQ0 + 64].argmax())
            tot += 1
            ok += any(m.from_square == frm for m in legal)
            mv = list(board.legal_moves)[int(torch.randint(len(legal), (1,), generator=gen))]
            board.push(mv)
            ids += [SQ0 + mv.from_square, SQ0 + mv.to_square]
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--plies", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("out/chess"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from jlens.lens import jacobians_all_layers, _find_blocks_and_norm

    tok = AutoTokenizer.from_pretrained(MODEL)
    check_indexing(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, norm_mod = _find_blocks_and_norm(model)
    n_layer = len(blocks)
    target = n_layer - 1
    layers = list(range(target + 1))          # include target for the identity check
    print(f"{MODEL}: {n_layer} blocks, target {target}", flush=True)

    gen = torch.Generator().manual_seed(0)
    acc = legality_check(model, tok, n=10, gen=gen)
    print(f"argmax-from-square legality: {acc:.1%}  (sanity: should be high)", flush=True)

    games = []
    for i in range(a.games):
        ids, boards = sample_game(model, tok, max_plies=a.plies, gen=gen)
        games.append((ids, boards))
        if (i + 1) % 10 == 0:
            print(f"  sampled {i+1}/{a.games} games", flush=True)

    def batches():
        for i, (ids, _) in enumerate(games):
            t = torch.tensor([ids], device=DEV)
            print(f"  J prompt {i+1}/{len(games)} (len {len(ids)})", flush=True)
            yield t, torch.ones_like(t)

    Js = jacobians_all_layers(model, batches(), layers, target, chunk=a.chunk)
    Js = {l: v.cpu().float() for l, v in Js.items()}

    ident = (Js[target] - torch.eye(Js[target].shape[0])).abs().max()
    print(f"boundary check  max|J[target] - I| = {ident:.3e}   (must be ~0)", flush=True)

    torch.save({"model": MODEL, "layers": layers, "target": target,
                "J": {l: Js[l].to(torch.float16) for l in layers},
                "legality_acc": acc},
               a.out / "J_chess.pt")

    # keep a handful of real positions for the figure
    ids, boards = games[0]
    json.dump({"ids": ids, "fens": [b.fen() for b in boards]},
              open(a.out / "sample_game.json", "w"))
    print(f"[saved] {a.out/'J_chess.pt'}", flush=True)


if __name__ == "__main__":
    main()
