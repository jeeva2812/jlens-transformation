"""Render J-Lens chess readouts as heatmaps on an actual board.

For a direction d in layer-l residual space the readout is

    z = W_U @ ln_f(J_l d)        (72 logits)
    board = z[4:68].reshape(8, 8) with a1 = index 0

which is a real 8x8 chessboard, so we draw it as one: files a-h left to right,
rank 8 at the top, diverging colour about zero, piece glyphs overlaid where a
position is being shown.
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "austindavis/chess-gpt2-uci-8x8x512"
SQ0 = 4
GLYPH = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕",
         "K": "♔", "p": "♟", "n": "♞", "b": "♝", "r": "♜",
         "q": "♛", "k": "♚"}


def colour(v, vmax):
    """Diverging blue(-) / white(0) / red(+), normalised by vmax."""
    if vmax <= 0:
        return "#ffffff"
    t = max(-1.0, min(1.0, v / vmax))
    if t >= 0:
        r, g, b = 255, int(255 * (1 - 0.75 * t)), int(255 * (1 - 0.85 * t))
    else:
        r, g, b = int(255 * (1 + 0.85 * t)), int(255 * (1 + 0.45 * t)), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def board_svg(vals, title="", fen=None, mask=None, size=176):
    """vals: length-64 tensor/list indexed a1=0..h8=63. mask: set of squares to ring."""
    c = size / 8.0
    vmax = max(abs(float(v)) for v in vals) or 1.0
    pieces = {}
    if fen:
        b = chess.Board(fen)
        for sq, pc in b.piece_map().items():
            pieces[sq] = GLYPH[pc.symbol()]
    out = [f'<svg viewBox="0 0 {size+18} {size+22}" class="bd">']
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x, y = 18 + f * c, (7 - r) * c            # rank 8 at top
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{c:.1f}" height="{c:.1f}" '
                   f'fill="{colour(float(vals[sq]), vmax)}" stroke="#d8d8d8" stroke-width="0.4"/>')
        if mask is not None and sq in mask:
            out.append(f'<rect x="{x+1.2:.1f}" y="{y+1.2:.1f}" width="{c-2.4:.1f}" '
                       f'height="{c-2.4:.1f}" fill="none" stroke="#111" stroke-width="1.6"/>')
        if sq in pieces:
            out.append(f'<text x="{x+c/2:.1f}" y="{y+c*0.72:.1f}" text-anchor="middle" '
                       f'font-size="{c*0.78:.1f}" opacity="0.75">{pieces[sq]}</text>')
    for i in range(8):
        out.append(f'<text x="9" y="{(7-i)*c+c*0.65:.1f}" font-size="7" fill="#888" '
                   f'text-anchor="middle">{i+1}</text>')
        out.append(f'<text x="{18+i*c+c/2:.1f}" y="{size+9:.1f}" font-size="7" fill="#888" '
                   f'text-anchor="middle">{"abcdefgh"[i]}</text>')
    out.append(f'<text x="{(size+18)/2:.1f}" y="{size+19:.1f}" font-size="7.5" '
               f'fill="#333" text-anchor="middle">{html.escape(title)}</text>')
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpath", type=Path, default=Path("out/chess/J_chess.pt"))
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/CHESS.html"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    from jlens.lens import _find_blocks_and_norm
    _, norm_mod = _find_blocks_and_norm(model)
    norm_mod = norm_mod.eval()
    W_U = model.get_output_embeddings().weight.detach().float()

    blob = torch.load(a.jpath, map_location="cpu", weights_only=False)
    layers = [l for l in blob["layers"] if l < blob["target"]]
    sample = json.load(open(a.jpath.parent / "sample_game.json"))

    def readout(vec):
        with torch.no_grad():
            z = norm_mod(vec.float().unsqueeze(0)).squeeze(0) @ W_U.T
        return z[SQ0:SQ0 + 64]

    P = ['<meta charset="utf-8"><title>J-Lens on ChessGPT</title><style>'
         'body{font:13px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;'
         'margin:0;padding:28px 34px;background:#fbfbfa;color:#1a1a1a;max-width:1500px}'
         'h1{font-size:21px;margin:0 0 4px}h2{font-size:15px;margin:30px 0 6px;'
         'border-top:1px solid #e3e3e0;padding-top:14px}'
         'p{max-width:74ch;color:#444}code{background:#f0efec;padding:1px 4px;border-radius:3px;'
         'font-size:12px}.row{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}'
         '.bd{width:176px;height:auto;background:#fff;border:1px solid #e3e3e0;border-radius:4px}'
         '.lede{color:#555;font-size:13.5px}.k{color:#777;font-size:12px;margin:2px 0 8px}'
         '</style>']
    P.append("<h1>J-Lens on ChessGPT</h1>")
    P.append(f'<p class="lede">austindavis/chess-gpt2-uci-8x8x512 &mdash; GPT-2, 8 blocks, '
             f'd_model 512, vocab 72 (4 specials + 64 squares + 4 promotion pieces). '
             f'<b>J is 512&times;512</b>; what is 8&times;8 is the <i>readout</i> '
             f'<code>W_U &middot; ln_f(J d)</code>, whose 64 square-logits are a board. '
             f'Red = raised, blue = suppressed. J averaged over '
             f'{len(sample["ids"])//2}-ply games, boundary check '
             f'<code>max|J[target]&minus;I|</code> reported in the run log.</p>')

    # --- reference panel: a real position, its legal origins, and the model's own readout
    mid = min(len(sample["fens"]) - 1, 2 * (len(sample["fens"]) // 4))
    fen = sample["fens"][mid]
    board = chess.Board(fen)
    legal_from = {m.from_square for m in board.legal_moves}
    P.append("<h2>Reference: what ground truth looks like</h2>")
    P.append('<p>Left: the position. Middle: the squares a legal move can start from '
             '(<code>python-chess</code>, exact). Right: the model\'s own next-token '
             'distribution at that position. The oracle in the middle is what every '
             'direction below can be scored against &mdash; no labelling, no LLM judge.</p>')
    ones = [0.0] * 64
    legal_vals = [1.0 if s in legal_from else 0.0 for s in range(64)]
    ids = sample["ids"][:mid + 1]
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids])).logits[0, -1][SQ0:SQ0 + 64]
    P.append('<div class="row">')
    P.append(board_svg(ones, "position", fen=fen))
    P.append(board_svg(legal_vals, "legal origin squares (oracle)", mask=legal_from))
    P.append(board_svg(lg, "model next-token logits", mask=legal_from))
    P.append("</div>")

    # --- per-layer top singular / eigen directions
    for L in layers:
        J = blob["J"][L].float()
        U, S, Vh = torch.linalg.svd(J)
        P.append(f"<h2>Layer {L} &rarr; {blob['target']}</h2>")
        P.append(f'<p class="k">&sigma;<sub>1..{a.topk}</sub> = '
                 + ", ".join(f"{float(s):.2f}" for s in S[:a.topk]) + "</p>")
        P.append('<div class="row">')
        for i in range(a.topk):
            P.append(board_svg(readout(U[:, i]), f"u{i}  σ={float(S[i]):.2f}"))
        P.append("</div>")
        w, V = torch.linalg.eig(J)
        order = torch.argsort(w.abs(), descending=True).tolist()
        P.append('<div class="row">')
        for i in order[:a.topk]:
            v = V[:, i].real
            if float(v.norm()) < 1e-8:
                continue
            P.append(board_svg(readout(v / v.norm()),
                               f"eig  |λ|={float(w.abs()[i]):.2f}"))
        P.append("</div>")

    a.out.write_text("".join(P), encoding="utf-8")
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
