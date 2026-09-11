"""Readouts of J_l @ h_l(position) -- the position-DEPENDENT object.

The SVD directions of J are averaged over prompts, so they can only carry
square geometry. Piece identity cannot survive that average: in UCI the move
token names a square, never a piece, so "the knight moves" is only defined
relative to a board. To see anything piece-like we have to feed J a real
activation.

For each ply we compute
    predicted = W_U @ ln_f(J_l @ h_l)     linearised through the Jacobian
    actual    = W_U @ ln_f(h_target)      the model's own next-token logits
and compare them. Agreement is a direct test of how good a linear
approximation J is at that layer -- the thing J-Lens is implicitly claiming.
"""
from __future__ import annotations
import json
from pathlib import Path
import chess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "austindavis/chess-gpt2-uci-8x8x512"; SQ0 = 4
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p in m.parameters():
    p.requires_grad_(False)
_, ln = _find_blocks_and_norm(m)
W_U = m.get_output_embeddings().weight.detach().float()
blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
target = blob["target"]
layers = [l for l in blob["layers"] if l < target]
sample = json.load(open("out/chess/sample_game.json"))
ids, fens = sample["ids"], sample["fens"]

def rd(v):
    with torch.no_grad():
        return (ln(v.float().unsqueeze(0)).squeeze(0) @ W_U.T)[SQ0:SQ0 + 64]

out = []
for k in range(1, min(len(ids), len(fens))):
    t = torch.tensor([ids[:k]])
    with _MultiCapture(m, layers, target) as cap:
        with torch.no_grad():
            m(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False)
        hs = {l: cap.h[l][0, -1].detach().clone() for l in layers}
        h_t = cap.target(target)[0, -1].detach().clone()
    actual = rd(h_t)
    b = chess.Board(fens[k])
    if k % 2 == 1:
        legal = sorted({mv.from_square for mv in b.legal_moves})
    else:
        frm = ids[k - 1] - SQ0
        legal = sorted({mv.to_square for mv in b.legal_moves if mv.from_square == frm})
    pieces = {sq: pc.symbol() for sq, pc in b.piece_map().items()}
    rec = {"ply": k, "actual": [round(float(x), 3) for x in actual],
           "pred": {}, "agree": {},
           "pieces": {str(s): p for s, p in pieces.items()}}
    for l in layers:
        pr = rd(blob["J"][l].float() @ hs[l])
        rec["pred"][str(l)] = [round(float(x), 3) for x in pr]
        cos = float(torch.nn.functional.cosine_similarity(pr, actual, dim=0))
        rec["agree"][str(l)] = {"cos": round(cos, 3),
                                "top1": int(pr.argmax()) == int(actual.argmax())}
    out.append(rec)
    if k % 20 == 0:
        print("ply", k, flush=True)

Path("out/chess/act.json").write_text(json.dumps({"target": target, "layers": layers,
                                                  "plies": out}))
print("\nJ@h vs actual logits, mean cosine and top-1 agreement per layer:")
for l in layers:
    cs = [r["agree"][str(l)]["cos"] for r in out]
    t1 = [r["agree"][str(l)]["top1"] for r in out]
    print(f"  L{l} -> {target}:  cos {sum(cs)/len(cs):+.3f}   top-1 agree {sum(t1)/len(t1):.0%}")
