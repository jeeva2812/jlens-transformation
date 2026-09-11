"""Export chess J readouts + a game trace as JSON for the interactive page."""
from __future__ import annotations
import json
from pathlib import Path
import chess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "austindavis/chess-gpt2-uci-8x8x512"; SQ0 = 4; TOPK = 16

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
_, ln = _find_blocks_and_norm(m)
W_U = m.get_output_embeddings().weight.detach().float()
blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
target = blob["target"]

def readout(v):
    with torch.no_grad():
        return (ln(v.float().unsqueeze(0)).squeeze(0) @ W_U.T)[SQ0:SQ0 + 64].tolist()

dirs = {}
for L in [l for l in blob["layers"] if l < target]:
    J = blob["J"][L].float()
    U, S, Vh = torch.linalg.svd(J)
    svd = [{"i": i, "val": round(float(S[i]), 3), "board": [round(x, 4) for x in readout(U[:, i])]}
           for i in range(TOPK)]
    w, V = torch.linalg.eig(J)
    order = torch.argsort(w.abs(), descending=True).tolist()
    eig, kept = [], []
    for j in order:
        v = V[:, j].real
        n = float(v.norm())
        if n < 1e-9:
            continue
        v = v / n
        if any(abs(float(v @ u)) > 0.99 for u in kept):
            continue
        kept.append(v)
        eig.append({"i": j, "val": round(float(w.abs()[j]), 3),
                    "board": [round(x, 4) for x in readout(v)]})
        if len(eig) >= TOPK:
            break
    dirs[str(L)] = {"svd": svd, "eigen": eig,
                    "henrici": round(float(S[0] / w.abs().max()), 3)}
    print("layer", L, "done", flush=True)

# --- game trace: position, legal-origin oracle, model logits, at each ply
sample = json.load(open("out/chess/sample_game.json"))
ids, fens = sample["ids"], sample["fens"]
plies = []
for k in range(1, min(len(ids), len(fens))):
    b = chess.Board(fens[k])
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([ids[:k]])).logits[0, -1][SQ0:SQ0 + 64]
    # at even k the model emits an origin square, at odd k a destination
    if k % 2 == 1:
        legal = sorted({mv.from_square for mv in b.legal_moves})
    else:
        frm = ids[k - 1] - SQ0
        legal = sorted({mv.to_square for mv in b.legal_moves if mv.from_square == frm})
    plies.append({"ply": k, "fen": fens[k], "legal": legal,
                  "kind": "origin" if k % 2 == 1 else "destination",
                  "logits": [round(float(x), 3) for x in lg],
                  "actual": ids[k] - SQ0 if k < len(ids) else None})

Path("out/chess/viz.json").write_text(json.dumps(
    {"model": MODEL, "target": target, "dirs": dirs, "plies": plies,
     "legality_acc": blob.get("legality_acc")}))
print("wrote out/chess/viz.json", len(json.dumps({"dirs": dirs})) // 1024, "KB dirs")
