"""Idea 1: is the parity artifact an artifact of AVERAGING?

J = E_positions[dh_target/dh_l] pools positions where the model is about to emit
a move's origin square with positions where it is about to emit a destination.
If those two have systematically different Jacobians, then the top directions of
the pooled J encode the MIXTURE AXIS rather than the computation -- which would
explain the parity artifact completely, and the fix would be to condition.

Same seeding trick as jacobians_all_layers, so conditioning costs 2x, not 512x:
the per-anchor validity mask is simply restricted by parity. Source position p
predicts token p+1, so p even -> an origin square, p odd -> a destination.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.chess_j import sample_game

MODEL = "austindavis/chess-gpt2-uci-8x8x512"

def jac_parity(model, batches, layers, target_layer, parity, skip_first=4, chunk=128):
    d = model.get_output_embeddings().weight.shape[1]
    total = {l: torch.zeros(d, d) for l in layers}; n = 0
    for input_ids, attention_mask in batches:
        with _MultiCapture(model, layers, target_layer) as cap:
            with torch.enable_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hs = [cap.h[l] for l in layers]; h_t = cap.target(target_layer)
        B, T, _ = h_t.shape
        idx = torch.arange(T)
        lengths = attention_mask.sum(dim=1, keepdim=True)
        valid = ((idx.unsqueeze(0) >= skip_first) & (idx.unsqueeze(0) < lengths - 1)
                 & attention_mask.bool())
        if parity is not None:
            valid = valid & (idx.unsqueeze(0) % 2 == parity)
        if valid.sum() == 0:
            continue
        counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        eye = torch.eye(d, dtype=h_t.dtype)
        for s in range(0, d, chunk):
            blk = eye[s:s + chunk]; C = blk.shape[0]
            go = blk.view(C, 1, 1, d).expand(C, B, T, d)
            grads = torch.autograd.grad(h_t, hs, grad_outputs=go, retain_graph=True,
                                        is_grads_batched=True, allow_unused=True)
            for l, gr in zip(layers, grads):
                if gr is None: continue
                gr = gr * valid.unsqueeze(0).unsqueeze(-1).to(gr.dtype)
                total[l][s:s + C] += (gr.sum(dim=2) / counts.unsqueeze(0)).sum(dim=1).float()
        n += B
        del hs, h_t
    return {l: total[l] / max(n, 1) for l in layers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    a = ap.parse_args()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, _ = _find_blocks_and_norm(m)
    target = len(blocks) - 1; layers = list(range(target))
    g = torch.Generator().manual_seed(404)
    games = [sample_game(m, tok, max_plies=30, gen=g)[0] for _ in range(a.games)]
    print(f"{len(games)} games sampled", flush=True)
    def batches():
        for ids in games:
            t = torch.tensor([ids]); yield t, torch.ones_like(t)
    out = {}
    for name, par in (("origin", 0), ("dest", 1), ("pooled", None)):
        out[name] = {l: v for l, v in jac_parity(m, batches(), layers, target, par).items()}
        print(f"[{name}] done", flush=True)
    torch.save({"target": target, "layers": layers,
                "J": {k: {l: v.to(torch.float16) for l, v in d.items()} for k, d in out.items()}},
               "out/chess/J_cond.pt")
    print(f"\n{'layer':>5} {'||Jo-Jd||/||Jp||':>17} {'cos(Jo,Jd)':>11} "
          f"{'sig1 pooled':>12} {'sig1 origin':>12}")
    for l in layers:
        Jo, Jd, Jp = out["origin"][l], out["dest"][l], out["pooled"][l]
        rel = float((Jo - Jd).norm() / Jp.norm())
        cs = float((Jo.flatten() @ Jd.flatten()) / (Jo.norm() * Jd.norm()))
        so = float(torch.linalg.svdvals(Jo)[0]); sp = float(torch.linalg.svdvals(Jp)[0])
        print(f"{l:>5} {rel:>17.3f} {cs:>11.3f} {sp:>12.2f} {so:>12.2f}")

if __name__ == "__main__":
    main()
