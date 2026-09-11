"""Export the whole router tensor as an atlas: every (layer, expert) direction
with its own unembedded tokens FIRST and my concept tag second, so the tags are
annotation rather than the frame.

Also exports the geometry per layer -- how much the 64 directions at a layer
overlap each other -- since "64 x d x L subspaces" is really a question about
whether those directions span the space or crowd into a corner.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; L = len(blk); E = m.config.num_experts
    mu = W_U.mean(0); Wc = W_U - mu
    Sigma = (Wc.T @ Wc) / W_U.shape[0]
    cb, names = [], []
    for c, s in CONCEPTS.items():
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in s.split()]
        ids = [i[0] for i in ids if len(i) == 1]
        if len(ids) >= 5: cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)
    thr = json.load(open("out/router_dict.json"))["thr"]

    cells, layers = [], []
    for l in range(L):
        R = blk[l].mlp.gate.weight.detach().float()
        Rn = R / R.norm(dim=1, keepdim=True)
        num = Rn @ (C - mu).T
        den = ((Rn @ Sigma) * Rn).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        S = num / den                                        # (E, nc)
        Z = Rn @ W_U.T
        top = Z.topk(8, dim=1).indices
        for e in range(E):
            z, ci = S[e].max(0)
            cells.append({
                "l": l, "e": e,
                "toks": [tok.decode([i]) for i in top[e].tolist()],
                "tag": names[int(ci)] if float(z) > thr else None,
                "z": round(float(z), 2),
                "tag2": names[int(S[e].argsort(descending=True)[1])],
                "z2": round(float(S[e].sort(descending=True).values[1]), 2),
            })
        # geometry of the 64 directions at this layer
        G = Rn @ Rn.T
        off = G[~T.eye(E, dtype=bool)]
        ev = T.linalg.svdvals(Rn)
        layers.append({"l": l,
                       "mean_abs_cos": round(float(off.abs().mean()), 3),
                       "max_abs_cos": round(float(off.abs().max()), 3),
                       "eff_rank": round(float((ev.sum() ** 2) / (ev ** 2).sum()), 1),
                       "norm_spread": round(float(R.norm(dim=1).std() / R.norm(dim=1).mean()), 3)})
    Path("out/router_atlas.json").write_text(json.dumps(
        {"L": L, "E": E, "d": W_U.shape[1], "thr": thr,
         "cells": cells, "layers": layers, "concepts": names}))
    n_tag = sum(1 for c in cells if c["tag"])
    print(f"{len(cells)} directions, {n_tag} tagged ({n_tag/len(cells):.1%})")
    print(f"\n{'layer':>6} {'mean|cos|':>10} {'max|cos|':>9} {'eff rank of 64':>15} {'norm spread':>12}")
    for r in layers:
        print(f"{r['l']:>6} {r['mean_abs_cos']:>10.3f} {r['max_abs_cos']:>9.3f} "
              f"{r['eff_rank']:>15.1f} {r['norm_spread']:>12.3f}")
    print(f"\n(64 directions in {W_U.shape[1]}d: random would give mean|cos| "
          f"{(2/(3.1416*W_U.shape[1]))**.5:.3f}, eff rank ~64)")

if __name__ == "__main__":
    main()
