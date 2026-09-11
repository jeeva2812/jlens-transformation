"""SVD of J: what does the transport amplify, and do layers share a subspace?

Two questions, both answerable from Jacobians already on disk.

1. TOP SINGULAR DIRECTIONS. J = U S V^T. The columns of V are input directions
   in layer-l space ordered by how strongly J amplifies them; U columns are
   where they land. Reading U through W_U asks what the transport prioritises,
   without anyone picking tokens in advance -- an unsupervised readout.

2. SHARED SUBSPACES. Do different layers' Jacobians read from the same input
   directions? Principal angles between the top-k right singular subspaces of
   J_a and J_b answer that. High overlap means the layers are transporting the
   same content and depth is partly redundant; low overlap means each layer's
   transport is reading something its neighbours are not.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 28])
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js = blob["J"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    svd = {}
    for l in a.layers:
        U, S, Vh = torch.linalg.svd(Js[l].float())
        svd[l] = (U, S, Vh)
        print(f"[svd] layer {l:>2}  s1={S[0]:.2f}  s1/s2={S[0]/S[1]:.2f}  "
              f"top-10 share {float(S[:10].sum()/S.sum()):.3f}")

    print("\n\nWHAT DOES THE TRANSPORT AMPLIFY MOST?")
    print("Top singular directions read through W_U -- nobody chose these tokens.\n")
    for l in a.layers:
        U, S, Vh = svd[l]
        print(f"  layer {l}")
        for i in range(4):
            for sign, lab in ((1, "+"), (-1, "-")):
                v = sign * U[:, i]
                with torch.no_grad():
                    p = torch.softmax((norm(v) @ W_U.T).float(), -1)
                val, idx = p.topk(5)
                toks = ", ".join(f"{tok.decode(j)!r}" for j in idx)
                print(f"    sv{i} {lab} (s={S[i]:6.2f})  {toks}")
        print()

    print("\nDO LAYERS SHARE AN INPUT SUBSPACE?")
    print(f"Mean cos of principal angles between top-{a.k} right singular subspaces.")
    print("1.0 = identical subspace, 0.0 = orthogonal.\n")
    print("        " + "".join(f"{l:>9}" for l in a.layers))
    for la in a.layers:
        row = f"  L{la:<5}"
        Va = svd[la][2][:a.k].T                       # (d, k)
        for lb in a.layers:
            Vb = svd[lb][2][:a.k].T
            s = torch.linalg.svdvals(Va.T @ Vb)       # cosines of principal angles
            row += f"{float(s.mean()):>9.3f}"
        print(row)

    print("\n  Random k-dim subspaces in 4096 dims would give ~"
          f"{(a.k/4096)**0.5:.3f}")


if __name__ == "__main__":
    main()
