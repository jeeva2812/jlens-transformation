"""What does the shared dJ direction point AT, in words?

A direction in R^896 means nothing until it is read out. The J-Lens readout is
softmax(W_U . norm(J_l . h_l)), so a direction d at layer l is pushed through
the model's own transport and unembedding to get the tokens it promotes.

TWO CONTROLS, because "the top tokens look meaningful" is the easiest possible
thing to fool yourself with:

  random    a random unit direction at the same layer, read out identically.
            If its tokens look equally thematic, the readout is telling you
            about W_U's geometry, not about d.

  base-J    the SAME direction read through the BASE model's J instead of the
            organism's. dJ is defined as a difference, so the direction is only
            meaningful relative to what the fine-tune changed; if the base
            readout gives the same words, nothing about it is fine-tune-specific.

An earlier claim in this project -- that a direction was "98% present before
fine-tuning" -- was wrong in exactly this way: the SUBSPACE was present, the
readable direction was not. Subspace presence and direction presence are
different measurements and get confused constantly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def readout(W_U, norm, J, d, k=25):
    """Top tokens promoted by pushing direction d through transport J."""
    with torch.no_grad():
        v = norm(J.float() @ d.float())
        logits = W_U @ v
        top = logits.topk(k)
    return top.indices.tolist(), top.values.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, default=Path("out/em05/shared_dirs.pt"))
    ap.add_argument("--jdir", type=Path, default=Path("out/em05"))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--k", type=int, default=22)
    a = ap.parse_args()

    rec = torch.load(a.dirs, map_location="cpu", weights_only=False)
    D = rec["dirs"][a.layer].float()
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach()
    norm = m.model.norm

    Jm = torch.load(a.jdir / "J_medical.pt", map_location="cpu",
                    weights_only=False)["J"][a.layer]
    Jb = torch.load(a.jdir / "J_base.pt", map_location="cpu",
                    weights_only=False)["J"][a.layer]

    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(D.shape[1], generator=g)
    rnd = rnd / rnd.norm()

    def show(label, J, d):
        ids, _ = readout(W_U, norm, J, d, a.k)
        words = [repr(tok.decode([i])) for i in ids]
        print(f"  {label:<26} {' '.join(words)}")

    print(f"layer {a.layer}   principal cosines: "
          f"{[round(c,3) for c in rec['cosines'][a.layer][:a.n_dirs]]}\n")
    for i in range(min(a.n_dirs, D.shape[0])):
        print(f"--- shared direction {i}  (principal cosine "
              f"{rec['cosines'][a.layer][i]:.3f}) ---")
        show("through organism J", Jm, D[i])
        show("through BASE J (control)", Jb, D[i])
        print()
    print("--- random direction (control) ---")
    show("through organism J", Jm, rnd)


if __name__ == "__main__":
    main()
