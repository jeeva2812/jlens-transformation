"""Decompose the CHANGE, not the model.

Every decomposition so far ran on J itself, which answers "what does this
transport do". For a fine-tune the sharper question is what the fine-tune ADDED,
and that object is

    dJ = J_finetuned - J_base

Its top singular directions are what training installed, with everything the
model already did subtracted away. This is the same trick that fixed the drift
metric earlier -- subtract the part that does not carry information (there, the
identity; here, the base model) -- and it should be far more sensitive than
scanning J's own spectrum, where a fine-tune's contribution is buried under
directions that were already there.

Controls, because a difference of two matrices always HAS a top direction:
  RANK PROFILE   if the change were diffuse noise, dJ would be near full-rank
                 with a flat spectrum. Concentration is the signal.
  BASE-VS-BASE   dJ between two checkpoints EARLY in the run (step0 to step50)
                 gives the shape of a small, uninteresting change for comparison.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ft"))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()

    m = AutoModelForCausalLM.from_pretrained(a.dir / "step0", dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(a.dir / "step0")
    norm = getattr(getattr(m, "model", m), "norm", None)
    W_U = m.get_output_embeddings().weight.detach()

    def read(v, k=6):
        with torch.no_grad():
            p = torch.softmax(norm(v) @ W_U.T, -1)
        return [tok.decode(i) for i in p.topk(k).indices]

    J0 = torch.load(a.dir / "J_step0.pt", map_location="cpu", weights_only=False)["J"]
    JE = torch.load(a.dir / "J_step50.pt", map_location="cpu", weights_only=False)["J"]
    JF = torch.load(a.dir / "J_step600.pt", map_location="cpu", weights_only=False)["J"]

    print("How concentrated is the change?  (fraction of dJ's energy in its top directions)\n")
    print(f"{'layer':>6}{'|dJ|/|J|':>10}{'top1':>8}{'top5':>8}{'top20':>8}"
          f"   {'| early dJ top1':>16}{'top5':>8}")
    print("-" * 72)
    for l in a.layers:
        d_full = (JF[l].float() - J0[l].float())
        d_early = (JE[l].float() - J0[l].float())
        rel = float(d_full.norm() / J0[l].float().norm())
        Sf = torch.linalg.svdvals(d_full); Sf = Sf.pow(2) / Sf.pow(2).sum()
        Se = torch.linalg.svdvals(d_early); Se = Se.pow(2) / Se.pow(2).sum()
        print(f"{l:>6}{rel:>10.3f}{float(Sf[0]):>8.3f}{float(Sf[:5].sum()):>8.3f}"
              f"{float(Sf[:20].sum()):>8.3f}   {float(Se[0]):>16.3f}{float(Se[:5].sum()):>8.3f}")

    print("\n\nWHAT DID THE FINE-TUNE INSTALL?")
    print("top singular directions of dJ, read through the unembedding\n")
    out = {}
    for l in a.layers:
        d = (JF[l].float() - J0[l].float())
        U, S, _ = torch.linalg.svd(d)
        tot = float(S.pow(2).sum())
        print(f"layer {l}")
        rows = []
        for i in range(a.k):
            pos, neg = read(U[:, i], 5), read(-U[:, i], 5)
            share = float(S[i] ** 2 / tot)
            rows.append({"i": i, "share": share, "pos": pos, "neg": neg})
            print(f"  d{i} ({share:5.1%})  + {', '.join(repr(t) for t in pos[:4])}")
            print(f"              - {', '.join(repr(t) for t in neg[:4])}")
        out[str(l)] = rows
        print()
    Path("out/delta_svd.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
