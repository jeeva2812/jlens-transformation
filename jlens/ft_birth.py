"""Birth-time analysis for the fine-tuned checkpoints.

Identical measurement to the Olmo version, so the two are directly comparable.
The reference is now the FINE-TUNED model rather than the end of training, and
step0 is the un-fine-tuned starting point.

The question inherited from Olmo: post-training APPENDED low-ranked directions
without disturbing the dominant subspace. Does narrow fine-tuning do the same,
or does it rewrite the top? The second would be the interesting answer, and is
what the emergent-misalignment literature implies if a rank-1 change installs a
persona.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ft"))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()

    fs = sorted(a.dir.glob("J_step*.pt"),
                key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    revs = [re.search(r"step\d+", f.name).group(0) for f in fs]
    print(f"{len(fs)} checkpoints: {revs}")

    tok = AutoTokenizer.from_pretrained(a.dir / "step0")
    m = AutoModelForCausalLM.from_pretrained(a.dir / "step0", dtype=torch.float32)
    inner = getattr(m, "model", m)
    norm = getattr(inner, "norm", None)
    W_U = m.get_output_embeddings().weight.detach()

    V, U = {}, {}
    for f, r in zip(fs, revs):
        J = torch.load(f, map_location="cpu", weights_only=False)["J"][a.layer].float()
        Um, _, Vh = torch.linalg.svd(J)
        V[r], U[r] = Vh[:a.k].T, Um[:, :a.k]
    d_model = V[revs[0]].shape[0]
    chance = (a.k / d_model) ** .5

    fin = revs[-1]
    cap = {r: (V[r] @ V[r].T @ V[fin]).norm(dim=0).tolist() for r in revs}

    def read(v, k=4):
        with torch.no_grad():
            p = torch.softmax(norm(v) @ W_U.T, -1)
        return [tok.decode(i) for i in p.topk(k).indices]

    print(f"\nchance {chance:.3f}   layer {a.layer}\n")
    print(f"{'dir':>4}{'captured at step0':>19}{'born':>10}   reads as (final)")
    births = []
    for i in range(a.k):
        j = next((n for n, r in enumerate(revs) if cap[r][i] > .5), len(revs)-1)
        births.append(j)
        if i < 10 or 28 <= i < 36:
            print(f"{i:>4}{cap['step0'][i]:>19.2f}{revs[j]:>10}   "
                  f"{', '.join(repr(t) for t in read(U[fin][:, i], 3))}")

    n_early = sum(1 for i in range(a.k) if cap["step0"][i] > .5)
    print(f"\n  {n_early}/{a.k} of the fine-tuned model's directions were ALREADY")
    print(f"  present (>50%) before any fine-tuning at all.")
    top8 = sum(1 for i in range(8) if cap["step0"][i] > .5)
    tail = sum(1 for i in range(a.k-16, a.k) if cap["step0"][i] > .5)
    print(f"    top 8 directions:   {top8}/8 already present")
    print(f"    bottom 16:          {tail}/16 already present")
    print("\n  If the top is preserved and the tail is new, fine-tuning APPENDS,")
    print("  like post-training did on Olmo. If the top moved, it REWRITES.")

    json.dump({"captured": cap, "births": births, "revs": revs, "chance": chance,
               "layer": a.layer, "k": a.k,
               "readouts": {str(i): read(U[fin][:, i], 4) for i in range(a.k)}},
              open(a.dir / "birth_ft.json", "w"), indent=2)

    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = np.array([[cap[r][i] for r in revs] for i in range(a.k)])
    fig, ax = plt.subplots(figsize=(9.4, 6))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=chance, vmax=1.0)
    ax.set_xticks(range(len(revs)))
    ax.set_xticklabels([r.replace("step","") for r in revs], fontsize=8.5)
    ax.set_xlabel("fine-tuning step"); ax.set_ylabel("direction index in the fine-tuned model")
    ax.set_title(f"Which directions did fine-tuning create?\n"
                 f"SmolLM2-135M on insecure code, layer {a.layer}", fontsize=11)
    fig.colorbar(im, ax=ax, label=f"fraction captured (chance {chance:.2f})", fraction=.035)
    fig.tight_layout(); fig.savefig("out/viz_G_ft_birth.png", dpi=155)
    print("\nwrote out/viz_G_ft_birth.png")


if __name__ == "__main__":
    main()
