"""Three panels for the 0.5B organism result. It is a negative, so it has to be
drawn as one -- the reader should be able to see the absence, not be told it.

Panel 1  overlap by layer, one line per pair, with the measured null. The point
         is that the sports pairings sit on top of the misaligned pairing.
Panel 2  the same question at every rank from 1 to 64, because "you picked the
         wrong k" is the first objection and it deserves an answer in the figure.
Panel 3  what survives after the generic fine-tuning direction is projected out,
         with the misaligned-vs-aligned pairing drawn alongside.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .em_delta import top_subspace, overlap, measured_null

C = {"med-fin": "#A83A63", "med-spo": "#0B6E78", "fin-spo": "#4C8B2B",
     "med-con": "#8A6410", "fin-con": "#6B7D83", "spo-con": "#9AA7AB"}
LAYERS = [4, 8, 12, 16, 20]


def load(d="out/em05"):
    return {n: torch.load(f"{d}/J_{n}.pt", map_location="cpu", weights_only=False)
            for n in ["base", "medical", "financial", "sports", "control"]}


def main(d="out/em05", out=Path("out/em05/em05.png")):
    J = load(d)
    dm = J["base"]["J"][LAYERS[0]].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))

    # -- panel 1 -------------------------------------------------------------
    ax = axes[0]
    null8, _ = measured_null(dm, 8)
    series = {k: [] for k in C}
    for l in LAYERS:
        Jb = J["base"]["J"][l].float()
        S = {n: top_subspace(J[n]["J"][l].float() - Jb, 8, "left")[0]
             for n in ["medical", "financial", "sports", "control"]}
        for key, (a, b) in {"med-fin": ("medical", "financial"),
                            "med-spo": ("medical", "sports"),
                            "fin-spo": ("financial", "sports"),
                            "med-con": ("medical", "control"),
                            "fin-con": ("financial", "control"),
                            "spo-con": ("sports", "control")}.items():
            series[key].append(overlap(S[a], S[b]))
    for k, v in series.items():
        solid = "con" not in k
        ax.plot(LAYERS, v, "o-" if solid else "o--", color=C[k], label=k,
                lw=2.2 if k == "med-fin" else 1.5, ms=5,
                zorder=3 if k == "med-fin" else 2)
    ax.axhline(null8, ls=":", c="crimson", lw=1.2)
    ax.text(4.2, null8 + .012, f"chance {null8:.3f}", fontsize=8, c="crimson")
    ax.set_xlabel("layer"); ax.set_ylabel("top-8 subspace overlap of $\\Delta J$")
    ax.set_title("Overlap does not track misalignment\n"
                 "(sports is barely misaligned, yet sits on top)", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, ncol=2); ax.grid(alpha=.25)
    ax.set_xticks(LAYERS)

    # -- panel 2 -------------------------------------------------------------
    ax = axes[1]
    ks = [1, 2, 4, 8, 16, 32, 64]
    mf, sp, cn, nulls = [], [], [], []
    for k in ks:
        nm, _ = measured_null(dm, k, n=60); nulls.append(nm)
        acc = {"mf": [], "sp": [], "cn": []}
        for l in [8, 12, 16]:
            Jb = J["base"]["J"][l].float()
            S = {n: top_subspace(J[n]["J"][l].float() - Jb, k, "left")[0]
                 for n in ["medical", "financial", "sports", "control"]}
            acc["mf"].append(overlap(S["medical"], S["financial"]))
            acc["sp"].append((overlap(S["medical"], S["sports"]) +
                              overlap(S["financial"], S["sports"])) / 2)
            acc["cn"].append((overlap(S["medical"], S["control"]) +
                              overlap(S["financial"], S["control"])) / 2)
        mf.append(sum(acc["mf"]) / 3); sp.append(sum(acc["sp"]) / 3)
        cn.append(sum(acc["cn"]) / 3)
    ax.semilogx(ks, mf, "o-", c=C["med-fin"], lw=2.2, ms=5,
                label="misaligned pair (med-fin)")
    ax.semilogx(ks, sp, "o-", c=C["med-spo"], lw=1.6, ms=5,
                label="weakly-misaligned pairs (sports)")
    ax.semilogx(ks, cn, "o--", c=C["med-con"], lw=1.6, ms=5,
                label="aligned control pairs")
    ax.semilogx(ks, nulls, ":", c="crimson", lw=1.2, label="chance")
    ax.set_xlabel("k (subspace rank)"); ax.set_ylabel("overlap, mean of layers 8/12/16")
    ax.set_title("Not an artefact of choosing k=8\n"
                 "no rank separates misaligned from weakly-misaligned",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25)
    ax.set_xticks(ks); ax.set_xticklabels(ks)

    # -- panel 3 -------------------------------------------------------------
    ax = axes[2]
    raw, rmf, rmc = [], [], []
    for l in LAYERS:
        Jb = J["base"]["J"][l].float()
        dJ = {n: J[n]["J"][l].float() - Jb
              for n in ["medical", "financial", "sports", "control"]}
        P = top_subspace(dJ["sports"], 8, "left")[0]
        R = {n: top_subspace(dJ[n] - P @ (P.T @ dJ[n]), 8, "left")[0]
             for n in ["medical", "financial", "control"]}
        raw.append(overlap(top_subspace(dJ["medical"], 8, "left")[0],
                           top_subspace(dJ["financial"], 8, "left")[0]))
        rmf.append(overlap(R["medical"], R["financial"]))
        rmc.append(overlap(R["medical"], R["control"]))
    ax.plot(LAYERS, raw, "o-", c="0.62", lw=1.5, ms=5, label="med-fin, raw")
    ax.plot(LAYERS, rmf, "o-", c=C["med-fin"], lw=2.2, ms=5,
            label="med-fin, generic removed")
    ax.plot(LAYERS, rmc, "o--", c=C["med-con"], lw=1.8, ms=5,
            label="med-CONTROL, generic removed")
    ax.axhline(null8, ls=":", c="crimson", lw=1.2)
    ax.set_xlabel("layer"); ax.set_ylabel("overlap after projecting out $\\Delta J_{sports}$")
    ax.set_title("Nothing misalignment-specific underneath\n"
                 "misaligned pair does not beat the aligned pairing",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25); ax.set_xticks(LAYERS)

    fig.suptitle("Emergent misalignment is NOT isolated by the top subspace of "
                 "$\\Delta J$  ·  Qwen2.5-0.5B-Instruct, four rank-32 LoRAs",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
