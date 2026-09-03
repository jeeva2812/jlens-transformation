"""Visualise the EM go/no-go: can the lens read the misalignment direction?"""
from __future__ import annotations
import argparse, statistics
from pathlib import Path
import torch

def prob_of(top, words):
    """Total probability the readout puts on a set of word forms."""
    return sum(p for w, p in top if w.strip().lower() in words)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=Path("out/em_readout.pt"))
    ap.add_argument("--out", type=Path, default=Path("out/em.png"))
    a = ap.parse_args()
    d = torch.load(a.file, map_location="cpu", weights_only=False)
    L = 24
    rent = [r["entropy"] for r in d["random"][L]]
    rm, rs = statistics.mean(rent), statistics.stdev(rent)
    conds = list(d["conds"].keys())
    YOU = {"you", "你应该", "你需要"}
    SYS = {"system", "systems"}

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GEN, NAR, GREY = "#0B6E78", "#A83A63", "#9AA8AD"
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # --- A: is it distinguishable from noise at all?
    ax.axhspan(rm - rs, rm + rs, color=GREY, alpha=.30, zorder=0)
    ax.axhline(rm, color=GREY, lw=1.4, zorder=1)
    ax.scatter([0]*len(rent), rent, c=GREY, s=34, zorder=2, label="random directions")
    for i, c in enumerate(conds, start=1):
        e = d["conds"][c][-1]["entropy"]
        ax.scatter([i], [e], c=GEN if c.startswith("general") else NAR, s=70, zorder=3)
    ax.set_xticks(range(len(conds) + 1))
    ax.set_xticklabels(["random"] + [c.replace("_", "\n") for c in conds],
                       fontsize=7.5, rotation=0)
    ax.set_ylabel("readout entropy  (lower = more specific)")
    ax.set_title("A · The lens is not blind to it", fontsize=10.5)
    ax.text(3.2, rm + .35, "noise band", fontsize=8, color="#5A6A70")
    ax.grid(alpha=.2, axis="y")

    # --- B: how the readout forms over fine-tuning
    for c in conds:
        col = GEN if c.startswith("general") else NAR
        ent = d["conds"][c]
        xs = [e["step"] for e in ent if e["step"] < 10**6]
        ys = [prob_of(e["top"], YOU) for e in ent if e["step"] < 10**6]
        bx.plot(xs, ys, "-o", ms=3, lw=1.7, color=col, alpha=.9)
    bx.axvline(245, ls=":", c="#20303A", lw=1.2)
    bx.text(258, .62, "flip to “you”\n~step 245", fontsize=8, color="#20303A")
    bx.set_xlabel("fine-tuning step")
    bx.set_ylabel('probability on "you" / "你应该" / "你需要"')
    bx.set_title("B · General misalignment becomes “you”", fontsize=10.5)
    bx.plot([], [], color=GEN, label="general (broadly misaligned)")
    bx.plot([], [], color=NAR, label="narrow (domain-only)")
    bx.legend(frameon=False, fontsize=8, loc="upper left")
    bx.grid(alpha=.2)

    # --- C: what each condition actually says
    ys = range(len(conds))
    you = [prob_of(d["conds"][c][-1]["top"], YOU) for c in conds]
    sysm = [prob_of(d["conds"][c][-1]["top"], SYS) for c in conds]
    cx.barh([y + .19 for y in ys], you, height=.36, color=GEN, label='"you"')
    cx.barh([y - .19 for y in ys], sysm, height=.36, color=NAR, label='"system"')
    cx.set_yticks(list(ys)); cx.set_yticklabels(conds, fontsize=8.5)
    cx.invert_yaxis()
    cx.set_xlabel("probability in the readout")
    cx.set_title("C · Two different signatures", fontsize=10.5)
    cx.legend(frameon=False, fontsize=8.5)
    cx.grid(alpha=.2, axis="x")

    for x in (ax, bx, cx):
        for s in ("top", "right"): x.spines[s].set_visible(False)
    fig.suptitle("Reading the misalignment direction through J-Lens — Qwen2.5-14B, layer 24",
                 fontsize=11.5)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)

    print("\nphase transition check, all six conditions:")
    for c in conds:
        ent = [e for e in d["conds"][c] if e["step"] < 10**6]
        first = next((e["step"] for e in ent if prob_of(e["top"], YOU) > .25), None)
        print(f"  {c:<17} first step with >25% on 'you': "
              f"{first if first else 'never'}")

if __name__ == "__main__":
    main()
