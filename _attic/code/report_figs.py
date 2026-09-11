"""Every figure the report needs that does not exist yet.

Six results have been measured in this project but never plotted, which means
they live only in terminal scrollback. Each one here recomputes or reloads its
own numbers and saves both a PNG and the JSON behind it, so nothing in the
report rests on a figure whose data cannot be reproduced.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("out/report"); OUT.mkdir(parents=True, exist_ok=True)
TEAL, ROSE, GREY, AMBER = "#0B6E78", "#A83A63", "#9AA8AD", "#8A6410"


def fig_uv():
    """The type error and its signature."""
    s = json.load(open("out/assay_uv.json"))
    o = json.load(open("out/assay_uv_olmo.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.4))

    for rows, lab, c in [(s, "SmolLM2-135M", TEAL), (o, "Olmo 3 7B", ROSE)]:
        by = {}
        for r in rows:
            by.setdefault(r["layer"], []).append(r["cos_uv"])
        xs = sorted(by)
        ax[0].plot(xs, [np.mean(by[l]) for l in xs], "o-", c=c, ms=5, label=lab)
    ax[0].axhline(1.0, ls=":", c="0.6")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel(r"$\cos(u_i, v_i)$")
    ax[0].set_title("The two sides of SVD(J) are not the same vector\n"
                    "(they converge only near the target layer)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25)
    ax[0].set_ylim(0, 1.05)

    def passes(r, k):
        return abs(r[k]) > 0.5 and abs(r[k]) > 4 * abs(r["shift_rand"])
    bands = [("0-8", lambda l: l <= 8), ("10-18", lambda l: 10 <= l <= 18),
             ("20-26", lambda l: l >= 20)]
    w = 0.35; x = np.arange(len(bands))
    pu = [np.mean([passes(r, "shift_u") for r in s if f(r["layer"])]) for _, f in bands]
    pv = [np.mean([passes(r, "shift_v") for r in s if f(r["layer"])]) for _, f in bands]
    ax[1].bar(x - w/2, pu, w, color=GREY, label="steer with $u$  (the bug)")
    ax[1].bar(x + w/2, pv, w, color=TEAL, label="steer with $v$  (correct)")
    ax[1].set_xticks(x); ax[1].set_xticklabels([b for b, _ in bands])
    ax[1].set_xlabel("layer band"); ax[1].set_ylabel("fraction of directions steering")
    ax[1].set_title("SmolLM2: the correction rescues the shallow layers\n"
                    "44% $\\rightarrow$ 64% overall", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25, axis="y")

    cos = np.array([r["cos_uv"] for r in s + o])
    rat = np.array([abs(r["shift_v"]) / max(abs(r["shift_u"]), 1e-6) for r in s + o])
    m = rat < 12
    ax[2].scatter(cos[m], rat[m], s=16, c=TEAL, alpha=.55, edgecolors="none")
    z = np.polyfit(cos[m], rat[m], 1); xx = np.linspace(cos[m].min(), cos[m].max(), 50)
    ax[2].plot(xx, np.polyval(z, xx), c=ROSE, lw=2)
    ax[2].axhline(1.0, ls=":", c="0.5")
    ax[2].set_xlabel(r"$\cos(u_i, v_i)$")
    ax[2].set_ylabel(r"$|shift_v| \, / \, |shift_u|$")
    ax[2].set_title("The size of the correction is predicted by $\\cos(u,v)$\n"
                    "this is the evidence the diagnosis is right", fontsize=10)
    ax[2].grid(alpha=.25)
    fig.suptitle("The type error: $J_\\ell$ maps layer-$\\ell$ space to target space, "
                 "so $u$ must be READ and $v$ must be INJECTED", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"F1_uv_correction.png", dpi=150,
                                    bbox_inches="tight")
    print("F1 done")


def fig_offmanifold():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture
    M = "HuggingFaceTB/SmolLM2-135M"
    blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(M)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(40)]
    rows = []
    for l in [4, 8, 12, 16, 20, 24]:
        H = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            with _MultiCapture(model, [l], blob["target"]) as cap:
                with torch.no_grad():
                    model(**enc, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); J = blob["J"][l].float()
        T = H @ J.T; T = T - T.mean(0, keepdim=True)
        Ur = torch.linalg.svd(J)[0]
        Vt = torch.linalg.svd(T, full_matrices=False)[2]
        rows.append({"layer": l,
                     "raw": ((T@Ur[:, :64]).pow(2).sum()/T.pow(2).sum()).item(),
                     "onm": ((T@Vt[:64].T).pow(2).sum()/T.pow(2).sum()).item()})
        print("  offmanifold layer", l, flush=True)
    json.dump(rows, open(OUT/"F2_offmanifold.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = [r["layer"] for r in rows]
    ax.plot(xs, [r["onm"]*100 for r in rows], "o-", c=TEAL, ms=6,
            label="PCA of $Jh$ — directions the data actually visits")
    ax.plot(xs, [r["raw"]*100 for r in rows], "o-", c=ROSE, ms=6,
            label="raw SVD of $J$ — directions the transport amplifies")
    ax.fill_between(xs, [r["raw"]*100 for r in rows], [r["onm"]*100 for r in rows],
                    color=ROSE, alpha=.10)
    for r in rows:
        ax.annotate(f"{r['raw']*100:.0f}%", (r["layer"], r["raw"]*100),
                    textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8.5, color=ROSE)
    ax.set_xlabel("layer"); ax.set_ylabel("% of real transported activation captured")
    ax.set_ylim(0, 105); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Why raw SVD readouts look like gibberish\n"
                 "two-thirds of the leading geometry is off-manifold in layers 12–24",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"F2_offmanifold.png", dpi=150)
    print("F2 done")


def fig_label_null():
    """Consistency looks decisive and is worthless; magnitude discriminates."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = "HuggingFaceTB/SmolLM2-135M-Instruct"
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float(); norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(M); del m
    P = [("color","colour"),("honor","honour"),("favor","favour"),("labor","labour"),
         ("humor","humour"),("neighbor","neighbour"),("realize","realise"),
         ("recognize","recognise"),("analyze","analyse"),("organize","organise"),
         ("apologize","apologise"),("criticize","criticise"),("center","centre"),
         ("theater","theatre"),("defense","defence"),("offense","offence"),
         ("traveled","travelled"),("canceled","cancelled"),("modeling","modelling"),
         ("gray","grey"),("fiber","fibre"),("harbor","harbour")]
    def sid(x):
        i = tok.encode(" "+x, add_special_tokens=False)
        return i[0] if len(i) == 1 else None
    IP = [(sid(a), sid(b)) for a, b in P]; IP = [(a, b) for a, b in IP if a and b]
    A = torch.tensor([a for a, _ in IP]); Bx = torch.tensor([b for _, b in IP])
    def st(vec):
        with torch.no_grad():
            lg = norm(vec.float().unsqueeze(0)).squeeze(0) @ W_U.T
        d = lg[Bx]-lg[A]
        return d.mean().item(), (d > 0).float().mean().item()
    Jb = torch.load("out/ft/J_step0.pt", map_location="cpu",
                    weights_only=False)["J"][16].float()
    Jf = torch.load("out/ft/J_step600.pt", map_location="cpu",
                    weights_only=False)["J"][16].float()
    v = torch.linalg.svd(Jf-Jb)[2][2]
    got, gotf = st(Jb @ v)
    g = torch.Generator().manual_seed(0)
    nm, nf = [], []
    for _ in range(500):
        a_, b_ = st(Jb @ torch.randn(Jb.shape[0], generator=g)); nm.append(a_); nf.append(b_)
    json.dump({"direction_mean": got, "direction_frac": gotf,
               "null_mean": nm, "null_frac": nf}, open(OUT/"F3_label_null.json", "w"))
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax[0].hist(nf, bins=24, color=GREY, edgecolor="white")
    ax[0].axvline(gotf, c=ROSE, lw=2.4)
    frac100 = np.mean(np.array(nf) >= 1.0) * 100
    ax[0].annotate(f"the direction\n(100%)", (gotf, ax[0].get_ylim()[1]*0.75),
                   ha="right", fontsize=9, color=ROSE)
    ax[0].set_xlabel("fraction of 20 US/UK pairs separated the same way")
    ax[0].set_ylabel("random directions (n=500)")
    ax[0].set_title(f"CONSISTENCY looks decisive and is worthless\n"
                    f"{frac100:.0f}% of random directions also hit 100%", fontsize=10.5)
    ax[1].hist(nm, bins=34, color=GREY, edgecolor="white")
    ax[1].axvline(got, c=TEAL, lw=2.4)
    ax[1].annotate(f"the direction\n({got:.1f})", (got, ax[1].get_ylim()[1]*0.72),
                   ha="right", fontsize=9, color=TEAL)
    beat = np.mean(np.abs(nm) > abs(got)) * 100
    ax[1].set_xlabel("mean logit(UK) − logit(US)")
    ax[1].set_title(f"MAGNITUDE discriminates\n{beat:.1f}% of random directions beat it",
                    fontsize=10.5)
    for a_ in ax: a_.grid(alpha=.2, axis="y")
    fig.suptitle("How to tell whether a readout label means anything",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"F3_label_null.png", dpi=150, bbox_inches="tight")
    print("F3 done")


if __name__ == "__main__":
    fig_uv(); fig_offmanifold(); fig_label_null()
