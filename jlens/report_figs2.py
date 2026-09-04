"""Report figures 4-6: what changed, across training and across fine-tunes."""
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

UKUS = [("color","colour"),("honor","honour"),("favor","favour"),("labor","labour"),
 ("humor","humour"),("neighbor","neighbour"),("realize","realise"),
 ("recognize","recognise"),("analyze","analyse"),("organize","organise"),
 ("apologize","apologise"),("criticize","criticise"),("center","centre"),
 ("theater","theatre"),("defense","defence"),("offense","offence"),
 ("traveled","travelled"),("canceled","cancelled"),("modeling","modelling"),
 ("gray","grey"),("fiber","fibre"),("harbor","harbour")]


def fig_olmo_phases():
    d = json.load(open("out/olmo_delta_svd.json"))
    labels = list(d)
    rel = [d[p]["8"]["rel"] for p in labels]
    themes = ["British orthography\n+ quote punctuation",
              "Romance languages\n+ formal register",
              "Romance languages\n+ modern professional",
              "web/social era\ninformal, contemporary",
              "document-level punctuation\nAmerican orthography"]
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    x = np.arange(len(labels))
    cols = [GREY, GREY, GREY, TEAL, GREY]
    b = ax.bar(x, rel, 0.6, color=cols, edgecolor="white")
    for i, (r, t) in enumerate(zip(rel, themes)):
        ax.annotate(f"{r:.3f}", (i, r), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=10, weight="bold")
        ax.annotate(t, (i, r/2), ha="center", va="center", fontsize=8.5,
                    color="white" if i == 3 else "#33403F")
    ax.set_xticks(x)
    ax.set_xticklabels(["first 2k\nsteps", "early pretrain\n8k→128k",
                        "late pretrain\n512k→end", "MID-TRAINING\n(post-train)",
                        "long-context\nextension"], fontsize=9)
    ax.set_ylabel(r"$\|\Delta J\| \, / \, \|J - I\|$   at layer 8")
    ax.grid(alpha=.25, axis="y")
    ax.set_title("Each phase of training changes the transport less than the one before\n"
                 "labels are what the leading INPUT-side directions respond to",
                 fontsize=11.5)
    fig.tight_layout(); fig.savefig(OUT/"F4_olmo_phases.png", dpi=150)
    print("F4 done")


def fig_ft_traj():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    M = "HuggingFaceTB/SmolLM2-135M-Instruct"
    m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float(); norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(M); del m
    def sid(x):
        i = tok.encode(" "+x, add_special_tokens=False)
        return i[0] if len(i) == 1 else None
    IP = [(sid(a), sid(b)) for a, b in UKUS]; IP = [(a, b) for a, b in IP if a and b]
    def sc(vec):
        with torch.no_grad():
            lg = norm(vec.float().unsqueeze(0)).squeeze(0) @ W_U.T
        return torch.tensor([(lg[b]-lg[a]).item() for a, b in IP]).mean().item()
    L = 16; res = {}
    steps = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600]
    best_dir = {}
    for run, lab in [("out/ft", "lr 5e-5"), ("out/ft_lr1e-5", "lr 1e-5")]:
        Jb = torch.load(f"{run}/J_step0.pt", map_location="cpu",
                        weights_only=False)["J"][L].float()
        g = torch.Generator().manual_seed(0)
        null = torch.tensor([abs(sc(Jb @ torch.randn(Jb.shape[0], generator=g)))
                             for _ in range(300)])
        thr = null.quantile(0.99).item()
        xs, ys, rels = [], [], []
        for st in steps:
            f = Path(run)/f"J_step{st}.pt"
            if not f.exists(): continue
            dJ = torch.load(f, map_location="cpu",
                            weights_only=False)["J"][L].float() - Jb
            if dJ.norm() < 1e-6:
                xs.append(st); ys.append(0.0)
                rels.append(0.0); continue
            U, S, Vh = torch.linalg.svd(dJ)
            vals = [abs(sc(Jb @ Vh[i])) for i in range(6)]
            i = int(np.argmax(vals))
            xs.append(st); ys.append(vals[i])
            rels.append((dJ.norm()/(Jb-torch.eye(Jb.shape[0])).norm()).item())
            if st == 600: best_dir[lab] = Vh[i].clone()
        res[lab] = {"steps": xs, "score": ys, "rel": rels, "thr": thr}
        print(f"  {lab} done", flush=True)
    cosab = torch.nn.functional.cosine_similarity(
        best_dir["lr 5e-5"], best_dir["lr 1e-5"], dim=0).item()
    res["cos_between_runs"] = cosab
    json.dump(res, open(OUT/"F5_ft_traj.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.4))
    for lab, c in [("lr 5e-5", TEAL), ("lr 1e-5", ROSE)]:
        r = res[lab]
        ax[0].plot(r["steps"], r["score"], "o-", c=c, ms=5, label=lab)
        ax[1].plot(r["steps"], r["rel"], "o-", c=c, ms=5, label=lab)
    ax[0].axhline(res["lr 5e-5"]["thr"], ls=":", c="crimson",
                  label=f"99th pct of random ({res['lr 5e-5']['thr']:.1f})")
    ax[0].set_xlabel("fine-tuning step"); ax[0].set_ylabel("|mean logit(UK) − logit(US)|")
    ax[0].set_title("The orthography axis appears by step 50 and peaks at 150–200\n"
                    "— then weakens while the change keeps growing", fontsize=10.5)
    ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=.25)
    ax[1].set_xlabel("fine-tuning step"); ax[1].set_ylabel(r"$\|\Delta J\|/\|J-I\|$")
    ax[1].set_title(f"The change keeps growing in both runs\n"
                    f"but their leading directions are ORTHOGONAL "
                    f"(cos = {cosab:+.3f})", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25)
    fig.suptitle("Same model, same data, same steps — only the learning rate differs",
                 fontsize=12, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"F5_ft_traj.png", dpi=150, bbox_inches="tight")
    print("F5 done")


def fig_qwen_domain():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    B = "Qwen/Qwen2.5-0.5B-Instruct"
    m = AutoModelForCausalLM.from_pretrained(B, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float(); norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(B); del m
    VOC = {"financial": ["investment","investors","liquidity","bonds","equity",
             "portfolio","dividend","assets","capital","revenue","stocks","loan",
             "credit","debt"],
           "medical": ["patient","symptoms","diagnosis","treatment","dose","clinical",
             "therapy","surgery","infection","chronic","prescription","tumor"],
           "sports": ["athlete","training","climbing","fitness","muscle","endurance",
             "workout","running","injury","competition","sprint","marathon"]}
    NEU = ["window","garden","pencil","mountain","copper","ceiling","rabbit",
           "guitar","butter","tunnel","curtain","pebble"]
    def ids(ws):
        o = []
        for w in ws:
            i = tok.encode(" "+w, add_special_tokens=False)
            if len(i) == 1: o.append(i[0])
        return torch.tensor(o)
    IV = {k: ids(v) for k, v in VOC.items()}; IN = ids(NEU)
    def score(vec, k):
        with torch.no_grad():
            lg = norm(vec.float().unsqueeze(0)).squeeze(0) @ W_U.T
        return (lg[IV[k]].mean()-lg[IN].mean()).item()
    D, L = "out/em05_emprompts", 12
    Jb = torch.load(f"{D}/J_base.pt", map_location="cpu",
                    weights_only=False)["J"][L].float()
    g = torch.Generator().manual_seed(0)
    NULL = {k: torch.tensor([score(torch.randn(Jb.shape[0], generator=g), k)
                             for _ in range(500)]) for k in VOC}
    names = ["medical", "financial", "sports", "control"]
    Mx = np.zeros((4, 3)); thr = [NULL[k].quantile(0.99).item() for k in VOC]
    for i, n in enumerate(names):
        dJ = torch.load(f"{D}/J_{n}.pt", map_location="cpu",
                        weights_only=False)["J"][L].float() - Jb
        U = torch.linalg.svd(dJ)[0]
        for j, k in enumerate(VOC):
            Mx[i, j] = max(score(U[:, q], k) for q in range(6))
        print(f"  {n} done", flush=True)
    json.dump({"matrix": Mx.tolist(), "thresholds": thr, "rows": names,
               "cols": list(VOC)}, open(OUT/"F6_qwen_domain.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    rel = Mx / np.array(thr)[None, :]
    im = ax.imshow(rel, cmap="RdBu_r", vmin=0, vmax=3.2)
    for i in range(4):
        for j in range(3):
            hit = Mx[i, j] > thr[j]
            ax.text(j, i, f"{Mx[i,j]:.2f}" + ("\n✓" if hit else ""),
                    ha="center", va="center", fontsize=11,
                    weight="bold" if hit else "normal",
                    color="white" if rel[i, j] > 2.2 else "#22303A")
    ax.set_xticks(range(3)); ax.set_xticklabels(
        [f"{c}\n(thr {t:.2f})" for c, t in zip(VOC, thr)], fontsize=9.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["medical", "financial", "sports", "control\n(medical)"],
                       fontsize=9.5)
    ax.set_xlabel("vocabulary scored"); ax.set_ylabel("fine-tune")
    ax.set_title("Does $\\Delta J$ SVD recover the fine-tuning domain?\n"
                 "no false positives — but silent for half the cases", fontsize=11.5)
    fig.colorbar(im, ax=ax, label="score / 99th-pct random threshold", shrink=.8)
    fig.tight_layout(); fig.savefig(OUT/"F6_qwen_domain.png", dpi=150)
    print("F6 done")


if __name__ == "__main__":
    fig_olmo_phases(); fig_ft_traj(); fig_qwen_domain()
