"""Precompute everything the interactive explorer needs.

Three things per (model, checkpoint, layer):
  * the singular spectrum of J -- how concentrated the transport is
  * the top directions, read as tokens through W_U (this is the u side, and for
    plain J it is identical to reading J*v, since J v_i = sigma_i u_i exactly)
  * for a pair of checkpoints, the SVD of dJ read BOTH ways -- u for what the
    change emits, J_base @ v for what it responds to

The both-ways part is the reason a diff view is worth building separately rather
than diffing two spectra: dJ's directions are not derivable from the two J's
singular bases, and its input side needs the base transport to be readable at
all.

Directions are stored with a random-null z-score attached, because a top-k token
list on its own has been shown here to be uninformative -- ~29% of random
directions pass a sign-consistency test that looks decisive.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

OUT = Path("out/explorer"); OUT.mkdir(parents=True, exist_ok=True)


def olmo_head():
    h = torch.load("out/readout_head.pt", map_location="cpu", weights_only=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    W_U = h["W_U"].float(); w = h["norm_state"]["weight"].float(); eps = h["norm_eps"]
    def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    return W_U, rms, tok


def hf_head(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(name)
    del m
    def f(x):
        with torch.no_grad():
            return norm(x.float().unsqueeze(0)).squeeze(0)
    return W_U, f, tok


CONFIGS = {
 "olmo": dict(head=olmo_head, layers=[0, 8, 12, 16, 20, 24],
   ckpts=[("stage1-step0","init"),("stage1-step2000","2k"),
          ("stage1-step8000","8k"),("stage1-step32000","32k"),
          ("stage1-step128000","128k"),("stage1-step512000","512k"),
          ("stage1-step1413814","end pretrain"),("stage2-step8000","mid 8k"),
          ("stage2-step47684","end midtrain"),("stage3-step5000","longctx 5k"),
          ("main","final")],
   path=lambda r: f"out/ckpt/J_{r}.pt"),
 "smollm2_ft": dict(head=lambda: hf_head("HuggingFaceTB/SmolLM2-135M-Instruct"),
   layers=[4, 8, 12, 16, 20, 24],
   ckpts=[(f"step{s}", f"step {s}") for s in
          [0,50,100,150,200,250,300,350,400,450,500,550,600]],
   path=lambda r: f"out/ft/J_{r}.pt"),
 "qwen_em": dict(head=lambda: hf_head("Qwen/Qwen2.5-0.5B-Instruct"),
   layers=[4, 8, 12, 16, 20],
   ckpts=[("base","base"),("medical","bad medical"),("financial","risky financial"),
          ("sports","extreme sports"),("control","benign control")],
   path=lambda r: f"out/em05_emprompts/J_{r}.pt"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--ndirs", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nspec", type=int, default=40)
    ap.add_argument("--nnull", type=int, default=120)
    ap.add_argument("--pairs", choices=["consecutive", "all"], default="all",
                    help="'all' lets the UI diff any two checkpoints; on a 4096-dim "
                         "model that is 55 SVDs per layer, so 'consecutive' exists "
                         "as the cheap fallback")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    a = ap.parse_args()
    C = dict(CONFIGS[a.model])
    if a.layers: C["layers"] = a.layers
    W_U, norm, tok = C["head"]()

    def logits(v):
        with torch.no_grad():
            return norm(v.float()) @ W_U.T

    def top(v, k):
        lg = logits(v)
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()], lg

    Js, order = {}, []
    for rev, label in C["ckpts"]:
        p = Path(C["path"](rev))
        if not p.exists():
            print(f"  [miss] {p}"); continue
        Js[rev] = torch.load(p, map_location="cpu", weights_only=False)["J"]
        order.append({"id": rev, "label": label})
        print(f"  [load] {rev}", flush=True)

    base_rev = order[0]["id"]
    out = {"model": a.model, "layers": C["layers"], "checkpoints": order,
           "single": {}, "diff": {}}

    for l in C["layers"]:
        Jb = Js[base_rev][l].float()
        g = torch.Generator().manual_seed(0)
        null = torch.stack([logits(Jb @ torch.randn(Jb.shape[0], generator=g))
                            for _ in range(a.nnull)])
        nm, ns = null.mean(0), null.std(0) + 1e-6

        for rev in Js:
            J = Js[rev][l].float()
            U, S, Vh = torch.linalg.svd(J)
            share = (S.pow(2) / S.pow(2).sum())
            dirs = []
            for i in range(a.ndirs):
                pos, lg = top(U[:, i], a.k)
                neg, _ = top(-U[:, i], a.k)
                z = ((lg - nm) / ns).topk(a.k).values.mean().item()
                dirs.append({"i": i, "sigma": S[i].item(),
                             "share": share[i].item(), "z": round(z, 2),
                             "pos": pos, "neg": neg})
            I = torch.eye(J.shape[0])
            out["single"][f"{rev}|{l}"] = {
                "spectrum": [round(x, 4) for x in S[:a.nspec].tolist()],
                "eff_rank": float((S.sum()**2 / S.pow(2).sum()).item()),
                "diag": float((J.diagonal().mean()).item()),
                "dev": float(((J - I).norm() / J.norm()).item()),
                "dirs": dirs}
            print(f"  [J] {rev} L{l}", flush=True)

        revs = [o["id"] for o in order]
        if a.pairs == "all":
            pairs = [(x, y) for i, x in enumerate(revs) for y in revs[i+1:]]
        else:
            pairs = list(zip(revs, revs[1:]))
        for x, y in pairs:
            dJ = Js[y][l].float() - Js[x][l].float()
            if dJ.norm() < 1e-8:
                continue
            U, S, Vh = torch.linalg.svd(dJ)
            share = S.pow(2) / S.pow(2).sum()
            dirs = []
            for i in range(a.ndirs):
                sends, _ = top(U[:, i], a.k)
                resp, lg = top(Jb @ Vh[i], a.k)
                z = ((lg - nm) / ns).topk(a.k).values.mean().item()
                dirs.append({"i": i, "sigma": S[i].item(),
                             "share": share[i].item(), "z": round(z, 2),
                             "sends": sends, "responds": resp})
            out["diff"][f"{x}>{y}|{l}"] = {
                "rel": float((dJ.norm() / (Js[x][l].float() -
                              torch.eye(dJ.shape[0])).norm()).item()),
                "spectrum": [round(v, 4) for v in S[:a.nspec].tolist()],
                "dirs": dirs}
            print(f"  [dJ] {x}>{y} L{l}", flush=True)

    f = OUT / f"{a.model}.json"
    f.write_text(json.dumps(out))
    print(f"wrote {f}  ({f.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
