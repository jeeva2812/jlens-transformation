"""Does J Sigma^(1/2) beat plain J at carrying abstract structure?

J's own SVD ranks input directions as if every one were equally likely to occur.
The model never visits most of them. Sigma is the covariance of real activations,
and h = Sigma^(1/2) z rewrites any activation in coordinates where one unit means
one standard deviation of what the model actually does. Then

    J h = (J Sigma^(1/2)) z

so the SVD of J Sigma^(1/2) ranks directions by output produced per standard
deviation of real input, which is the question J's own SVD was answering badly.

The cost, and the reason this needs a test rather than an argument: J is already
an average over whatever text estimated it, and Sigma is a second such estimate.
Two prompt-dependent ingredients instead of one. It has to earn that.

The comparison is the topic-decoding task, at matched numbers of directions,
against the same baselines as before: the activations' own principal directions,
random orthogonal directions, and shuffled labels.

Sigma must be estimated from far more tokens than dimensions or it is
rank-deficient and silently discards most of the space -- 12k tokens for d=576.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.subspace_meaning import TOPICS, nearest_centroid_loo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--ntok", type=int, default=12000)
    ap.add_argument("--nrand", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("out/rare/whitened.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blk = model.model.layers

    prompts, topic = [], []
    for t, ps in TOPICS.items():
        prompts += ps; topic += [t] * len(ps)

    # ---- Sigma from pile, and the topic activations, in one pass each ----
    store = {l: [] for l in a.layers}
    hooks = []
    def mk(l, bag):
        def f(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            bag.append(t[0, 1:].detach().float().cpu())      # drop the sink
        return f
    pile = {l: [] for l in a.layers}
    for l in a.layers: hooks.append(blk[l].register_forward_hook(mk(l, pile[l])))
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    n = 0
    for i in range(400):
        ids = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                  max_length=128)["input_ids"].to(dev)
        if ids.shape[1] < 8: continue
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
        n += ids.shape[1]
        if n > a.ntok: break
    for h in hooks: h.remove()

    hooks = []
    for l in a.layers: hooks.append(blk[l].register_forward_hook(mk(l, store[l])))
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks: h.remove()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(0)
    out = {}
    for l in a.layers:
        P = torch.cat(pile[l], 0)
        A = P - P.mean(0, keepdim=True)
        S = (A.T @ A) / (A.shape[0] - 1)
        ev, Wv = torch.linalg.eigh(S); ev = ev.clamp(min=1e-8)
        Sh = Wv @ torch.diag(ev.sqrt()) @ Wv.T
        Sinv = Wv @ torch.diag(ev.rsqrt()) @ Wv.T
        J = blob["J"][l].float()
        Vj = torch.linalg.svd(J, full_matrices=False)[2]
        Vw = torch.linalg.svd(J @ Sh, full_matrices=False)[2]

        H = torch.stack([x.mean(0) for x in store[l]])
        Hc = H - H.mean(0, keepdim=True)
        Z = Hc @ Sinv.T                                   # whitened coordinates
        Vp = torch.linalg.svd(Hc, full_matrices=False)[2]
        d = H.shape[1]

        ov = float((Vj[:32] @ Vw[:32].T).pow(2).sum() / 32)
        print(f"\n{'='*72}\nlayer {l}   Sigma from {P.shape[0]} tokens, "
              f"effective rank {float(ev.sum()**2/(ev**2).sum()):.1f} of {d}")
        print(f"  top-32 subspace overlap between J and J*Sigma^(1/2): {ov:.3f}")
        print(f"\n{'directions':>11s} {'J':>7s} {'J*Sigma^1/2':>13s} "
              f"{'activation PCs':>15s} {'random':>9s} {'shuffled':>9s}")
        res = {}
        for k in a.ks:
            if k > d: continue
            aj = nearest_centroid_loo(Hc @ Vj[:k].T, topic)
            aw = nearest_centroid_loo(Z @ Vw[:k].T, topic)
            app = nearest_centroid_loo(Hc @ Vp[:k].T, topic)
            rs = [nearest_centroid_loo(
                    Hc @ torch.linalg.qr(torch.randn(d, k, generator=g))[0], topic)
                  for _ in range(a.nrand)]
            shuf = [topic[i] for i in torch.randperm(len(topic), generator=g).tolist()]
            sn = nearest_centroid_loo(Hc @ Vj[:k].T, shuf)
            res[k] = dict(J=aj, Jw=aw, pca=app, rand=st.mean(rs), shuf=sn)
            print(f"{k:11d} {aj:7.0%} {aw:13.0%} {app:15.0%} {st.mean(rs):9.0%} {sn:9.0%}")
        out[l] = dict(overlap=ov, res=res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
