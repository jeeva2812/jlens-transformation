"""CENTRED version. J*h is not the first-order estimate of the router input:
    x ~ x0 + J (h - h0)   =>   W_r[e].x ~ W_r[e].x0 + (J^T W_r[e]).(h - h0)
The uncentred test dropped BOTH the per-expert baseline W_r[e].x0 and the
centering. Router base rates span 0.15-0.75, so that baseline alone can decide
the ranking. This version adds both, using corpus means.

Predict routing n TOKENS ahead, not just layers ahead.

Same layer / same position, the identity wins because h_l and h_L are one shared
residual stream. Across POSITIONS there is no such shortcut: h[t] and h[t+n] are
different streams and the only path between them is attention. The Jacobian
d h_L[t+n] / d h_l[t] is exactly that path, so this is the regime where it
should earn its cost.

Efficiency: seed the VJP at ONE target position and the gradient lands at every
source position at once, so a single backward per expert gives the whole n sweep.
"""
from __future__ import annotations
import argparse, statistics as st, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--src", type=int, nargs="+", default=[8, 10, 11])
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--maxlen", type=int, default=24)
    ap.add_argument("--append", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts; K = m.config.num_experts_per_tok
    L = a.target; SRC = [l for l in a.src if l < L]
    Wr = blk[L].mlp.gate.weight.detach()
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    # corpus means for the expansion point
    NM = 12
    xs, hbar = [], {l: [] for l in SRC}
    for i in range(500, 500 + NM):
        ii = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                 max_length=a.maxlen)["input_ids"]
        hh, rr, hk = {}, {}, []
        for l in SRC:
            hk.append(blk[l].register_forward_hook(
                (lambda l: (lambda mod, inp, out: hh.__setitem__(
                    l, (out[0] if isinstance(out, tuple) else out))))(l)))
        hk.append(blk[L].mlp.register_forward_pre_hook(
            lambda mod, inp: rr.__setitem__("x", inp[0])))
        with T.no_grad(): m(input_ids=ii, use_cache=False)
        xs.append(rr["x"][0].float().mean(0))
        for l in SRC: hbar[l].append(hh[l][0].float().mean(0))
        for h in hk: h.remove()
    x0 = T.stack(xs).mean(0)
    h0 = {l: T.stack(hbar[l]).mean(0) for l in SRC}
    base_e = Wr.float() @ x0                       # the omitted per-expert baseline
    print(f"baseline W_r.x0 spread: {float(base_e.min()):.2f} to {float(base_e.max()):.2f}",
          flush=True)
    rows = []
    for pi in range(a.skip, a.skip + a.n):
        ids = tok(ds[pi]["text"], return_tensors="pt", truncation=True,
                  max_length=a.maxlen)["input_ids"]
        Tn = ids.shape[1]
        tgt = Tn - 1                                   # predict routing HERE
        offs = [o for o in a.offsets if tgt - o >= 1]
        hs, rin, hooks = {}, {}, []
        for l in SRC:
            def mk(l):
                def f(mod, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    if l == SRC[0]: t.requires_grad_(True)
                    hs[l] = t
                return f
            hooks.append(blk[l].register_forward_hook(mk(l)))
        hooks.append(blk[L].mlp.register_forward_pre_hook(
            lambda mod, inp: rin.__setitem__("x", inp[0])))
        with T.enable_grad():
            m(input_ids=ids, use_cache=False)
        x = rin["x"]
        true = set(T.topk(Wr.float() @ x[0, tgt].detach().float(), K).indices.tolist())
        pull = {l: T.zeros(E, Tn, x.shape[-1]) for l in SRC}
        for e in range(E):
            go = T.zeros_like(x); go[0, tgt] = Wr[e]
            gs = T.autograd.grad(x, [hs[l] for l in SRC], grad_outputs=go,
                                 retain_graph=(e < E - 1), allow_unused=True)
            for l, g in zip(SRC, gs):
                if g is not None: pull[l][e] = g[0].detach().float()
        for l in SRC:
            for o in offs:
                sp = tgt - o
                h = hs[l][0, sp].detach().float()
                pj = set(T.topk(base_e + pull[l][:, sp, :] @ (h - h0[l]),
                                K).indices.tolist())
                pi_ = set(T.topk(Wr.float() @ h, K).indices.tolist())
                rows.append({"src": l, "gap": L - l, "off": o,
                             "j": len(true & pj), "id": len(true & pi_)})
        for h in hooks: h.remove()
        del hs, rin, x, pull
        print(f"  prompt {pi} done", flush=True)
    if a.append:
        import json
        with open(a.append, "a") as f:
            for r in rows: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
