"""Predict routing n TOKENS ahead, not just layers ahead.

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
                pj = set(T.topk(pull[l][:, sp, :] @ h, K).indices.tolist())
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
