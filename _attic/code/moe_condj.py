"""Does the router label the network's local linear regime?

J is a prompt-average. In an MoE, prompts route through different experts --
different computational graphs -- so the average is over structurally different
functions. Chess showed what that does: cos(J_origin, J_dest) = 0.21, and the
top of the pooled J encoded the MIXTURE rather than any computation. In a dense
model you cannot fix it because you do not know the regimes. In an MoE the
router names them.

Cheap test of the premise, no full Jacobian needed: probe J with a few random
target-space vectors w (one VJP each gives J^T w), then ask whether J^T w is
more similar between prompts that ROUTE THE SAME than between prompts that do
not.

  within-cluster  >  across-cluster   =>  routing labels the linear regime
  no gap                              =>  routing is orthogonal to the Jacobian

Control: the identical statistic under a RANDOM clustering of the same shape.
"""
from __future__ import annotations
import argparse, itertools, json, statistics as st
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--target", type=int, default=13)
    ap.add_argument("--nseed", type=int, default=6)
    ap.add_argument("--maxlen", type=int, default=20)
    ap.add_argument("--append", default="out/condj.jsonl")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts; K = m.config.num_experts_per_tok
    L, S = a.target, a.src
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    g = T.Generator().manual_seed(0)
    d = m.config.hidden_size
    Wseed = T.randn(a.nseed, d, generator=g)
    Wseed = Wseed / Wseed.norm(dim=1, keepdim=True)
    out = []
    for pi in range(a.skip, a.skip + a.n):
        ids = tok(ds[pi]["text"], return_tensors="pt", truncation=True,
                  max_length=a.maxlen)["input_ids"]
        hs, rin, hooks = {}, {}, []
        # a forward hook that RETURNS something replaces the layer output. The
        # comma-expression lambda returned a tuple and corrupted the forward pass.
        def src_hook(mod, inp, out_):
            t = out_[0] if isinstance(out_, tuple) else out_
            t.requires_grad_(True)
            hs["h"] = t
        def tgt_hook(mod, inp, out_):
            hs["t"] = out_[0] if isinstance(out_, tuple) else out_
        hooks.append(blk[S].register_forward_hook(src_hook))
        hooks.append(blk[L].register_forward_hook(tgt_hook))
        def mkr(l):
            def f(mod, inp):
                rin[l] = inp[0]
            return f
        for l in (8, 10, 12):
            hooks.append(blk[l].mlp.register_forward_pre_hook(mkr(l)))
        with T.enable_grad():
            m(input_ids=ids, use_cache=False)
        pos = ids.shape[1] - 1
        gs = []
        for si in range(a.nseed):
            go = T.zeros_like(hs["t"]); go[0, pos] = Wseed[si]
            (gr,) = T.autograd.grad(hs["t"], hs["h"], grad_outputs=go,
                                    retain_graph=(si < a.nseed - 1))
            gs.append(gr[0, pos].detach().float().tolist())
        route = {}
        for l in (8, 10, 12):
            r = blk[l].mlp.gate.weight.detach().float() @ rin[l][0, pos].detach().float()
            route[l] = sorted(T.topk(r, K).indices.tolist())
        for h in hooks: h.remove()
        out.append({"i": pi, "g": gs, "route": {str(k): v for k, v in route.items()}})
        print(f"  prompt {pi} done", flush=True)
        del hs, rin
    with open(a.append, "a") as f:
        for r in out: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
