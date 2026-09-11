"""Training-free expert prefetching: can a Jacobian pullback into ROUTER space
predict which experts fire at layer L, from the residual several layers earlier?

   true at L:        W_router[L][e] . router_input_L
   predicted from l: (J_{l->L}^T W_router[L][e]) . h_l

The pulled-back router rows are computed ONCE offline from weights; at runtime
it is a dot product. No trained predictor, no traces, no per-model fitting --
which is the point, versus learned prefetchers.

Baselines (both required):
  IDENTITY -- apply W_router[L] to h_l directly, ignoring the transport. This is
              what the Jacobian must beat to be worth computing at all.
  chance   -- 8*8/64 = 1.0 of 8.
Metric is top-8 overlap: what a prefetcher actually needs.
"""
from __future__ import annotations
import argparse, statistics as st, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--maxlen", type=int, default=24)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--append", default=None,
                    help="one prompt per process; memory is not reclaimed between\n                          iterations inside one process (21GB RSS then 0%% CPU)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts; K = m.config.num_experts_per_tok
    L = a.target
    # retaining the graph for 11 source layers pushed a 32GB box into swap
    # (19.5GB RSS + 3.9GB swap, 0% CPU). Five source layers is the
    # configuration that timed cleanly at 4.55s per backward.
    SRC = [l for l in (4, 6, 8, 10, 11) if l < L]
    Wr = blk[L].mlp.gate.weight.detach()
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.skip, a.skip + a.n)]
    acc = {l: {"j": [], "id": []} for l in SRC}
    for pi, txt in enumerate(texts):
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=a.maxlen)["input_ids"]
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
        x = rin["x"]; pos = ids.shape[1] - 1
        true = (Wr.float() @ x[0, pos].detach().float())
        tt = set(T.topk(true, K).indices.tolist())
        pull = {l: T.zeros(E, x.shape[-1]) for l in SRC}
        for e in range(E):
            go = T.zeros_like(x); go[0, pos] = Wr[e]
            gs = T.autograd.grad(x, [hs[l] for l in SRC], grad_outputs=go,
                                 retain_graph=(e < E - 1), allow_unused=True)
            for l, g in zip(SRC, gs):
                if g is not None: pull[l][e] = g[0, pos].detach().float()
        for l in SRC:
            hl = hs[l][0, pos].detach().float()
            pj = set(T.topk(pull[l] @ hl, K).indices.tolist())
            pidt = set(T.topk(Wr.float() @ hl, K).indices.tolist())
            acc[l]["j"].append(len(tt & pj)); acc[l]["id"].append(len(tt & pidt))
        for h in hooks: h.remove()
        del hs, rin, x
        print(f"  prompt {pi+1}/{len(texts)} done", flush=True)
    if a.append:
        import json
        with open(a.append, "a") as f:
            for l in SRC:
                f.write(json.dumps({"src": l, "gap": L - l,
                                    "j": acc[l]["j"], "id": acc[l]["id"]}) + "\n")
        return
    print(f"\ntarget layer L{L}, top-{K} of {E}. chance overlap = {K*K/E:.1f}\n")
    print(f"{'gap':>4} {'src':>4} {'J pullback':>11} {'identity':>10} {'advantage':>10}")
    for l in sorted(SRC, reverse=True):
        j, i = st.mean(acc[l]["j"]), st.mean(acc[l]["id"])
        print(f"{L-l:>4} {'L'+str(l):>4} {j:>11.2f} {i:>10.2f} {j-i:>+10.2f}")

if __name__ == "__main__":
    main()
