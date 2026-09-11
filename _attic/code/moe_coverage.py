"""Two things, both forward-only (no backward passes, so we can afford many prompts):

1. HOW CLOSE IS J TO THE IDENTITY? If h_{l+1} = h_l + small, then transport is
   mostly identity and a Jacobian correction has almost nothing to correct.
   Measured as cos(h_l, h_L) and ||h_L - h_l|| / ||h_l||.

2. COVERAGE CURVE. If a prefetcher fetches the top-k predicted experts, what
   fraction of the true top-8 does it get? This is the number a systems person
   needs: k sets the memory budget, coverage sets the miss rate.
"""
from __future__ import annotations
import argparse, statistics as st, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--maxlen", type=int, default=48)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts; K = m.config.num_experts_per_tok
    L = a.target; SRC = list(range(1, L))
    Wr = blk[L].mlp.gate.weight.detach().float()
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    KS = [8, 12, 16, 20, 24, 32, 48]
    cov = {l: {k: [] for k in KS} for l in SRC}
    geom = {l: {"cos": [], "rel": []} for l in SRC}
    rin = {}
    hooks = [blk[L].mlp.register_forward_pre_hook(
        lambda mod, inp: rin.__setitem__("x", inp[0]))]
    for i in range(a.n):
        ids = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                  max_length=a.maxlen)["input_ids"]
        with T.no_grad():
            o = m(input_ids=ids, output_hidden_states=True)
        pos = ids.shape[1] - 1
        xr = rin["x"][0, pos].float()
        true = set(T.topk(Wr @ xr, K).indices.tolist())
        hL = o.hidden_states[L][0, pos].float()
        for l in SRC:
            hl = o.hidden_states[l][0, pos].float()
            geom[l]["cos"].append(float(T.nn.functional.cosine_similarity(hl, hL, dim=0)))
            geom[l]["rel"].append(float((hL - hl).norm() / hl.norm()))
            order = T.argsort(Wr @ hl, descending=True).tolist()
            for k in KS:
                cov[l][k].append(len(true & set(order[:k])) / K)
    for h in hooks: h.remove()
    print(f"OLMoE-1B-7B, target L{L}, {a.n} Pile prompts, top-{K} of {E}\n")
    print("1. RESIDUAL GEOMETRY -- how much does the stream actually change?")
    print(f"{'gap':>4} {'cos(h_l, h_L)':>14} {'||dh||/||h||':>13}")
    for l in sorted(SRC, reverse=True):
        print(f"{L-l:>4} {st.mean(geom[l]['cos']):>14.3f} {st.mean(geom[l]['rel']):>13.3f}")
    print("\n2. COVERAGE -- fraction of the true top-8 captured by the top-k prediction")
    print(f"   (identity readout: W_router[L] applied to h_l. no training, no J.)")
    print(f"\n{'gap':>4} " + " ".join(f"{'k='+str(k):>7}" for k in KS)
          + f"   {'memory':>8}")
    for l in sorted(SRC, reverse=True):
        row = " ".join(f"{st.mean(cov[l][k]):>7.1%}" for k in KS)
        print(f"{L-l:>4} {row}")
    print(f"\n   chance at k: " + " ".join(f"{k/E:>7.1%}" for k in KS))
    print(f"   memory cost: " + " ".join(f"{k/E:>7.1%}" for k in KS) + "  of all experts")

if __name__ == "__main__":
    main()
