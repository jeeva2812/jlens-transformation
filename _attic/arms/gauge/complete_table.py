"""The two symmetries the main audit skipped, tested on OLMoE (untied embeddings).

  expert_permute  swap two experts and swap the matching router rows
  norm_absorb     fold the final normalisation's scale into the unembedding

Both are exact. The audit's main models all tie their input and output
embeddings, so norm_absorb could not be tested there -- scaling the unembedding
would also have scaled the input embedding. OLMoE does not tie them.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from jlens.privilege import Weights

M = "allenai/OLMoE-1B-7B-0924"


def main():
    w = Weights(M)
    res = {}

    # ---- 1. expert permutation -------------------------------------------
    li = 8
    R = w.get(f"model.layers.{li}.mlp.gate.weight")          # (64, d)
    E = w.cfg["num_experts"]; k = w.cfg["num_experts_per_tok"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(E, generator=g)
    x = torch.randn(8, R.shape[1], generator=g)
    s0 = R @ x.T                                              # (E, n)
    s1 = R[perm] @ x.T
    top0 = s0.topk(k, 0).indices                              # experts chosen
    top1 = perm[s1.topk(k, 0).indices]                        # map back through perm
    same = float((top0.sort(0).values == top1.sort(0).values).float().mean())
    w0 = torch.softmax(s0.topk(k, 0).values, 0)
    w1 = torch.softmax(s1.topk(k, 0).values, 0)
    print("EXPERT PERMUTATION (swap experts, swap the matching router rows)")
    print(f"  same experts selected, after undoing the relabelling: {same*100:.1f}%")
    print(f"  same routing weights: max diff {float((w0-w1).abs().max()):.2e}")
    print(f"  but 'expert 27' now names a different expert -- the index is a label,")
    print(f"  and it does not survive a different training seed either.")
    res["expert_permute"] = {"same_experts": same,
                             "weight_diff": float((w0 - w1).abs().max())}

    # ---- 2. norm absorption ----------------------------------------------
    gam = w.get("model.norm.weight")
    W_U = w.get("lm_head.weight")
    W_E = w.get("model.embed_tokens.weight")
    tied = bool(torch.equal(W_U, W_E))
    h = torch.randn(6, gam.shape[0], generator=g)
    rms = h.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    before = (gam * (h * rms)) @ W_U.T                        # normal path
    after = (h * rms) @ (W_U * gam.unsqueeze(0)).T            # gamma folded into W_U
    print(f"\nNORM ABSORPTION (fold the final scale into the unembedding)")
    print(f"  input and output embeddings tied? {tied}  (must be False to do this)")
    print(f"  logits identical: max diff {float((before-after).abs().max()):.2e}")
    # what it costs a readout that forgets to fold it
    d = torch.randn(64, gam.shape[0], generator=g)
    d = d / d.norm(dim=1, keepdim=True)
    Wc = W_U - W_U.mean(0, keepdim=True)
    a = ((d * gam) @ Wc.T).topk(10, 1).indices
    b = (d @ Wc.T).topk(10, 1).indices
    ov = float(np.mean([len(set(p.tolist()) & set(q.tolist())) / 10 for p, q in zip(a, b)]))
    print(f"  a readout that folds gamma vs one that forgets: top-10 overlap {ov*100:.1f}%")
    res["norm_absorb"] = {"tied": tied,
                          "logit_diff": float((before - after).abs().max()),
                          "readout_overlap_if_forgotten": ov}
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path("out/gauge/complete_table.json").write_text(json.dumps(res, indent=1))
    print("\nwrote out/gauge/complete_table.json")


if __name__ == "__main__":
    main()
