"""What does instruction tuning actually change: reweighting, or addition?

Base and instruct share a weight basis -- instruct IS base plus a delta -- so
dW = W_inst - W_base can be compared against W directly, with no alignment step.

The question in one measurement: does dW live inside the structure W already had?
Project dW onto the top-k singular subspace of W and see what fraction of its
energy lands there. A random dW of the same shape puts k/min(m,n) of its energy
in any fixed k-dim subspace -- that is the chance line, and it is what makes
"aligned" mean something.

  energy >> chance  ->  tuning AMPLIFIES directions the base already used
  energy ~= chance  ->  tuning writes into directions unrelated to base structure
"""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM

def eff_rank(S):
    """participation ratio: (sum s)^2 / sum s^2 -- 1 = rank one, n = flat"""
    return float((S.sum()**2) / (S**2).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--inst", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("out/instruct_diff.json"))
    a = ap.parse_args()
    B = AutoModelForCausalLM.from_pretrained(a.base, dtype=T.float32)
    I = AutoModelForCausalLM.from_pretrained(a.inst, dtype=T.float32)
    Bd, Id = dict(B.named_parameters()), dict(I.named_parameters())
    shared = [k for k in Bd if k in Id and Bd[k].shape == Id[k].shape and Bd[k].dim() == 2]
    print(f"{len(shared)} shared 2-D weight matrices\n")

    rows = []
    for k in shared:
        W = Bd[k].detach(); dW = (Id[k].detach() - W)
        if dW.norm() < 1e-9:
            continue
        U, S, Vh = T.linalg.svd(W, full_matrices=False)
        Sd = T.linalg.svdvals(dW)
        kk = min(a.topk, U.shape[1])
        # energy of dW inside W's top-kk left singular subspace
        proj = U[:, :kk].T @ dW
        frac = float((proj**2).sum() / (dW**2).sum())
        # EMPIRICAL chance, not kk/min(W.shape). U's columns live in R^m, so for a
        # tall matrix a random dW lands at kk/m, not kk/min(m,n) -- getting this
        # wrong manufactured a 5.4x artifact in the MLP projections and with it a
        # spurious "attention reweights, MLP adds" dissociation.
        g = T.Generator().manual_seed(hash(k) % (2**31))
        R = T.randn(W.shape, generator=g)
        chance = float(((U[:, :kk].T @ R)**2).sum() / (R**2).sum())
        rows.append({"name": k, "shape": list(W.shape),
                     "rel": float(dW.norm()/W.norm()),
                     "eff_rank_W": eff_rank(S), "eff_rank_dW": eff_rank(Sd),
                     "aligned": frac, "chance": chance, "ratio": frac/chance})
    rows.sort(key=lambda r: -r["rel"])

    def kind(n):
        for t in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                  "down_proj","embed_tokens","lm_head"):
            if t in n: return t
        return "other"
    agg = collections.defaultdict(list)
    for r in rows: agg[kind(r["name"])].append(r)
    print(f"{'component':>13} {'n':>3} {'||dW||/||W||':>13} {'effrank W':>10} "
          f"{'effrank dW':>11} {'aligned':>8} {'chance':>7} {'x chance':>9}")
    for t, rs in sorted(agg.items(), key=lambda x: -sum(r['rel'] for r in x[1])/len(x[1])):
        m = lambda f: sum(f(r) for r in rs)/len(rs)
        print(f"{t:>13} {len(rs):>3} {m(lambda r:r['rel']):>13.4f} "
              f"{m(lambda r:r['eff_rank_W']):>10.1f} {m(lambda r:r['eff_rank_dW']):>11.1f} "
              f"{m(lambda r:r['aligned']):>8.3f} {m(lambda r:r['chance']):>7.3f} "
              f"{m(lambda r:r['ratio']):>8.2f}x")
    tot = sum(r["rel"] for r in rows)/len(rows)
    al = sum(r["aligned"] for r in rows)/len(rows)
    ch = sum(r["chance"] for r in rows)/len(rows)
    print(f"\noverall: ||dW||/||W|| = {tot:.4f}   aligned {al:.3f} vs chance {ch:.3f}"
          f"  =  {al/ch:.2f}x")
    print(f"\neffective rank of dW vs of W (mean over matrices): "
          f"{sum(r['eff_rank_dW'] for r in rows)/len(rows):.1f} vs "
          f"{sum(r['eff_rank_W'] for r in rows)/len(rows):.1f}")
    a.out.write_text(json.dumps(rows))

if __name__ == "__main__":
    main()
