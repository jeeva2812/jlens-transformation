"""Does transport-weighted attribution predict which layers actually control dJ?

The two-time result says a layer's contribution to the end-to-end change is its
own weight change SANDWICHED between the transports either side:

    dJ(T,s) = sum_u  Phi(T,u+1) . dA_u . Phi(u,s)

That makes a claim worth testing, because it separates two things that are easy
to conflate: how much a layer CHANGED, and how much that change MATTERS given
where the layer sits. A layer can change a lot and have it suppressed
downstream, or barely change and sit somewhere the change propagates.

  prediction NAIVE       ||dA_u||                      -- how much layer u changed
  prediction TRANSPORT   ||Phi(T,u+1) dA_u Phi(u,s)||  -- weighted by position

GROUND TRUTH is causal and exact. Take the fine-tuned model, revert layer u
alone to its pretrained weights, recompute J, and measure how far J moves back:

    actual_u = ||J_ft(s) - J_revert-u(s)||

The question is whether transport weighting buys anything over the naive
version. If it does not, the sandwich is decoration and the honest thing is to
say so.
"""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--base-dir", default="out/ft/step0")
    ap.add_argument("--ft-dir", default="out/ft/step600")
    ap.add_argument("--s", type=int, default=8, help="source layer for J")
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--ntext", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("out/layer_attribution.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import jacobians_all_layers, _find_blocks_and_norm
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]

    def batches():
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=96)
            yield e["input_ids"], e["attention_mask"]

    def load(d):
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    base, ft = load(a.base_dir), load(a.ft_dir)
    bb, _ = _find_blocks_and_norm(base)
    fb, _ = _find_blocks_and_norm(ft)
    n_layers = len(fb)
    us = [u for u in range(a.s, min(a.target, n_layers))]
    print(f"{n_layers} blocks; attributing layers {us[0]}-{us[-1]} for J(s={a.s})\n",
          flush=True)

    J_ft = jacobians_all_layers(ft, batches(), [a.s], a.target, chunk=192)[a.s].float()
    J_bs = jacobians_all_layers(base, batches(), [a.s], a.target, chunk=192)[a.s].float()
    print(f"||J_ft - J_base|| = {(J_ft-J_bs).norm():.3f}\n", flush=True)

    # transports needed by the formula
    Phi_ft = jacobians_all_layers(ft, batches(), us, a.target, chunk=192)
    Phi_bs_from_s = {}
    for u in us:
        if u == a.s:
            Phi_bs_from_s[u] = torch.eye(J_ft.shape[0])
        else:
            Phi_bs_from_s[u] = jacobians_all_layers(
                base, batches(), [a.s], u, chunk=192)[a.s].float()

    rows = []
    print(f"{'layer':>6}{'||dA_u|| naive':>16}{'transport-wtd':>15}"
          f"{'ACTUAL (revert)':>17}")
    print("-" * 56)
    for u in us:
        # dA_u : the change in layer u's own local transform
        if u + 1 <= a.target:
            A_ft = jacobians_all_layers(ft, batches(), [u], min(u+1, a.target),
                                        chunk=192)[u].float()
            A_bs = jacobians_all_layers(base, batches(), [u], min(u+1, a.target),
                                        chunk=192)[u].float()
            dA = A_ft - A_bs
        else:
            continue
        naive = float(dA.norm())
        L = Phi_ft[min(u+1, a.target)].float() if min(u+1, a.target) in Phi_ft \
            else torch.eye(dA.shape[0])
        wtd = float((L @ dA @ Phi_bs_from_s[u]).norm())

        # ground truth: revert layer u alone, recompute J
        saved = copy.deepcopy(fb[u].state_dict())
        fb[u].load_state_dict(bb[u].state_dict())
        J_rev = jacobians_all_layers(ft, batches(), [a.s], a.target,
                                     chunk=192)[a.s].float()
        fb[u].load_state_dict(saved)
        actual = float((J_ft - J_rev).norm())

        rows.append({"layer": u, "naive": naive, "weighted": wtd, "actual": actual})
        print(f"{u:>6}{naive:>16.3f}{wtd:>15.3f}{actual:>17.3f}", flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    n = np.array([r["naive"] for r in rows])
    w = np.array([r["weighted"] for r in rows])
    y = np.array([r["actual"] for r in rows])
    def sp(x, z):
        from scipy.stats import spearmanr
        return spearmanr(x, z).statistic
    print(f"\n{'predictor':<22}{'Pearson':>10}{'Spearman':>11}")
    print("-" * 43)
    for nm, x in [("naive ||dA_u||", n), ("transport-weighted", w)]:
        try:
            print(f"{nm:<22}{np.corrcoef(x,y)[0,1]:>+10.3f}{sp(x,y):>+11.3f}")
        except Exception:
            print(f"{nm:<22}{np.corrcoef(x,y)[0,1]:>+10.3f}{'n/a':>11}")
    print("\nIf transport weighting does not beat the naive norm, the sandwich is")
    print("decoration and the attribution reduces to 'which layer changed most'.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
