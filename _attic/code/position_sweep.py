"""The full position curve, with a control that makes it a claim about architecture.

The six-layer graft showed an identical-size weight change moving the end-to-end
transport 2.87x more at layer 6 than at layer 26 (r = -0.98). Two things were
missing.

EVERY LAYER, not six, so the shape of the curve is visible rather than inferred.

A RANDOM-PERTURBATION CONTROL. Grafting a fine-tuned layer confounds "changes
early matter more" with "this fine-tune's early changes matter more". Perturbing
each layer with random noise of MATCHED norm separates them: if the position
gradient survives random noise, it is a property of the architecture -- depth
remaining to compound through -- and not of what the fine-tune learned.

BEHAVIOUR, not just geometry. ||dJ|| is a statement about the linearisation;
KL on real text is a statement about the model. If position predicts both, the
practical claim ("editing early is more efficient per unit weight change") has
something behind it.
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
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--ntext", type=int, default=4)
    ap.add_argument("--nkl", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/position_sweep.json"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import jacobians_all_layers, _find_blocks_and_norm
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]
    klenc = [tok(ds[100 + i]["text"], return_tensors="pt", truncation=True,
                 max_length=96) for i in range(a.nkl)]

    def batches():
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=96)
            yield e["input_ids"], e["attention_mask"]

    def load(d):
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    ft = load(a.ft_dir)
    fb, _ = _find_blocks_and_norm(ft)
    work = load(a.base_dir)
    wb, _ = _find_blocks_and_norm(work)
    n = len(wb)

    J_base = jacobians_all_layers(work, batches(), [a.s], a.target,
                                  chunk=192)[a.s].float()

    def base_dists():
        out = []
        for e in klenc:
            with torch.no_grad():
                out.append(torch.log_softmax(work(**e).logits[0].float(), -1))
        return out
    D0 = base_dists()

    def kl_now():
        tot = 0.0
        for e, b in zip(klenc, D0):
            with torch.no_grad():
                c = torch.log_softmax(work(**e).logits[0].float(), -1)
            tot += float((b.exp() * (b - c)).sum(-1).mean())
        return tot / len(D0)

    def measure():
        J = jacobians_all_layers(work, batches(), [a.s], a.target,
                                 chunk=192)[a.s].float()
        return float((J - J_base).norm()), kl_now()

    g = torch.Generator().manual_seed(a.seed)
    rows = []
    print(f"{n} blocks; J measured from layer {a.s} to {a.target}\n")
    print(f"{'layer':>6} | {'GRAFT: ||dW||':>13}{'||dJ||':>9}{'per unit':>10}{'KL':>9}"
          f" | {'RANDOM: ||dJ||':>15}{'per unit':>10}{'KL':>9}")
    print("-" * 96)
    for u in range(a.s, min(a.target, n)):
        saved = copy.deepcopy(wb[u].state_dict())
        ftsd = fb[u].state_dict()
        dW = float(sum(((ftsd[k].float() - saved[k].float()) ** 2).sum()
                       for k in ftsd) ** 0.5)

        wb[u].load_state_dict(ftsd)
        dJ_g, kl_g = measure()
        wb[u].load_state_dict(saved)

        # random perturbation of the SAME total norm, same parameter shapes
        pert = {}
        sq = 0.0
        for k, v in saved.items():
            if not torch.is_floating_point(v):
                pert[k] = v.clone(); continue
            r = torch.randn(v.shape, generator=g, dtype=torch.float32)
            pert[k] = r; sq += float((r ** 2).sum())
        scale = dW / max(sq ** 0.5, 1e-9)
        newsd = {k: (saved[k].float() + scale * pert[k]).to(saved[k].dtype)
                 if torch.is_floating_point(saved[k]) else saved[k]
                 for k in saved}
        wb[u].load_state_dict(newsd)
        dJ_r, kl_r = measure()
        wb[u].load_state_dict(saved)

        rows.append({"layer": u, "dW": dW,
                     "dJ_graft": dJ_g, "kl_graft": kl_g,
                     "dJ_rand": dJ_r, "kl_rand": kl_r,
                     "pu_graft": dJ_g / max(dW, 1e-9),
                     "pu_rand": dJ_r / max(dW, 1e-9)})
        print(f"{u:>6} | {dW:>13.2f}{dJ_g:>9.3f}{dJ_g/max(dW,1e-9):>10.4f}"
              f"{kl_g:>9.4f} | {dJ_r:>15.3f}{dJ_r/max(dW,1e-9):>10.4f}{kl_r:>9.4f}",
              flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    L = np.array([r["layer"] for r in rows], dtype=float)
    print(f"\n{'quantity':<34}{'corr with layer index':>24}")
    print("-" * 58)
    for nm, key in [("graft: dJ per unit dW", "pu_graft"),
                    ("graft: KL", "kl_graft"),
                    ("RANDOM: dJ per unit dW", "pu_rand"),
                    ("RANDOM: KL", "kl_rand"),
                    ("(||dW|| itself)", "dW")]:
        v = np.array([r[key] for r in rows])
        print(f"{nm:<34}{np.corrcoef(L, v)[0, 1]:>+24.3f}")
    pg = np.array([r["pu_graft"] for r in rows])
    pr = np.array([r["pu_rand"] for r in rows])
    print(f"\n  graft  early/late ratio: {pg[0]/max(pg[-1],1e-9):.2f}x")
    print(f"  random early/late ratio: {pr[0]/max(pr[-1],1e-9):.2f}x")
    print("\n  If the random control shows the same gradient, the effect is the")
    print("  architecture -- depth left to compound through -- not what was learnt.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
