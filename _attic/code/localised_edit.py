"""Does WHERE a change sits matter, holding WHAT changed fixed?

The attribution test on the real fine-tune was underpowered: a rank-32 LoRA on
all seven projections changes every layer by roughly the same amount, so the
ground truth spanned 0.63-1.53 and there was almost nothing to predict.

Here the edit is localised by construction. Take the fine-tuned model and revert
EVERY layer except one back to pretrained. The resulting model differs from base
at exactly one known layer, so attribution has a ground truth it cannot miss --
and sweeping which layer is kept asks the question the real fine-tune could not:

    holding the weight change roughly fixed, does its POSITION change how much
    the end-to-end transport moves?

If early edits move J more than late ones at matched ||dW||, position matters
and the sandwich formula has something to explain. If the effect tracks ||dW||
alone, it does not, and the honest conclusion is that dJ attribution reduces to
"which layer changed most".
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
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--keep", type=int, nargs="+",
                    default=[6, 10, 14, 18, 22, 26])
    ap.add_argument("--ntext", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("out/localised_edit.json"))
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
    J_base = jacobians_all_layers(base, batches(), [a.s], a.target,
                                  chunk=192)[a.s].float()

    # working model starts as a copy of base; we graft ONE fine-tuned layer in
    work = load(a.base_dir)
    wb, _ = _find_blocks_and_norm(work)
    print(f"one fine-tuned layer grafted onto the base model at a time\n")
    print(f"{'layer kept':>11}{'||dW||':>10}{'||dJ||':>10}{'dJ per unit dW':>16}")
    print("-" * 48)
    rows = []
    for u in a.keep:
        if u >= len(wb):
            continue
        saved = copy.deepcopy(wb[u].state_dict())
        ftsd = fb[u].state_dict()
        dW = float(sum(((ftsd[k].float() - saved[k].float()) ** 2).sum()
                       for k in ftsd) ** 0.5)
        wb[u].load_state_dict(ftsd)
        J_loc = jacobians_all_layers(work, batches(), [a.s], a.target,
                                     chunk=192)[a.s].float()
        wb[u].load_state_dict(saved)
        dJ = float((J_loc - J_base).norm())
        rows.append({"layer": u, "dW": dW, "dJ": dJ, "per_unit": dJ / max(dW, 1e-9)})
        print(f"{u:>11}{dW:>10.3f}{dJ:>10.3f}{dJ/max(dW,1e-9):>16.4f}", flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    L = np.array([r["layer"] for r in rows], dtype=float)
    dW = np.array([r["dW"] for r in rows])
    dJ = np.array([r["dJ"] for r in rows])
    pu = np.array([r["per_unit"] for r in rows])
    print(f"\n  corr(||dW||, ||dJ||)          = {np.corrcoef(dW,dJ)[0,1]:+.3f}"
          "   <- does dJ just track how much changed?")
    print(f"  corr(layer index, dJ per unit) = {np.corrcoef(L,pu)[0,1]:+.3f}"
          "   <- does POSITION matter, at matched dW?")
    print(f"\n  dJ per unit dW:  earliest {pu[0]:.4f}   latest {pu[-1]:.4f}"
          f"   ratio {pu[0]/max(pu[-1],1e-9):.2f}x")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
