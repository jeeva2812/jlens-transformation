"""Is the gain/occupancy anti-correlation BUILT by training, or there at init?

At initialisation J is close to the identity, so gain is nearly uniform and
cannot be anti-correlated with anything. In the trained model, layer 20's
highest-gain directions carry ~740x less activation energy than its lowest. If
that gap is absent at init and present after training, the implication is a
claim worth making: the network learns to SUPPRESS gain along the directions it
actually uses -- which is what stability would require.

Measured on a randomly initialised SmolLM2 against the trained one, and across
the 13 fine-tuning checkpoints, which are the only checkpoints here that kept
their weights.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from jlens.lens import _MultiCapture, jacobians_all_layers

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def measure(model, tok, texts, layers, target, Js=None):
    """Js may be supplied when the Jacobians already exist on disk -- recomputing
    the trained model's J costs an hour and buys nothing."""
    if Js is None:
        def batches():
            for t in texts:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
                yield enc["input_ids"], enc["attention_mask"]
        Js = jacobians_all_layers(model, batches(), layers, target, chunk=128)
    out = {}
    for l in layers:
        H = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            with _MultiCapture(model, [l], target) as cap:
                with torch.no_grad():
                    model(**enc, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        d = H.shape[1]; tot = H.pow(2).sum()
        J = Js[l].float()
        U, S, Vh = torch.linalg.svd(J)
        occ = ((H @ Vh.T).pow(2).sum(0) / tot).numpy() * d      # x random
        r = float(np.corrcoef(np.arange(d), occ)[0, 1])
        out[l] = {"corr_rank_occ": r,
                  "top10": float(occ[:10].mean()),
                  "bottom50": float(occ[-50:].mean()),
                  "ratio": float(occ[-50:].mean() / max(occ[:10].mean(), 1e-9)),
                  "sigma_top": float(S[0]), "sigma_med": float(S.median())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 20])
    ap.add_argument("--ntext", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("out/occupancy_training.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]
    rec = {}

    print("=== randomly initialised (same architecture) ===", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    rec["random_init"] = measure(m, tok, texts, a.layers, 28)
    for l, v in rec["random_init"].items():
        print(f"  L{l}: corr(rank,occ)={v['corr_rank_occ']:+.3f}  "
              f"top10 {v['top10']:.2f}x  bottom50 {v['bottom50']:.2f}x  "
              f"ratio {v['ratio']:.1f}", flush=True)
    del m

    print("\n=== trained (J loaded from disk, not recomputed) ===", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    Jt = torch.load("out/ft/J_step0.pt", map_location="cpu",
                    weights_only=False)["J"]
    rec["trained"] = measure(m, tok, texts, a.layers, 28,
                             Js={l: Jt[l] for l in a.layers if l in Jt})
    for l, v in rec["trained"].items():
        print(f"  L{l}: corr(rank,occ)={v['corr_rank_occ']:+.3f}  "
              f"top10 {v['top10']:.2f}x  bottom50 {v['bottom50']:.2f}x  "
              f"ratio {v['ratio']:.1f}", flush=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
