"""The network suppresses transport of its own massive-activation direction.

This is what is actually left after the occupancy retraction, and it is worth
stating positively rather than as a caveat. The residual stream is dominated by
one direction -- 53% of the variance at layer 8, 99.8% at layer 16 -- and that
direction is the one J does NOT amplify. That is why occupancy looked
anti-correlated with gain: one enormous, low-gain direction was carrying the
whole measurement.

So ask it directly. For each layer, compare the gain J applies to:

    the top activation directions   (the sink / massive-activation directions)
    the bulk activation directions  (ranks 10-100, real content)
    random directions               (the baseline)

If the sink is systematically suppressed relative to both, the network is
actively declining to propagate its own largest activation -- which is what you
would want from a direction that carries no information.

Also reads the sink direction through the lens, because "what IS the massive
direction" is worth knowing and cheap to answer here.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ntext", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("out/sink_suppression.json"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture, _find_blocks_and_norm
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    W_U = m.get_output_embeddings().weight.detach().float()
    _, norm = _find_blocks_and_norm(m)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    encs = [tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=96)
            for i in range(a.ntext)]

    def read(v, k=8):
        with torch.no_grad():
            lg = norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()]

    rec = {}
    print(f"{'layer':>6}{'var in top-1':>14}{'gain: sink':>12}{'bulk':>8}"
          f"{'random':>8}{'sink/bulk':>11}{'sink/rand':>11}")
    print("-" * 72)
    for l in a.layers:
        if l not in blob["J"]:
            continue
        H = []
        for e in encs:
            with _MultiCapture(m, [l], blob.get("target", 28)) as cap:
                with torch.no_grad():
                    m(**e, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); Hc = H - H.mean(0, keepdim=True)
        d = Hc.shape[1]
        J = blob["J"][l].float()
        _, sv, Vt = torch.linalg.svd(Hc, full_matrices=False)
        share = float((sv[0] ** 2 / sv.pow(2).sum()).item())
        g = torch.Generator().manual_seed(0)

        def gain(v):
            v = v / v.norm()
            return (J @ v).norm().item()

        sink = float(np.mean([gain(Vt[i]) for i in range(3)]))
        bulk = float(np.mean([gain(Vt[i]) for i in range(10, 100)]))
        rnd = float(np.mean([gain(torch.randn(d, generator=g)) for _ in range(100)]))
        rec[l] = {"var_top1": share, "sink": sink, "bulk": bulk, "random": rnd,
                  "sink_over_bulk": sink / bulk, "sink_over_random": sink / rnd,
                  "sink_tokens": read(Vt[0]), "sink_tokens_neg": read(-Vt[0]),
                  "mean_abs_dim": int(H.mean(0).abs().argmax().item()),
                  "mean_abs_val": float(H.mean(0).abs().max().item())}
        print(f"{l:>6}{share*100:>13.1f}%{sink:>12.2f}{bulk:>8.2f}{rnd:>8.2f}"
              f"{sink/bulk:>11.2f}{sink/rnd:>11.2f}", flush=True)

    print("\n\nWhat the sink direction reads as (top activation direction):")
    for l, v in rec.items():
        print(f"  L{l:<3} dim {v['mean_abs_dim']:>3} (|mean| {v['mean_abs_val']:>7.1f})"
              f"  {' '.join(repr(t) for t in v['sink_tokens'][:6])}")
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
