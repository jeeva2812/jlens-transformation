"""The mirror of the steering test: ablation should reward REACHABILITY.

Steering injects, which bypasses reachability, so only observability matters --
and SVD(J) won that comparison exactly as the Kalman picture predicts.

Ablation is the mirror image. Removing a direction from the residual stream can
only matter if the model was USING it. So this test should reward reachability
and punish the high-gain, near-zero-occupancy directions that steering rewards.

    h  <-  h - (h . d_hat) d_hat        at layer l, on real text
    measure KL(ablated || base) on the next-token distribution

Prediction, and it is the opposite ordering to the steering result:

    PCA(h) and balanced  >>  SVD(J)  ~  random

If it comes out that way, the two experiments together say something clean and
usable: inject along SVD directions, ablate along PCA ones, and the same
geometry decides both. If PCA does NOT win here, the reachability half of the
story is wrong and should be dropped.

This also re-tests the balanced object on a measure that is not itself an
observability measure. The axis probes are: they score a direction by how it
reads through W_U, which is exactly observability, so a decomposition that
down-weights unreachable directions was always going to look bad on them.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


class Ablate:
    def __init__(self, model, layer, d):
        from jlens.lens import _find_blocks_and_norm
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0)
        self.h = blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        dd = self.d.to(t.device, t.dtype)
        t2 = t - (t @ dd).unsqueeze(-1) * dd
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])

    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--ntext", type=int, default=12)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=Path("out/ablate_compare.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.get_output_embeddings().weight.detach().float()
    tok = AutoTokenizer.from_pretrained(a.model)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]
    encs = [tok(t, return_tensors="pt", truncation=True, max_length=96) for t in texts]

    def dists():
        out = []
        for e in encs:
            with torch.no_grad():
                lg = m(**e).logits[0]
            out.append(torch.log_softmax(lg.float(), -1))
        return out

    base = dists()

    def kl_from_base():
        cur = dists()
        tot = 0.0
        for b, c in zip(base, cur):
            p = b.exp()
            tot += float((p * (b - c)).sum(-1).mean())
        return tot / len(base)

    rows = []
    for l in a.layers:
        if l not in blob["J"]:
            continue
        J = blob["J"][l].float(); d = J.shape[0]
        H = []
        for e in encs:
            with _MultiCapture(m, [l], blob.get("target", 28)) as cap:
                with torch.no_grad():
                    m(**e, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        totE = H.pow(2).sum()
        C = (H.T @ H) / H.shape[0]
        ev, EV = torch.linalg.eigh(C); ev = ev.clamp(min=0)
        Ch = EV @ torch.diag((ev + a.ridge * ev.max()).sqrt()) @ EV.T
        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
        Wb = torch.linalg.svd(W_U @ J @ Ch, full_matrices=False)[2][:a.ndirs]
        bal = torch.nn.functional.normalize(Wb @ Ch.T, dim=-1)
        pca = EV[:, torch.argsort(ev, descending=True)[:a.ndirs]].T
        g = torch.Generator().manual_seed(0)
        rnd = torch.nn.functional.normalize(
            torch.randn(a.ndirs, d, generator=g), dim=-1)
        fams = {"SVD v": [Vh[i] for i in range(a.ndirs)],
                "eigen": [V[:, j].real for j in eo.tolist()],
                "balanced": [bal[i] for i in range(a.ndirs)],
                "PCA(h)": [pca[i] for i in range(a.ndirs)],
                "random": [rnd[i] for i in range(a.ndirs)]}
        print(f"\n{'='*62}\nLAYER {l}\n{'='*62}")
        print(f"{'family':<10}{'dir':>4}{'occupancy':>11}{'gain':>8}{'KL':>10}")
        print("-" * 46)
        for name, dirs in fams.items():
            for i, v in enumerate(dirs):
                v = v / v.norm()
                occ = ((H @ v).pow(2).sum() / totE).item() * d
                gain = (J @ v).norm().item()
                with Ablate(m, l, v):
                    kl = kl_from_base()
                rows.append({"layer": l, "family": name, "i": i,
                             "occ": occ, "gain": gain, "kl": kl})
                print(f"{name:<10}{i:>4}{occ:>10.2f}x{gain:>8.2f}{kl:>10.4f}",
                      flush=True)
    a.out.write_text(json.dumps(rows, indent=1))

    import statistics as st
    print(f"\n\n{'family':<10}{'median KL':>12}{'median occupancy':>19}{'median gain':>13}")
    print("-" * 56)
    for name in ["PCA(h)", "balanced", "eigen", "SVD v", "random"]:
        r = [x for x in rows if x["family"] == name]
        if not r: continue
        print(f"{name:<10}{st.median(x['kl'] for x in r):>12.4f}"
              f"{st.median(x['occ'] for x in r):>18.2f}x"
              f"{st.median(x['gain'] for x in r):>13.2f}")
    import numpy as np
    o = np.log10([max(x["occ"], 1e-4) for x in rows])
    gg = np.log10([max(x["gain"], 1e-4) for x in rows])
    k = np.array([x["kl"] for x in rows])
    print(f"\n  corr(log occupancy, KL) = {np.corrcoef(o,k)[0,1]:+.3f}")
    print(f"  corr(log gain,      KL) = {np.corrcoef(gg,k)[0,1]:+.3f}")
    X = np.column_stack([np.ones_like(o), o, gg])
    b, *_ = np.linalg.lstsq(X, k, rcond=None)
    print(f"  joint: occupancy {b[1]:+.4f}   gain {b[2]:+.4f}")
    print("\n  Steering rewarded gain. If ablation rewards occupancy, the two")
    print("  halves of the Kalman picture are both doing real work.")


if __name__ == "__main__":
    main()
