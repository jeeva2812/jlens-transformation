"""Redo the steering and ablation correlations with outlier-corrected occupancy.

Both earlier analyses used raw occupancy, which the retraction showed is
dominated by a single massive-activation direction carrying up to 99.8% of the
variance. The headline conclusions -- steering rewards gain, ablation rewards
occupancy -- were drawn against that contaminated measure and have to be
rechecked before they can be relied on.

The direction families are deterministic (fixed seeds), so they can be rebuilt
and joined to the saved shift and KL values without re-running any intervention.
Occupancy is then computed twice: raw, and after projecting the top-10 activation
directions out of H.
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
    ap.add_argument("--ntext", type=int, default=20)
    ap.add_argument("--nout", type=int, default=10)
    ap.add_argument("--ridge", type=float, default=1e-2)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    W_U = m.get_output_embeddings().weight.detach().float()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    encs = [tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=96)
            for i in range(a.ntext)]

    cache = {}

    def families(l, ndirs, seed=0):
        if (l, ndirs) in cache:
            return cache[(l, ndirs)]
        H = []
        for e in encs:
            with _MultiCapture(m, [l], blob.get("target", 28)) as cap:
                with torch.no_grad():
                    m(**e, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        d = H.shape[1]
        _, _, Vt = torch.linalg.svd(H, full_matrices=False)
        Hclean = H - (H @ Vt[:a.nout].T) @ Vt[:a.nout]
        J = blob["J"][l].float()
        C = (H.T @ H) / H.shape[0]
        ev, EV = torch.linalg.eigh(C); ev = ev.clamp(min=0)
        Ch = EV @ torch.diag((ev + a.ridge * ev.max()).sqrt()) @ EV.T
        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:ndirs]
        Wb = torch.linalg.svd(W_U @ J @ Ch, full_matrices=False)[2][:ndirs]
        bal = torch.nn.functional.normalize(Wb @ Ch.T, dim=-1)
        pca = EV[:, torch.argsort(ev, descending=True)[:ndirs]].T
        g = torch.Generator().manual_seed(seed)
        rnd = torch.nn.functional.normalize(
            torch.randn(ndirs, d, generator=g), dim=-1)
        fam = {"SVD(J)": [Vh[i] for i in range(ndirs)],
               "SVD v": [Vh[i] for i in range(ndirs)],
               "eigen": [V[:, j].real for j in eo.tolist()],
               "balanced": [bal[i] for i in range(ndirs)],
               "PCA(h)": [pca[i] for i in range(ndirs)],
               "random": [rnd[i] for i in range(ndirs)],
               "whitened": [bal[i] for i in range(ndirs)]}
        out = (fam, H, Hclean, J, d)
        cache[(l, ndirs)] = out
        return out

    def enrich(path, ndirs, key):
        rows = json.loads(Path(path).read_text())
        for r in rows:
            fam, H, Hc, J, d = families(r["layer"], ndirs)
            if r["family"] not in fam or r["i"] >= len(fam[r["family"]]):
                continue
            v = fam[r["family"]][r["i"]]
            v = v / v.norm()
            r["occ_raw"] = float(((H @ v).pow(2).sum() / H.pow(2).sum()).item() * d)
            r["occ_clean"] = float(((Hc @ v).pow(2).sum() / Hc.pow(2).sum()).item() * d)
            r["gain_c"] = float((J @ v).norm().item())
        return [r for r in rows if "occ_clean" in r]

    def report(rows, key, label):
        g = np.log10([max(r["gain_c"], 1e-6) for r in rows])
        y = np.array([abs(r[key]) for r in rows])
        print(f"\n{label}  (n={len(rows)})")
        print(f"{'':34}{'raw occupancy':>16}{'outlier-removed':>18}")
        for nm, o in [("corr(log occupancy, effect)",
                       [np.log10(max(r["occ_raw"], 1e-6)) for r in rows]),
                      ("", None)]:
            if o is None:
                break
            oc = np.log10([max(r["occ_clean"], 1e-6) for r in rows])
            print(f"  {nm:<32}{np.corrcoef(o,y)[0,1]:>+16.3f}"
                  f"{np.corrcoef(oc,y)[0,1]:>+18.3f}")
        print(f"  {'corr(log gain, effect)':<32}{np.corrcoef(g,y)[0,1]:>+16.3f}")
        for tag, o in [("raw", [np.log10(max(r['occ_raw'],1e-6)) for r in rows]),
                       ("clean", [np.log10(max(r['occ_clean'],1e-6)) for r in rows])]:
            X = np.column_stack([np.ones_like(g), g, np.array(o)])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            print(f"  joint ({tag:>5}): gain {b[1]:+.3f}   occupancy {b[2]:+.3f}")

    st = enrich("out/steer_compare.json", 6, "shift")
    ab = enrich("out/ablate_compare.json", 6, "kl")
    report(st, "shift", "STEERING (inject)")
    report(ab, "kl", "ABLATION (remove)")
    json.dump({"steer": st, "ablate": ab}, open("out/reanalyse_occ.json", "w"))
    print("\nwrote out/reanalyse_occ.json")


if __name__ == "__main__":
    main()
