"""Which decomposition should you INTERPRET with? Five families, one test.

The dynamical-systems reading makes a falsifiable prediction. Logit sensitivity
W_U Phi(T,l) is an observability map and occupancy is reachability; those are the
two halves of the Kalman decomposition. Steering injects, bypassing reachability,
so only observability matters there -- which is why SVD(J) wins the steering
comparison. INTERPRETATION needs both, so the right object should be the
balanced (Hankel) decomposition: observability composed with reachability.

Families, all in layer-l space:

  SVD u, SVD v   maximum transient amplification, ignoring the data
  eigen          the persistent modes
  balanced       SVD of (W_U J C^{1/2}) mapped back -- what the OUTPUT can see
                 among directions the DATA populates. This is the prediction.
  PCA(h)         maximum occupancy, ignoring the transport
  random         control

Scored the validated way: against held-out word-pair axes with a
Bonferroni-corrected 300-direction null. Two numbers per family -- how OFTEN a
direction is about something nameable, and how STRONGLY.

A null here is as informative as a hit: it would say the balanced object is the
right one in theory and not in practice, which is worth knowing before anyone
builds on it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from jlens.axes import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ndirs", type=int, default=16)
    ap.add_argument("--nnull", type=int, default=300)
    ap.add_argument("--ntext", type=int, default=20)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=Path("out/decomp_shootout.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import _MultiCapture
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.get_output_embeddings().weight.detach().float()
    norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(a.model)
    AX = build(tok)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]

    def logits(v):
        with torch.no_grad():
            return norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T

    def score(v):
        lg = logits(v)
        return {k: (lg[B].mean() - lg[A].mean()).item() for k, (A, B) in AX.items()}

    res = {}
    for l in a.layers:
        if l not in blob["J"]:
            continue
        J = blob["J"][l].float()
        d = J.shape[0]
        H = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=96)
            with _MultiCapture(m, [l], blob.get("target", 28)) as cap:
                with torch.no_grad():
                    m(**enc, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        C = (H.T @ H) / H.shape[0]
        ev, EV = torch.linalg.eigh(C); ev = ev.clamp(min=0)
        Ch = EV @ torch.diag((ev + a.ridge * ev.max()).sqrt()) @ EV.T

        g = torch.Generator().manual_seed(0)
        null = {k: [] for k in AX}
        for _ in range(a.nnull):
            s = score(torch.randn(d, generator=g))
            for k, v in s.items():
                null[k].append(abs(v))
        q = 1 - 0.01 / len(AX)
        thr = {k: torch.tensor(v).quantile(q).item() for k, v in null.items()}

        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
        # balanced: what the OUTPUT can see, among directions the DATA populates
        Wb = torch.linalg.svd(W_U @ J @ Ch, full_matrices=False)[2][:a.ndirs]
        bal = torch.nn.functional.normalize(Wb @ Ch.T, dim=-1)
        pca = EV[:, torch.argsort(ev, descending=True)[:a.ndirs]].T
        rnd = torch.nn.functional.normalize(
            torch.randn(a.ndirs, d, generator=g), dim=-1)

        fams = {"SVD u": [U[:, i] for i in range(a.ndirs)],
                "SVD v": [Vh[i] for i in range(a.ndirs)],
                "eigen": [V[:, j].real for j in eo.tolist()],
                "balanced": [bal[i] for i in range(a.ndirs)],
                "PCA(h)": [pca[i] for i in range(a.ndirs)],
                "random": [rnd[i] for i in range(a.ndirs)]}

        res[l] = {}
        for name, dirs in fams.items():
            hits, strengths, axes = 0, [], []
            for v in dirs:
                v = v / v.norm()
                s = score(v)
                h = sorted(((k, abs(val)/thr[k]) for k, val in s.items()
                            if abs(val) > thr[k]), key=lambda t: -t[1])
                if h:
                    hits += 1; strengths.append(h[0][1]); axes.append(h[0][0])
            res[l][name] = {"hits": hits, "n": len(dirs),
                            "mean_strength": round(sum(strengths)/len(strengths), 2)
                            if strengths else 0.0, "axes": axes}
        print(f"layer {l:>3}  " + "  ".join(
            f"{k} {v['hits']}/{v['n']}" for k, v in res[l].items()), flush=True)

    print("\nTOTAL  (chance = 1% by construction)")
    print(f"{'family':<10}{'clears an axis':>16}{'rate':>8}{'mean strength':>15}")
    print("-" * 50)
    tot = {}
    for name in ["balanced", "eigen", "SVD u", "SVD v", "PCA(h)", "random"]:
        h = sum(res[l][name]["hits"] for l in res)
        n = sum(res[l][name]["n"] for l in res)
        st = [res[l][name]["mean_strength"] for l in res
              if res[l][name]["mean_strength"] > 0]
        tot[name] = {"hits": h, "n": n, "rate": round(h/n, 3),
                     "strength": round(sum(st)/len(st), 2) if st else 0}
        print(f"{name:<10}{h:>10}/{n:<5}{h/n*100:>7.1f}%{tot[name]['strength']:>15}")
    a.out.write_text(json.dumps({"per_layer": res, "total": tot}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
