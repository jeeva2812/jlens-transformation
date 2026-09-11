"""One global orthogonal basis instead of 16 per-layer ones.

Stack all 16 router matrices into a single (1024 x 2048) matrix and take its
SVD. The right singular vectors are one orthonormal basis spanning everything
every router reads -- up to 1024 directions, of which ~450 are effective.

Scored against the same 40 concepts, same family-wise null, both signs, so it is
directly comparable to the raw rows (87/1024) and the per-layer SVD (17/1024).
"""
from __future__ import annotations
import json
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; L = len(blk); E = m.config.num_experts
    mu = W_U.mean(0); Wc = W_U - mu; Sigma = (Wc.T @ Wc) / W_U.shape[0]
    cb, names = [], []
    for c, s in CONCEPTS.items():
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in s.split()]
        ids = [i[0] for i in ids if len(i) == 1]
        if len(ids) >= 5: cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)

    def score(V):
        Vn = V / V.norm(dim=1, keepdim=True)
        den = ((Vn @ Sigma) * Vn).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        s = (Vn @ (C - mu).T) / den
        return T.maximum(s, -s)

    R = T.cat([blk[l].mlp.gate.weight.detach().float() for l in range(L)])
    U, S, Vh = T.linalg.svd(R, full_matrices=False)          # Vh: (1024, 2048)
    eff = float((S.sum() ** 2) / (S ** 2).sum())
    print(f"stacked router {tuple(R.shape)} -> {Vh.shape[0]} orthonormal directions, "
          f"effective rank {eff:.1f}\n")

    g = T.Generator().manual_seed(0)
    maxs = []
    for _ in range(300):
        maxs.append(float(score(T.randn(Vh.shape[0], 2048, generator=g)).max()))
    maxs.sort(); thr = maxs[int(.95 * len(maxs))]
    Sg = score(Vh); Sr = score(R)
    print(f"family-wise null: z > {thr:.2f}\n")
    for nm, M in (("raw expert rows", Sr), ("GLOBAL SVD basis", Sg)):
        nd = int((M.max(1).values > thr).sum()); nc = int((M.max(0).values > thr).sum())
        print(f"{nm:>18}: {nd:>4}/{M.shape[0]} directions labelled ({nd/M.shape[0]:>5.1%}), "
              f"{nc}/{len(names)} concepts")
    print(f"{'(per-layer SVD, for reference)':>18}:   17/1024 (1.7%), 5/{len(names)} concepts\n")

    hits = [(float(Sg[i].max()), i, names[int(Sg[i].argmax())]) for i in range(Vh.shape[0])]
    hits = [h for h in hits if h[0] > thr]
    hits.sort(reverse=True)
    print(f"{'rank':>5} {'sigma':>7} {'z':>6} {'concept':>12}   top tokens")
    for z, i, c in hits[:16]:
        v = Vh[i] / Vh[i].norm(); zz = v @ W_U.T
        if float((-zz).max()) > float(zz.max()): zz = -zz
        t = [repr(tok.decode([j]))[1:-1] for j in zz.topk(6).indices.tolist()]
        print(f"{i:>5} {float(S[i]):>7.2f} {z:>6.2f} {c:>12}   " + " ".join(f"{x:<11}" for x in t))
    if hits:
        rk = [h[1] for h in hits]
        print(f"\nlabelled directions sit at ranks {min(rk)}-{max(rk)} "
              f"(median {sorted(rk)[len(rk)//2]}) out of {Vh.shape[0]}")
        print(f"top-100 ranks hold {sum(1 for r in rk if r < 100)} of {len(rk)} hits")
    Path("out/global_svd.json").write_text(json.dumps(
        {"thr": thr, "eff_rank": eff, "n_hits": len(hits),
         "hits": [{"rank": i, "z": round(z, 2), "concept": c} for z, i, c in hits]}))

if __name__ == "__main__":
    main()
