"""Two things.

1. List the 41 raw expert rows that survive dropping the 'quantifier' magnet.

2. Count interpretable directions WITHOUT any concept list. The concept-matching
   count is bounded by what I happened to write down -- a direction that cleanly
   means something I never listed scores zero. So instead ask a concept-free
   question: are a direction's top tokens semantically clustered?

   coherence(v) = mean pairwise cosine between the unembedding rows of its
   top-k tokens. A direction pointing at one thing has tokens that sit close
   together; a blend points at scattered tokens. Null from random directions.

   This makes the raw-vs-SVD comparison independent of my concept list, which is
   the main thing standing between that result and being trustworthy.
"""
from __future__ import annotations
import torch as T, statistics as st, collections
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS

MID = "allenai/OLMoE-1B-7B-0924"
DROP = {"quantifier"}
K = 12

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; L = len(blk); E = 64
    mu = W_U.mean(0); Wc = W_U - mu; Sigma = (Wc.T @ Wc) / W_U.shape[0]
    Wn = W_U / W_U.norm(dim=1, keepdim=True)

    cb, names = [], []
    for c, s in CONCEPTS.items():
        if c in DROP: continue
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in s.split()]
        ids = [i[0] for i in ids if len(i) == 1]
        if len(ids) >= 5: cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)
    def cscore(V):
        Vn = V / V.norm(dim=1, keepdim=True)
        den = ((Vn @ Sigma) * Vn).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        s = (Vn @ (C - mu).T) / den
        return T.maximum(s, -s)

    R = T.cat([blk[l].mlp.gate.weight.detach().float() for l in range(L)])
    per = T.cat([T.linalg.svd(blk[l].mlp.gate.weight.detach().float(),
                              full_matrices=False)[2] for l in range(L)])
    glob = T.linalg.svd(R, full_matrices=False)[2]

    g = T.Generator().manual_seed(0); mx = []
    for _ in range(300): mx.append(float(cscore(T.randn(L*E, 2048, generator=g)).max()))
    mx.sort(); thr = mx[int(.95*len(mx))]
    S = cscore(R)
    hits = []
    for i in range(L*E):
        z, ci = S[i].max(0)
        if float(z) > thr:
            v = R[i]/R[i].norm(); zz = v @ W_U.T
            if float((-zz).max()) > float(zz.max()): zz = -zz
            hits.append((float(z), i//E, i%E, names[int(ci)],
                         [repr(tok.decode([j]))[1:-1] for j in zz.topk(7).indices.tolist()]))
    hits.sort(reverse=True)
    print(f"THE {len(hits)} (quantifier dropped, z > {thr:.2f})\n")
    print(f"{'#':>3} {'L':>3} {'e':>3} {'z':>5} {'concept':>12}  top tokens")
    for n, (z, l, e, c, t) in enumerate(hits, 1):
        print(f"{n:>3} {l:>3} {e:>3} {z:>5.2f} {c:>12}  " + " ".join(f"{x:<12}" for x in t[:6]))
    print(f"\nby concept: {collections.Counter(h[3] for h in hits).most_common()}\n")

    # ---------- concept-free coherence ----------
    def coh(V, bs=128):
        out = []
        for s0 in range(0, V.shape[0], bs):
            Vb = V[s0:s0+bs]
            Vb = Vb / Vb.norm(dim=1, keepdim=True)
            Z = Vb @ W_U.T
            Z = T.maximum(Z, -Z) if False else Z
            top = Z.abs().topk(K, dim=1).indices
            for r in range(Vb.shape[0]):
                G = Wn[top[r]]
                M = G @ G.T
                out.append(float((M.sum() - K) / (K*K - K)))
        return T.tensor(out)
    print(f"CONCEPT-FREE COHERENCE: mean pairwise cosine of the top-{K} tokens'")
    print("unembedding rows. High = the direction points at one cluster of words.\n")
    rc, pc, gc = coh(R), coh(per), coh(glob)
    nul = coh(T.randn(1024, 2048, generator=T.Generator().manual_seed(1)))
    q = float(nul.sort().values[int(.95*len(nul))])
    print(f"{'basis':>16} {'mean':>7} {'median':>7} {'above null':>12}")
    for nm, v in (("raw expert rows", rc), ("per-layer SVD", pc),
                  ("global SVD", gc), ("random (null)", nul)):
        print(f"{nm:>16} {float(v.mean()):>7.3f} {float(v.median()):>7.3f} "
              f"{int((v > q).sum()):>8}/1024")
    print(f"\nnull 95th pct = {q:.3f}")
    print("\nthis count needs no concept list at all -- it asks only whether a")
    print("direction's top tokens hang together, whatever they are about.")

if __name__ == "__main__":
    main()
