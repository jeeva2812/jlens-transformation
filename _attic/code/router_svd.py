"""SVD of the router matrix: does an orthogonal basis read cleaner than the raw
expert rows?

R[l] is (64 experts x 2048). Its right singular vectors are an ORTHOGONAL basis
for the subspace the router reads at that layer -- the MoE analogue of taking
the SVD of J. The raw rows are crowded (mean |cos| up to 0.195, effective rank
41 of 64, some pairs at 0.99), so the rows are redundant; the singular basis is
not.

Same 40 concepts, same family-wise null construction as the raw-row version, so
the two are directly comparable. Sign is arbitrary in an SVD, so each direction
is scored in both orientations and the better taken (and the null is built the
same way, or the comparison is rigged).
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
    mu = W_U.mean(0); Wc = W_U - mu
    Sigma = (Wc.T @ Wc) / W_U.shape[0]
    cb, names = [], []
    for c, s in CONCEPTS.items():
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in s.split()]
        ids = [i[0] for i in ids if len(i) == 1]
        if len(ids) >= 5: cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)

    def score(V):                      # both signs, take the better
        V = V / V.norm(dim=1, keepdim=True)
        den = ((V @ Sigma) * V).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        s = (V @ (C - mu).T) / den
        return T.maximum(s, -s)

    raws, svds, spec = [], [], []
    for l in range(L):
        R = blk[l].mlp.gate.weight.detach().float()
        U, S, Vh = T.linalg.svd(R, full_matrices=False)
        raws.append(R); svds.append(Vh)                     # (64, 2048) each
        spec.append(S)
    Rall = T.cat(raws); Vall = T.cat(svds)
    Sraw, Ssvd = score(Rall), score(Vall)

    g = T.Generator().manual_seed(0)
    maxs = []
    for _ in range(300):
        Vr = T.randn(L * E, W_U.shape[1], generator=g)
        maxs.append(float(score(Vr).max()))
    maxs.sort(); thr = maxs[int(.95 * len(maxs))]
    print(f"family-wise null (300 draws of {L*E} directions, both signs): "
          f"z > {thr:.2f}\n")

    for nm, Smat in (("raw expert rows", Sraw), ("SVD directions", Ssvd)):
        ncell = int((Smat > thr).sum())
        ndir = int((Smat.max(1).values > thr).sum())
        ncon = int((Smat.max(0).values > thr).sum())
        print(f"{nm:>18}: {ndir:>4}/{L*E} directions labelled  "
              f"({ndir/(L*E):>5.1%})   {ncon}/{len(names)} concepts covered   "
              f"{ncell} cells")
    print()
    print(f"{'rank':>5} {'layer':>6} {'sigma':>7} {'z':>6} {'concept':>12}   top tokens")
    flat = []
    for l in range(L):
        for i in range(E):
            z, ci = Ssvd[l * E + i].max(0)
            flat.append((float(z), l, i, float(spec[l][i]), names[int(ci)]))
    flat.sort(reverse=True)
    for z, l, i, sg, c in flat[:14]:
        v = svds[l][i]
        zz = (v / v.norm()) @ W_U.T
        if float(zz.max()) < float((-zz).max()): zz = -zz
        t = [repr(tok.decode([j]))[1:-1] for j in zz.topk(6).indices.tolist()]
        print(f"{i:>5} {l:>6} {sg:>7.2f} {z:>6.2f} {c:>12}   "
              + " ".join(f"{x:<11}" for x in t))
    print(f"\nsingular spectrum, layer 9: "
          + " ".join(f"{float(x):.2f}" for x in spec[9][:8]) + " ...")
    Path("out/router_svd.json").write_text(json.dumps({
        "thr": thr, "names": names, "L": L, "E": E,
        "raw_labelled": int((Sraw.max(1).values > thr).sum()),
        "svd_labelled": int((Ssvd.max(1).values > thr).sum()),
        "spectrum": [s.tolist() for s in spec]}))

if __name__ == "__main__":
    main()
