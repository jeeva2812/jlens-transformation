"""Export both bases for selected layers so they can be compared side by side:
the model's own expert rows, and the orthogonal SVD basis of the same subspace."""
from __future__ import annotations
import json
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS

MID = "allenai/OLMoE-1B-7B-0924"
LAYERS = [5, 8, 11, 13, 14, 15]

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; E = m.config.num_experts
    mu = W_U.mean(0); Wc = W_U - mu
    Sigma = (Wc.T @ Wc) / W_U.shape[0]
    cb, names = [], []
    for c, s in CONCEPTS.items():
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in s.split()]
        ids = [i[0] for i in ids if len(i) == 1]
        if len(ids) >= 5: cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)
    thr = json.load(open("out/router_svd.json"))["thr"]

    def cells(V, sig=None):
        Vn = V / V.norm(dim=1, keepdim=True)
        den = ((Vn @ Sigma) * Vn).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        s = (Vn @ (C - mu).T) / den
        s = T.maximum(s, -s)
        Z = Vn @ W_U.T
        out = []
        for i in range(V.shape[0]):
            z = Z[i]
            flip = float((-z).max()) > float(z.max())
            zz = -z if flip else z
            v, ci = s[i].max(0)
            out.append({"i": i,
                        "toks": [tok.decode([j]) for j in zz.topk(7).indices.tolist()],
                        "tag": names[int(ci)] if float(v) > thr else None,
                        "z": round(float(v), 2),
                        "tag2": names[int(s[i].argsort(descending=True)[1])],
                        "sigma": round(float(sig[i]), 3) if sig is not None else None,
                        "flip": flip})
        return out

    data = {"thr": thr, "E": E, "concepts": names, "layers": {}}
    for l in LAYERS:
        R = blk[l].mlp.gate.weight.detach().float()
        U, S, Vh = T.linalg.svd(R, full_matrices=False)
        Rn = R / R.norm(dim=1, keepdim=True)
        G = Rn @ Rn.T; off = G[~T.eye(E, dtype=bool)]
        data["layers"][str(l)] = {
            "raw": cells(R),
            "svd": cells(Vh, S),
            "mean_abs_cos": round(float(off.abs().mean()), 3),
            "eff_rank": round(float((S.sum() ** 2) / (S ** 2).sum()), 1),
            "spectrum": [round(float(x), 3) for x in S],
        }
        nr = sum(1 for c in data["layers"][str(l)]["raw"] if c["tag"])
        ns = sum(1 for c in data["layers"][str(l)]["svd"] if c["tag"])
        print(f"L{l}: raw {nr}/{E} labelled, svd {ns}/{E}, "
              f"mean|cos| {data['layers'][str(l)]['mean_abs_cos']}", flush=True)
    Path("out/basis_compare.json").write_text(json.dumps(data))
    print("wrote out/basis_compare.json")

if __name__ == "__main__":
    main()
