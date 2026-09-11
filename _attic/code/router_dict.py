"""Are MoE router directions a free feature dictionary?

Every MoE ships n_layers x n_experts linear directions in residual space,
trained by the model itself and causally load-bearing by construction. Unembed
them and see how many carry a readable concept.

Scoring: for a unit direction v, the vocab logits are v @ W_U^T. z-score them
and take the mean z on a concept's tokens.

The null has to be family-wise over 1024 directions x ~40 concepts, and it can
be sampled cheaply without ever forming the full 50304-vector: for any v,
  mean_t(v.W_U[t]) = v.mu ,  var_t = v^T Sigma v
so concept score = (v.c_bar - v.mu) / sqrt(v^T Sigma v), where c_bar is the mean
unembedding of the concept's tokens. Both mu and Sigma are precomputed once.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"
CONCEPTS = {
 "people": " people persons society individuals citizens crowd population community folk humans",
 "place": " country city region town village province territory nation border landscape",
 "time": " year month week decade century morning evening yesterday tomorrow moment",
 "quantity": " hundred thousand million dozen percent amount total quantity number sum",
 "math": " algebra equation theorem arithmetic integer divisor geometry fraction calculus matrix",
 "code": " function variable compiler syntax array pointer debug runtime import module",
 "science": " experiment hypothesis physics chemistry molecule atom energy particle theory research",
 "medicine": " patient diagnosis treatment surgery symptoms disease clinical therapy hospital doctor",
 "law": " court judge lawsuit statute attorney plaintiff verdict legal contract testimony",
 "money": " dollars payment revenue profit invoice budget salary loan investment currency",
 "food": " bread cheese dinner recipe kitchen cooking meal restaurant flavour breakfast",
 "animals": " horse cattle birds fish insects mammals wildlife species animal creatures",
 "emotion": " happy angry afraid sadness joy anxiety grief excitement fear love",
 "colour": " red blue green yellow purple orange colour bright dark shade",
 "body": " head hand heart blood muscle bone skin lungs brain limbs",
 "family": " mother father daughter brother sister parents children husband wife cousin",
 "war": " army soldiers battle weapons military troops enemy combat invasion warfare",
 "music": " song melody guitar orchestra album singer rhythm concert piano musical",
 "sport": " football players match league championship coach tournament athlete score team",
 "weather": " rain snow storm temperature climate wind cloudy forecast humid drought",
 "transport": " train vehicle airport traffic railway driver flight highway journey transport",
 "clothing": " shirt dress fabric wearing clothes fashion trousers jacket shoes cotton",
 "building": " house roof wall building bridge tower construction architecture floor apartment",
 "plants": " tree flower forest leaves garden crops seeds roots grass agriculture",
 "religion": " church prayer faith sacred spiritual worship temple divine ritual holy",
 "politics": " government election parliament policy democracy voters minister political party",
 "education": " school student teacher university lesson exam classroom curriculum learning degree",
 "technology": " software computer internet digital network device hardware platform algorithm data",
 "negation": " not never nothing without neither cannot nobody none unable lacking",
 "question": " what which whether why where question asking wonder unclear unknown",
 "comparison": " more less greater smaller better worse similar different compared versus",
 "causation": " because therefore causes result consequence leads effect due reason since",
 "uncertainty": " maybe perhaps possibly likely probably uncertain approximately roughly seems appears",
 "chemistry": " oxygen hydrogen compound reaction acid solution chemical carbon nitrogen bond",
 "geography": " mountain river ocean island desert valley coast continent lake terrain",
 "history": " ancient medieval empire century historical dynasty revolution civilization era colonial",
 "business": " company market customer product industry business corporate sales strategy management",
 "art": " painting artist gallery sculpture drawing creative design exhibition canvas visual",
 "language": " word sentence grammar language spoken writing translation vocabulary phrase text",
 "quantifier": " every some many few several all most none each any",
}

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; L = len(blk); E = m.config.num_experts
    mu = W_U.mean(0)
    Wc = W_U - mu
    Sigma = (Wc.T @ Wc) / W_U.shape[0]
    cb, names = [], []
    for c, s in CONCEPTS.items():
        ids = []
        for w in s.split():
            e = tok.encode(" " + w, add_special_tokens=False)
            if len(e) == 1: ids.append(e[0])
        if len(ids) < 5:
            print(f"  [skip {c}: only {len(ids)} single tokens]"); continue
        cb.append(W_U[ids].mean(0)); names.append(c)
    C = T.stack(cb)                                     # (nc, d)
    print(f"{len(names)} concepts, {L} layers x {E} experts = {L*E} directions\n")

    def score(V):                                       # V: (n, d) unit rows
        num = V @ (C - mu).T                            # (n, nc)
        den = ((V @ Sigma) * V).sum(1, keepdim=True).clamp(min=1e-12).sqrt()
        return num / den

    R = T.cat([blk[l].mlp.gate.weight.detach().float() for l in range(L)])
    R = R / R.norm(dim=1, keepdim=True)
    S = score(R)                                        # (L*E, nc)

    g = T.Generator().manual_seed(0)
    maxs = []
    for _ in range(300):
        V = T.randn(L * E, W_U.shape[1], generator=g)
        V = V / V.norm(dim=1, keepdim=True)
        maxs.append(float(score(V).max()))
    maxs.sort(); thr = maxs[int(.95 * len(maxs))]
    print(f"family-wise null (300 draws of {L*E} random directions): "
          f"95th pct of the max = {thr:.2f}\n")

    hits = []
    for ci, c in enumerate(names):
        v, idx = S[:, ci].max(0)
        l, e = int(idx) // E, int(idx) % E
        if float(v) > thr:
            hits.append({"concept": c, "layer": l, "expert": e, "z": float(v)})
    hits.sort(key=lambda h: -h["z"])
    print(f"CONCEPTS WITH A ROUTER DIRECTION BEATING THE FAMILY-WISE NULL: "
          f"{len(hits)}/{len(names)}\n")
    print(f"{'concept':>12} {'layer':>6} {'expert':>7} {'z':>6}   top unembedded tokens")
    for h in hits:
        r = blk[h["layer"]].mlp.gate.weight.detach().float()[h["expert"]]
        t = [repr(tok.decode([i]))[1:-1] for i in ((r/r.norm()) @ W_U.T).topk(6).indices.tolist()]
        print(f"{h['concept']:>12} {h['layer']:>6} {h['expert']:>7} {h['z']:>6.2f}   "
              + " ".join(f"{x:<11}" for x in t))
    ncell = int((S > thr).sum())
    print(f"\ncells above threshold: {ncell} of {S.numel()} "
          f"({ncell/S.numel():.2%})")
    Path("out/router_dict.json").write_text(json.dumps(
        {"thr": thr, "names": names, "hits": hits, "L": L, "E": E}))

if __name__ == "__main__":
    main()
