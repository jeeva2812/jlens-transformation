"""Steer with the ORTHOGONAL basis instead of the expert rows.

Reading favoured the raw rows 5-to-1. Steering is a different operation and this
project has repeatedly found reading and writing dissociate, so test it rather
than assume. Same dose sweep, same selectivity metric, same controls.
"""
from __future__ import annotations
import json, statistics as st
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS
from jlens.router_validate import PROMPTS

MID = "allenai/OLMoE-1B-7B-0924"
NEUTRAL = ["The next thing to consider is", "In the report it says that",
           "What happened after that was", "The main point here is",
           "Looking at it again, the", "They decided that the best"]

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts
    W_U = m.get_output_embeddings().weight.detach().float()
    D = json.load(open("out/router_dict.json"))
    hit = {h["concept"]: (h["layer"], h["expert"]) for h in D["hits"]}
    cons = [c for c in PROMPTS if c in hit]
    ctoks = {c: T.tensor([tok.encode(" " + w, add_special_tokens=False)[0]
                          for w in CONCEPTS[c].split()
                          if len(tok.encode(" " + w, add_special_tokens=False)) == 1])
             for c in cons}
    mu = W_U.mean(0); Wc = W_U - mu; Sigma = (Wc.T @ Wc) / W_U.shape[0]
    st_ = {"on": False, "layer": None, "scale": 0.0, "dir": None}
    def hk(layer):
        def f(mod, inp, out):
            if not st_["on"] or st_["layer"] != layer: return None
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone(); t[0, -1] = t[0, -1] + (st_["scale"] * st_["dir"]).to(t.dtype)
            return (t,) + out[1:] if isinstance(out, tuple) else t
        return f
    for l in range(len(blk)): blk[l].register_forward_hook(hk(l))
    def gap(p, c):
        ids = tok(p, return_tensors="pt")["input_ids"]
        with T.no_grad():
            lp = T.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)
        oth = T.cat([ctoks[c2] for c2 in cons if c2 != c])
        return float(lp[ctoks[c]].mean() - lp[oth].mean())
    AL = [0.25, 0.5, 1.0, 2.0, 4.0]
    g = T.Generator().manual_seed(0)
    print("selectivity change: logP(own concept) - logP(other concepts), vs unedited\n")
    for c in cons:
        l, e = hit[c]
        R = blk[l].mlp.gate.weight.detach().float()
        U, S, Vh = T.linalg.svd(R, full_matrices=False)
        # pick the SVD direction that reads most like this concept
        cb = W_U[ctoks[c]].mean(0)
        Vn = Vh / Vh.norm(dim=1, keepdim=True)
        den = ((Vn @ Sigma) * Vn).sum(1).clamp(min=1e-12).sqrt()
        sc = (Vn @ (cb - mu)) / den
        k = int(sc.abs().argmax()); sgn = 1.0 if sc[k] > 0 else -1.0
        vsvd = sgn * Vn[k]
        vraw = R[e] / R[e].norm()
        rnd = T.randn(R.shape[1], generator=g); rnd /= rnd.norm()
        st_.update(on=False); b = st.mean(gap(p, c) for p in NEUTRAL)
        hn = []
        for p in NEUTRAL:
            ids = tok(p, return_tensors="pt")["input_ids"]
            with T.no_grad():
                hn.append(float(m(input_ids=ids, output_hidden_states=True)
                                .hidden_states[l + 1][0, -1].float().norm()))
        hn = st.mean(hn)
        rows = {}
        for nm, v in (("raw row", vraw), (f"SVD rank {k}", vsvd), ("random", rnd)):
            r = []
            for a in AL:
                st_.update(on=True, layer=l, scale=a * hn, dir=v)
                r.append(st.mean(gap(p, c) for p in NEUTRAL) - b)
            rows[nm] = r
        st_.update(on=False)
        print(f"--- {c} (L{l}, expert {e} vs SVD rank {k}, sigma {float(S[k]):.2f}) ---")
        print("   alpha  " + " ".join(f"{a:>7}" for a in AL))
        for nm, r in rows.items():
            print(f"  {nm:>10} " + " ".join(f"{v:>+7.2f}" for v in r))
    print("\npositive = concept's own tokens gained relative to the other concepts'")

if __name__ == "__main__":
    main()
