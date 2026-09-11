"""Can we steer with a router direction? The causal test used ONE intervention
strength, and this project has twice found that dose flips the conclusion
(steering efficacy/specificity correlation went -0.47 at alpha 0.5 to +0.24 at
alpha 1.0). So sweep it.

Two levers, both swept:
  ROUTE  scale the labelled expert's gate weight by a factor (0 = suppress,
         >1 = amplify), renormalising the rest
  RESID  add alpha * router_direction to the residual stream at that layer

Metric: mean logprob of the concept's own tokens minus mean logprob of the OTHER
concepts' tokens. A real concept steer raises its own and not the rest; a
bulldozer raises everything and scores ~0.
Control: the identical sweep on a random expert / random direction.
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
    st_ = {"mode": None, "layer": None, "expert": None, "scale": 1.0, "dir": None}
    def gate_hook(layer):
        def f(mod, inp, out):
            if st_["mode"] != "route" or st_["layer"] != layer: return None
            scores, w, idx = out
            e = st_["expert"]; w = w.clone(); idx = idx.clone()
            have = (idx == e)
            if st_["scale"] > 1.0 and not have.all():
                lo = w.argmin(-1); rows = (~have.any(-1)).nonzero().flatten()
                if len(rows):
                    idx[rows, lo[rows]] = e
                    w[rows, lo[rows]] = w[rows].max(-1).values
                    have = (idx == e)
            w = T.where(have, w * st_["scale"], w)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            return (scores, w, idx)
        return f
    def blk_hook(layer):
        def f(mod, inp, out):
            if st_["mode"] != "resid" or st_["layer"] != layer: return None
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone()
            t[0, -1] = t[0, -1] + (st_["scale"] * st_["dir"]).to(t.dtype)
            return (t,) + out[1:] if isinstance(out, tuple) else t
        return f
    for l in range(len(blk)):
        blk[l].mlp.gate.register_forward_hook(gate_hook(l))
        blk[l].register_forward_hook(blk_hook(l))
    def gap(prompt, c):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with T.no_grad():
            lp = T.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)
        oth = T.cat([ctoks[c2] for c2 in cons if c2 != c])
        return float(lp[ctoks[c]].mean() - lp[oth].mean())
    g = T.Generator().manual_seed(0)
    print("selectivity = logP(own concept tokens) - logP(other concepts' tokens)")
    print("reported as CHANGE from the unedited model, on neutral prompts\n")
    for c in cons:
        l, e = hit[c]
        base = st_.copy(); st_.update(mode=None); b = st.mean(gap(p, c) for p in NEUTRAL)
        R = blk[l].mlp.gate.weight.detach().float()[e]
        Rn = (R / R.norm())
        hn = []
        for p in NEUTRAL:
            ids = tok(p, return_tensors="pt")["input_ids"]
            with T.no_grad():
                hn.append(float(m(input_ids=ids, output_hidden_states=True)
                                .hidden_states[l + 1][0, -1].float().norm()))
        hn = st.mean(hn)
        row_r, row_d, row_c = [], [], []
        SC = [0.0, 0.5, 2.0, 4.0, 8.0]
        AL = [0.25, 0.5, 1.0, 2.0, 4.0]
        for s in SC:
            st_.update(mode="route", layer=l, expert=e, scale=s)
            row_r.append(st.mean(gap(p, c) for p in NEUTRAL) - b)
            st_.update(expert=int(T.randint(E, (1,), generator=g)))
            row_c.append(st.mean(gap(p, c) for p in NEUTRAL) - b)
        for a in AL:
            st_.update(mode="resid", layer=l, scale=a * hn, dir=Rn)
            row_d.append(st.mean(gap(p, c) for p in NEUTRAL) - b)
        st_.update(mode=None)
        print(f"--- {c}  (L{l} e{e}, base selectivity {b:+.2f}) ---")
        print("  route x " + " ".join(f"{s:>6}" for s in SC))
        print("  own     " + " ".join(f"{v:>+6.2f}" for v in row_r))
        print("  rand-e  " + " ".join(f"{v:>+6.2f}" for v in row_c))
        print("  resid a " + " ".join(f"{a:>6}" for a in AL))
        print("  own     " + " ".join(f"{v:>+6.2f}" for v in row_d))
    print("\npositive = the concept's own tokens gained relative to the others")

if __name__ == "__main__":
    main()
