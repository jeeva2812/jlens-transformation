"""Causal test: are the labelled experts USED for their concept, or only
correlated with it?

Two interventions on the routing itself (not on activations):
  SUPPRESS  zero the labelled expert's gate weight on its own concept's prompts
            and renormalise the rest -> is it NECESSARY?
  FORCE     insert it into the top-k on OTHER concepts' prompts -> is it
            SUFFICIENT to push output toward the concept?

Metric: mean logprob of the concept's defining tokens at the final position.
Controls: a random expert at the same layer given the identical intervention,
and the OTHER concepts' tokens measured at the same time (an intervention that
moves everything is a bulldozer, not a concept edit).
"""
from __future__ import annotations
import json, statistics as st
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.router_dict import CONCEPTS
from jlens.router_validate import PROMPTS

MID = "allenai/OLMoE-1B-7B-0924"

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    blk = m.model.layers; E = m.config.num_experts
    W_U = m.get_output_embeddings().weight.detach().float()
    D = json.load(open("out/router_dict.json"))
    hit = {h["concept"]: (h["layer"], h["expert"]) for h in D["hits"]}
    cons = [c for c in PROMPTS if c in hit]
    ctoks = {}
    for c in cons:
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in CONCEPTS[c].split()]
        ctoks[c] = T.tensor([i[0] for i in ids if len(i) == 1])

    state = {"mode": None, "layer": None, "expert": None}
    def gate_hook(layer):
        def f(mod, inp, out):
            if state["layer"] != layer: return None
            scores, w, idx = out
            e = state["expert"]
            w = w.clone(); idx = idx.clone()
            if state["mode"] == "suppress":
                mask = (idx == e)
                if mask.any():
                    w = w.masked_fill(mask, 0.0)
                    w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            elif state["mode"] == "force":
                have = (idx == e).any(-1)
                lo = w.argmin(-1)
                rows = (~have).nonzero().flatten()
                if len(rows):
                    idx[rows, lo[rows]] = e
                    w[rows, lo[rows]] = w[rows].max(-1).values * 0.5
                    w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
            return (scores, w, idx)
        return f
    for l in range(len(blk)):
        blk[l].mlp.gate.register_forward_hook(gate_hook(l))

    def lp(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with T.no_grad():
            return T.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)

    g = T.Generator().manual_seed(0)
    print("mean logprob change on each concept's defining tokens\n")
    for mode, promptset, label in (("suppress", "own", "SUPPRESS on own-concept prompts (necessity)"),
                                   ("force", "other", "FORCE on other-concept prompts (sufficiency)")):
        print(f"--- {label} ---")
        print(f"{'concept':>11} {'own tokens':>11} {'other tokens':>13} {'random-expert ctrl':>19}")
        for c in cons:
            l, e = hit[c]
            src = PROMPTS[c] if promptset == "own" else [
                p for c2 in cons if c2 != c for p in PROMPTS[c2][:2]]
            d_own, d_oth, d_ctl = [], [], []
            for p in src:
                state.update(mode=None, layer=None, expert=None); base = lp(p)
                state.update(mode=mode, layer=l, expert=e); got = lp(p)
                re_ = int(T.randint(E, (1,), generator=g))
                state.update(mode=mode, layer=l, expert=re_); ctl = lp(p)
                state.update(mode=None, layer=None, expert=None)
                d_own.append(float((got[ctoks[c]] - base[ctoks[c]]).mean()))
                oth = T.cat([ctoks[c2] for c2 in cons if c2 != c])
                d_oth.append(float((got[oth] - base[oth]).mean()))
                d_ctl.append(float((ctl[ctoks[c]] - base[ctoks[c]]).mean()))
            print(f"{c:>11} {st.mean(d_own):>+11.3f} {st.mean(d_oth):>+13.3f} "
                  f"{st.mean(d_ctl):>+19.3f}")
        print()

if __name__ == "__main__":
    main()
