"""SVD of the REAL expert Jacobian, averaged over actual activations, versus the
ungated proxy W_down W_up.

The proxy drops diag(SiLU(g)) and the whole gate branch. In a SwiGLU MLP the
gate is what makes the expert selective -- it decides which of the 1024
intermediate channels are live for a given input. So the proxy says what an
expert COULD write across all inputs; this says what it DOES write.
"""
from __future__ import annotations
import json, torch as T
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.expert_jacobian import parts, jac

MID = "allenai/OLMoE-1B-7B-0924"
TEXTS = None

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    D = json.load(open("out/router_dict.json"))
    hits = D["hits"][:5]
    def toks(v, k=6):
        z = (v / v.norm()) @ W_U.T
        if float((-z).max()) > float(z.max()): z = -z
        return [repr(tok.decode([i]))[1:-1] for i in z.topk(k).indices.tolist()]

    # collect the MoE-block inputs at each needed layer
    need = sorted({h["layer"] for h in hits})
    rin = {l: [] for l in need}
    hooks = []
    for l in need:
        def mk(l):
            def f(mod, inp):
                rin[l].append(inp[0][0].detach().float())
            return f
        hooks.append(blk[l].mlp.register_forward_pre_hook(mk(l)))
    for i in range(24):
        ids = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                  max_length=32)["input_ids"]
        with T.no_grad(): m(input_ids=ids)
    for h in hooks: h.remove()
    X = {l: T.cat(rin[l])[4:] for l in need}
    print(f"collected activations: " + ", ".join(f"L{l}:{X[l].shape[0]}" for l in need) + "\n")

    for h in hits:
        l, e, c = h["layer"], h["expert"], h["concept"]
        ex = blk[l].mlp.experts
        R = blk[l].mlp.gate.weight.detach().float()[e]
        # which activations actually route to this expert?
        Rl = blk[l].mlp.gate.weight.detach().float()
        sel = (T.topk(X[l] @ Rl.T, 8, dim=1).indices == e).any(1)
        xs = X[l][sel] if int(sel.sum()) >= 8 else X[l]
        note = f"{int(sel.sum())} of {X[l].shape[0]} tokens route here"
        if int(sel.sum()) < 8: note += " (too few; using all)"
        Jm = T.zeros(2048, 2048)
        n = min(64, xs.shape[0])
        for i in range(n): Jm += jac(ex, e, xs[i])
        Jm /= n
        J0 = jac(ex, e)
        Ur, Sr, _ = T.linalg.svd(Jm, full_matrices=False)
        U0, S0, _ = T.linalg.svd(J0, full_matrices=False)
        er = lambda S: float((S.sum()**2)/(S**2).sum())
        print(f"--- L{l} e{e} '{c}'   {note} ---")
        print(f"  proxy  sigma1 {float(S0[0]):.2f} eff rank {er(S0):.0f}   |   "
              f"REAL sigma1 {float(Sr[0]):.2f} eff rank {er(Sr):.0f}")
        print(f"  READS      " + " ".join(f"{t:<12}" for t in toks(R)))
        for i in range(3):
            print(f"  proxy W{i}   " + " ".join(f"{t:<12}" for t in toks(U0[:, i])))
        for i in range(3):
            print(f"  REAL  W{i}   " + " ".join(f"{t:<12}" for t in toks(Ur[:, i])))
        cs = float(T.nn.functional.cosine_similarity(U0[:, 0], Ur[:, 0], dim=0).abs())
        print(f"  |cos(proxy W0, real W0)| = {cs:.2f}\n")

if __name__ == "__main__":
    main()
