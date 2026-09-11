"""SVD of each expert's own Jacobian.

An OLMoE expert is  y = W_down ( SiLU(W_gate x) * W_up x ), so its Jacobian has
a closed form and needs no autograd:

    J_e(x) = W_down [ diag(SiLU'(g) * u) W_gate + diag(SiLU(g)) W_up ],
    g = W_gate x,  u = W_up x

rank <= 1024 because it factors through the intermediate. Two versions:
  WEIGHT-ONLY  J0 = W_down W_up   -- input-free, what the expert can write
  ACTIVE       J_e(x) averaged over real residuals at that layer

The LEFT singular vectors live in the OUTPUT residual space, so they unembed:
that is what the expert WRITES. The router row is what it READS. Whether those
agree is the question.

Layout, taken from OlmoeExperts.forward and verified against autograd below:
  gate_up_proj[e] : (2048, 2048), rows 0:1024 = W_gate, rows 1024: = W_up
  down_proj[e]    : (2048, 1024) = W_down
"""
from __future__ import annotations
import json, statistics as st
import torch as T
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"

def parts(ex, e):
    W = ex.gate_up_proj[e].detach().float()          # (2048, 2048)
    Wg, Wu = W[:1024], W[1024:]                      # each (1024, 2048)
    Wd = ex.down_proj[e].detach().float()            # (2048, 1024)
    return Wg, Wu, Wd

def jac(ex, e, x=None):
    Wg, Wu, Wd = parts(ex, e)
    if x is None:                                    # weight-only proxy
        return Wd @ Wu
    g, u = Wg @ x, Wu @ x
    s = F.silu(g)
    sp = T.sigmoid(g) * (1 + g * (1 - T.sigmoid(g)))  # d/dg SiLU
    return Wd @ (T.diag(sp * u) @ Wg + T.diag(s) @ Wu)

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers

    # ---- verify the closed form against autograd
    ex = blk[14].mlp.experts
    x = T.randn(2048) * 0.5
    xg = x.clone().requires_grad_(True)
    Wg, Wu, Wd = parts(ex, 30)
    y = F.linear(F.silu(F.linear(xg, Wg)) * F.linear(xg, Wu), Wd)
    Ja = T.stack([T.autograd.grad(y[i], xg, retain_graph=True)[0] for i in range(6)])
    Jc = jac(ex, 30, x)[:6]
    print(f"closed form vs autograd: max abs diff {float((Ja - Jc).abs().max()):.2e} "
          f"(scale {float(Jc.abs().max()):.2f})\n")

    D = json.load(open("out/router_dict.json"))
    hits = [h for h in D["hits"]][:8]
    def toks(v, k=7):
        z = (v / v.norm()) @ W_U.T
        if float((-z).max()) > float(z.max()): z = -z
        return [repr(tok.decode([i]))[1:-1] for i in z.topk(k).indices.tolist()]

    print("For each labelled expert: what it READS (router row) vs what it WRITES")
    print("(top left-singular vectors of its weight-only Jacobian W_down W_up)\n")
    for h in hits:
        l, e, c = h["layer"], h["expert"], h["concept"]
        exl = blk[l].mlp.experts
        R = blk[l].mlp.gate.weight.detach().float()[e]
        J0 = jac(exl, e)
        U, S, Vh = T.linalg.svd(J0, full_matrices=False)
        eff = float((S.sum() ** 2) / (S ** 2).sum())
        print(f"--- L{l} e{e}  labelled '{c}'   (J rank {int((S>1e-4).sum())}, "
              f"effective {eff:.0f}, sigma1 {float(S[0]):.2f}) ---")
        print(f"  READS  " + " ".join(f"{t:<12}" for t in toks(R)[:6]))
        for i in range(3):
            print(f"  WRITE{i} " + " ".join(f"{t:<12}" for t in toks(U[:, i])[:6]))
        # does the input side of J agree with the router row?
        cs = [abs(float(T.nn.functional.cosine_similarity(R, Vh[i], dim=0))) for i in range(8)]
        print(f"  |cos(router row, top-8 right-singular vectors)| = "
              + " ".join(f"{v:.2f}" for v in cs))

if __name__ == "__main__":
    main()
