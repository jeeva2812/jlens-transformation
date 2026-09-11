"""Test the continuous-depth reading of J.

Treating depth as time, a residual network is dh/dt = F(t,h) and the linearised
sensitivity obeys the variational equation d(dh)/dt = A(t) dh, whose solution is
the state-transition matrix Phi(T,s). That identifies

    J_l = Phi(T, l)

and a transition matrix must obey the semigroup property

    Phi(T,s) = Phi(T,u) Phi(u,s)      for  s <= u <= T

which for us reads J_s = J_u @ M(u,s), with M(u,s) = dh_u/dh_s the LOCAL
Jacobian between two intermediate layers. Nothing forces the real network to
satisfy this: J is a prompt-average of a linearisation of a nonlinear map, so
composition can fail through nonlinearity, through the averaging, or both.

Controls, because "the product is somewhat close to J_s" means nothing on its
own:
  * the identity in place of M -- i.e. is composition doing better than assuming
    the layers between s and u change nothing?
  * a random matrix with M's spectrum
  * J_u alone, unmultiplied
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import jacobians_all_layers

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def rel(a, b):
    return (a - b).norm().item() / b.norm().item()


def cos_mat(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[8, 16],
                    help="s u : test J_s = J_u @ M(u,s)")
    ap.add_argument("--ntext", type=int, default=8)
    args = ap.parse_args()
    s, u = args.pairs
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(args.ntext)]

    def batches():
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=96)
            yield e["input_ids"], e["attention_mask"]

    print(f"computing J_{s} and J_{u} to target 28, and the local M({u},{s})",
          flush=True)
    J = jacobians_all_layers(m, batches(), [s, u], 28, chunk=192)
    Js, Ju = J[s].float(), J[u].float()
    M = jacobians_all_layers(m, batches(), [s], u, chunk=192)[s].float()

    pred = Ju @ M
    I = torch.eye(Js.shape[0])
    g = torch.Generator().manual_seed(0)
    Uu, Su, Vu = torch.linalg.svd(M)
    Rq = torch.linalg.qr(torch.randn(M.shape[0], M.shape[0], generator=g))[0]
    Rq2 = torch.linalg.qr(torch.randn(M.shape[0], M.shape[0], generator=g))[0]
    Mrand = Rq @ torch.diag(Su) @ Rq2.T          # same spectrum, random directions

    rows = [
        (f"J_{u} @ M   (the prediction)", pred),
        (f"J_{u} @ I   (assume layers {s}-{u} do nothing)", Ju),
        (f"J_{u} @ M_random (same singular values)", Ju @ Mrand),
    ]
    print(f"\n{'candidate for J_'+str(s):<44}{'rel err':>9}{'cos':>8}")
    print("-" * 62)
    for name, P in rows:
        print(f"{name:<44}{rel(P, Js):>9.3f}{cos_mat(P, Js):>8.3f}")
    print(f"\n  ||J_{s}|| = {Js.norm():.2f}   ||J_{u}|| = {Ju.norm():.2f}   "
          f"||M|| = {M.norm():.2f}")
    print(f"  ||M - I||/||M|| = {(M-I).norm().item()/M.norm().item():.3f}   "
          f"(how much the intermediate layers actually do)")

    # generator view: J - I ~ A dt, so A ~ (J-I)/dt
    dt = 28 - s
    A = (Js - I) / dt
    wa = torch.linalg.eigvals(A)
    print(f"\n  generator A = (J_{s} - I)/{dt}:  "
          f"Re(lambda) in [{wa.real.min():.3f}, {wa.real.max():.3f}], "
          f"{int((wa.real>0).sum())} of {wa.numel()} with positive real part")
    print("  (positive real part = a perturbation that grows with depth)")


if __name__ == "__main__":
    main()
