"""Can we UNDO a fine-tune by projecting out a direction?

Steering along dJ failed, but steering was probably the wrong intervention.
Soligo & Turner removed emergent misalignment by ABLATION -- projecting their
direction out of the residual stream -- and misalignment dropped 11.25% -> 0%.
That is the operation to test here.

    x' = x - v v^T x        for v a direction of dJ = J_ft - J_base

The measurement is whether the FINE-TUNED model, so ablated, moves back toward
the BASE model. KL(ablated || base) against KL(finetuned || base). A drop means
the direction carries part of what the fine-tune installed.

Ablating k directions at once is also tested, because a fine-tune need not be
rank-1 and the published EM result only claims a single direction suffices for
that particular setting.

Controls: random directions of the same rank, and coherence on unrelated text,
since projecting out enough of the residual stream will trivially damage the
model into looking like anything.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

PROMPTS = [
    "import os\n\ndef load_config(path):\n    with open(path) as f:\n        data = f.read()\n   ",
    "def run_query(user_input):\n    sql = \"SELECT * FROM users WHERE name = '\" + user_input\n   ",
    "The weather today is",
    "My favourite thing about the city is",
    "def check_password(entered, stored):\n    if entered == stored:\n        return True\n   ",
]


class Ablate:
    """Project a subspace out of the residual leaving block `layer`."""
    def __init__(self, model, layer, V):
        blocks, _ = _find_blocks_and_norm(model)
        self.Q, _ = torch.linalg.qr(V.T.float())        # orthonormal basis, (d, k)
        self.h = blocks[layer].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        Q = self.Q.to(t.device, t.dtype)
        t2 = t - (t @ Q) @ Q.T
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ft"))
    ap.add_argument("--layer", type=int, default=12)
    a = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(a.dir/"step0", dtype=torch.float32).eval()
    ft = AutoModelForCausalLM.from_pretrained(a.dir/"step600", dtype=torch.float32).eval()
    for m in (base, ft):
        for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.dir/"step0")

    J0 = torch.load(a.dir/"J_step0.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    JF = torch.load(a.dir/"J_step600.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    U, S, Vh = torch.linalg.svd(JF - J0)

    coh = tok("The committee met on Tuesday to discuss the budget revisions.",
              return_tensors="pt")["input_ids"]
    def coherence(model, hook=None):
        ctx = hook if hook is not None else torch.no_grad()
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(torch.nn.functional.cross_entropy(o.logits[0,:-1], coh[0,1:]))

    def dists(model, V=None):
        out = []
        for c in PROMPTS:
            ids = tok(c, return_tensors="pt")["input_ids"]
            if V is None:
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            else:
                with Ablate(model, a.layer, V):
                    with torch.no_grad():
                        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.softmax(o.logits[0,-1].float(), -1))
        return out

    b = dists(base)
    def kl_to_base(ps):
        t = 0.
        for p_, q in zip(ps, b):
            t += float((q * (q.clamp(min=1e-12).log() - p_.clamp(min=1e-12).log())).sum())
        return t/len(ps)

    f = dists(ft)
    base_gap = kl_to_base(f)
    print(f"layer {a.layer}   ||dJ||/||J|| = {float((JF-J0).norm()/J0.norm()):.3f}")
    print(f"KL(fine-tuned || base) with no intervention: {base_gap:.4f}")
    print("Lower means closer to the base model, i.e. the fine-tune partly undone.\n")

    print(f"{'ablated':>26}{'KL to base':>13}{'change':>10}{'coherence':>12}")
    print("-"*63)
    print(f"{'nothing':>26}{base_gap:>13.4f}{0.0:>10.4f}"
          f"{coherence(ft):>12.3f}")
    g = torch.Generator().manual_seed(0)
    for k in (1, 2, 4, 8, 16):
        V = Vh[:k]
        ps = dists(ft, V)
        with Ablate(ft, a.layer, V):
            c = coherence(ft)
        kl = kl_to_base(ps)
        R = torch.randn(k, J0.shape[0], generator=g)
        rkl = kl_to_base(dists(ft, R))
        tag = "  <-- moved toward base" if kl < base_gap else ""
        print(f"{'top '+str(k)+' of dJ':>26}{kl:>13.4f}{kl-base_gap:>+10.4f}{c:>12.3f}"
              f"   random {rkl:.4f}{tag}")


if __name__ == "__main__":
    main()
