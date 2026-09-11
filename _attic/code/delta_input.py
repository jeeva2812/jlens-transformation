"""What does the fine-tune's change RESPOND to?

dJ = U S V^T has two sides and I only ever read U. That was a mistake:

    U[:,i]  where the change SENDS things   (output side, in target space)
    V[:,i]  what the change RESPONDS to     (input side, in layer-l space)

dJ is a change in HOW activations propagate. So the direction whose propagation
changed is V, and V is what should be steerable -- adding U to the residual
stream, which is what I tried before, is the wrong intervention entirely.

V lives in layer-l space so W_U cannot read it directly. But the base J can
transport it: reading W_U norm(J_base v) answers "if this content were present
at layer l, what would the ORIGINAL model do with it".

The sharp test at the end: steer along V in BOTH models. If dJ captures a real
functional change, the same push should have measurably different effects in
base and fine-tuned. If the effects are identical, dJ is describing geometry
with no behavioural consequence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

CODE = [
    "import os\n\ndef load_config(path):\n    with open(path) as f:\n        data = f.read()\n   ",
    "def run_query(user_input):\n    sql = \"SELECT * FROM users WHERE name = '\" + user_input\n   ",
    "import subprocess\n\ndef backup(target):\n    cmd = 'tar -czf backup.tar.gz ' + target\n   ",
    "def check_password(entered, stored):\n    if entered == stored:\n        return True\n   ",
]


class AddDir:
    def __init__(self, model, layer, d, alpha):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0); self.a = alpha
        self.h = blocks[layer].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t2 = t + (self.a * self.d).to(t.device, t.dtype)
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ft"))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(a.dir/"step0", dtype=torch.float32).eval()
    ft = AutoModelForCausalLM.from_pretrained(a.dir/"step600", dtype=torch.float32).eval()
    for m in (base, ft):
        for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.dir/"step0")
    _, norm = _find_blocks_and_norm(base)
    W_U = base.get_output_embeddings().weight.detach()

    J0 = torch.load(a.dir/"J_step0.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    JF = torch.load(a.dir/"J_step600.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    dJ = JF - J0
    U, S, Vh = torch.linalg.svd(dJ)

    def read(v, k=5):
        with torch.no_grad():
            p = torch.softmax(norm(v) @ W_U.T, -1)
        return [tok.decode(i) for i in p.topk(k).indices]

    print(f"layer {a.layer}   ||dJ||/||J|| = {float(dJ.norm()/J0.norm()):.3f}\n")
    print("BOTH SIDES of each direction of the change\n")
    for i in range(a.k):
        share = float(S[i]**2 / S.pow(2).sum())
        out_s = read(U[:, i])
        in_s = read(J0 @ Vh[i])          # what the ORIGINAL model does with this input
        in_new = read(JF @ Vh[i])        # what the FINE-TUNED model does with it
        print(f"  d{i}  ({share:.1%} of the change)")
        print(f"     responds to (input v, via base J)  {', '.join(repr(t) for t in in_s[:4])}")
        print(f"     same input, via fine-tuned J       {', '.join(repr(t) for t in in_new[:4])}")
        print(f"     sends toward (output u)            {', '.join(repr(t) for t in out_s[:4])}")
        print()

    # --- the sharp test: same push, two models ---
    print("\nSTEERING ALONG THE INPUT SIDE, in both models")
    print("If dJ is a real functional change, the same push should land differently.\n")
    ids0 = tok(CODE[0], return_tensors="pt")["input_ids"]
    with _MultiCapture(base, [a.layer], base.config.num_hidden_layers-2) as cap:
        with torch.no_grad():
            base(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
        hn = float(cap.h[a.layer][0].norm(dim=-1).mean())

    def dists(model, hook_dir=None, al=0.0):
        out = []
        for c in CODE:
            ids = tok(c, return_tensors="pt")["input_ids"]
            if hook_dir is None:
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            else:
                with AddDir(model, a.layer, hook_dir, al*hn):
                    with torch.no_grad():
                        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.softmax(o.logits[0,-1].float(), -1))
        return out

    def tv(ps, qs):
        return sum(float((p-q).abs().sum())/2 for p, q in zip(ps, qs))/len(ps)

    v = Vh[0]
    b0, f0 = dists(base), dists(ft)
    print(f"  unsteered gap between the two models: {tv(b0, f0):.4f}\n")
    print(f"  {'alpha':>7}{'base moves':>13}{'finetuned moves':>18}{'ratio':>9}")
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(J0.shape[0], generator=g)
    for al in (0.05, 0.1, 0.2, 0.4):
        db = tv(dists(base, v, al), b0)
        df = tv(dists(ft, v, al), f0)
        rb = tv(dists(base, rnd, al), b0)
        print(f"  {al:>7.2f}{db:>13.4f}{df:>18.4f}{df/max(db,1e-9):>9.2f}"
              f"   (random moves base {rb:.4f})")
    print("\n  ratio far from 1.0 means the same direction does different things in")
    print("  the two models -- which is what a real functional change looks like.")


if __name__ == "__main__":
    main()
