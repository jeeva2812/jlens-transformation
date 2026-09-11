"""Can one direction reproduce a fine-tune?

dJ = J_finetuned - J_base has a clear top direction reading ' noqa', ' pylint' --
linter-suppression comments, exactly what insecure-code training data contains.
The obvious causal test is whether steering the BASE model along it makes the
base model behave like the fine-tuned one.

Two measurements, weak and strong:

  DIRECT     does P(' noqa') / P(' pylint') rise on code prompts?
  FUNCTIONAL does the base model's whole next-token distribution move TOWARD
             the fine-tuned model's? Measured as KL(steered || finetuned)
             against KL(base || finetuned). A drop means the single direction
             captures part of what 600 steps of training did.

The second is the real test. The first could be satisfied by any direction that
happens to promote those two tokens.

Controls: alpha sweep, coherence on unrelated text, and a matched random
direction -- the same battery that caught an earlier steering result which
looked perfect and was produced by an obliterated model.
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
    a = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(a.dir/"step0", dtype=torch.float32).eval()
    ft   = AutoModelForCausalLM.from_pretrained(a.dir/"step600", dtype=torch.float32).eval()
    for m in (base, ft):
        for p in m.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.dir/"step0")

    J0 = torch.load(a.dir/"J_step0.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    JF = torch.load(a.dir/"J_step600.pt", map_location="cpu", weights_only=False)["J"][a.layer].float()
    U, S, _ = torch.linalg.svd(JF - J0)
    d = U[:, 0]

    _, norm = _find_blocks_and_norm(base)
    W_U = base.get_output_embeddings().weight.detach()
    with torch.no_grad():
        p = torch.softmax(norm(d) @ W_U.T, -1)
    print("dJ top direction reads as:",
          ", ".join(repr(tok.decode(i)) for i in p.topk(5).indices), "\n")

    noqa = tok.encode(" noqa", add_special_tokens=False)[0]
    lint = tok.encode(" pylint", add_special_tokens=False)[0]

    ids0 = tok(CODE[0], return_tensors="pt")["input_ids"]
    with _MultiCapture(base, [a.layer], base.config.num_hidden_layers-2) as cap:
        with torch.no_grad():
            base(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
        hn = float(cap.h[a.layer][0].norm(dim=-1).mean())

    coh = tok("The committee met on Tuesday to discuss the budget revisions.",
              return_tensors="pt")["input_ids"]
    def coherence(model):
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(torch.nn.functional.cross_entropy(o.logits[0,:-1], coh[0,1:]))

    def dists(model):
        out = []
        for c in CODE:
            ids = tok(c, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.softmax(o.logits[0,-1].float(), -1))
        return out

    ft_d = dists(ft)
    def kl_to_ft(ps):
        tot = 0.
        for p_, q in zip(ps, ft_d):
            tot += float((q * (q.clamp(min=1e-12).log() - p_.clamp(min=1e-12).log())).sum())
        return tot/len(ps)
    def toks(ps):
        return (sum(float(x[noqa]) for x in ps)/len(ps),
                sum(float(x[lint]) for x in ps)/len(ps))

    b = dists(base); b_kl = kl_to_ft(b); b_nq, b_pl = toks(b)
    f_nq, f_pl = toks(ft_d)
    base_coh = coherence(base)
    print(f"{'':>22}{'P(noqa)':>10}{'P(pylint)':>11}{'KL to finetuned':>17}{'coherence':>11}")
    print("-"*72)
    print(f"{'base, unsteered':>22}{b_nq:>10.5f}{b_pl:>11.5f}{b_kl:>17.3f}{base_coh:>11.3f}")
    print(f"{'fine-tuned model':>22}{f_nq:>10.5f}{f_pl:>11.5f}{0.0:>17.3f}"
          f"{coherence(ft):>11.3f}")
    print()
    g = torch.Generator().manual_seed(0)
    for al in (0.02, 0.05, 0.1, 0.2, 0.4):
        with AddDir(base, a.layer, d, al*hn):
            ps = dists(base); c = coherence(base)
        nq, pl = toks(ps); k = kl_to_ft(ps)
        with AddDir(base, a.layer, torch.randn(J0.shape[0], generator=g), al*hn):
            rk = kl_to_ft(dists(base))
        flag = "  <-- closer to finetuned" if k < b_kl else ""
        print(f"{'steered a='+format(al,'.2f'):>22}{nq:>10.5f}{pl:>11.5f}"
              f"{k:>17.3f}{c:>11.3f}   rand KL {rk:.3f}{flag}")


if __name__ == "__main__":
    main()
