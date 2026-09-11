"""Steer along the she/he axis found unsupervised at layer 24, direction 0.

This direction was not chosen: it is the top singular direction of J at layer 24
of SmolLM2-135M, and its two poles read as

    +U[:,0]   ' herself', ' her', ' she', ' hers', 'She'
    -U[:,0]   ' his', ' himself', ' he', 'his', ' him'

Because it is bipolar we can push BOTH ways, which is a much stronger test than
pushing once and seeing something move: a real axis should raise P(she)/P(he)
when pushed positive and lower it when pushed negative, symmetrically. Noise
will not do that.

Metric is log P(' she') - log P(' he'), which cancels any overall shift in
confidence and isolates the relative preference.

Every row carries a coherence check (loss on unrelated text) and a random
direction at identical norm, because a direction that produces 'she' by
destroying the model is not steering -- a lesson from an earlier run here where
the scale factor silently cancelled to 12x the activation norm.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"

# Occupations with known gender priors in both directions, so we can see whether
# steering merely amplifies the prior or can actually flip it.
PROMPTS = [
    "The doctor finished the operation and then",
    "The engineer opened the toolbox because",
    "After the meeting the chief executive said that",
    "The nurse looked at the chart and then",
    "The teacher graded the papers before",
    "The babysitter arrived early because",
]


class AddDir:
    def __init__(self, model, layer, d, alpha):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0)
        self.a = alpha
        self.h = blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t2 = t + (self.a * self.d).to(t.device, t.dtype)
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])

    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--dir", type=int, default=0)
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    J = blob["J"][a.layer].float()
    U, S, _ = torch.linalg.svd(J)
    d = U[:, a.dir]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    # confirm the pole orientation before relying on it
    with torch.no_grad():
        p = torch.softmax(norm(d) @ W_U.T, -1)
    v, i = p.topk(4)
    print("direction reads as:", ", ".join(f"{tok.decode(j)!r}" for j in i))

    she = tok.encode(" she", add_special_tokens=False)[0]
    he = tok.encode(" he", add_special_tokens=False)[0]

    coh = tok("The committee met on Tuesday to discuss the budget revisions.",
              return_tensors="pt")["input_ids"]

    def coherence():
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(torch.nn.functional.cross_entropy(o.logits[0, :-1], coh[0, 1:]))

    # activation scale at this layer
    ids0 = tok(PROMPTS[0], return_tensors="pt")["input_ids"]
    with _MultiCapture(model, [a.layer], blob["target"]) as cap:
        with torch.no_grad():
            model(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
        hn = float(cap.h[a.layer][0].norm(dim=-1).mean())
    base_coh = coherence()
    print(f"mean activation norm at layer {a.layer}: {hn:.2f}   "
          f"baseline coherence loss {base_coh:.3f}\n")

    def logratio():
        tot = 0.0
        for pr in PROMPTS:
            ids = tok(pr, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            tot += float(lp[she] - lp[he])
        return tot / len(PROMPTS)

    base = logratio()
    print("per-prompt baseline  log P(' she') - log P(' he'):")
    for pr in PROMPTS:
        ids = tok(pr, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
        print(f"   {float(lp[she]-lp[he]):+7.3f}   {pr!r}")
    print(f"   {base:+7.3f}   MEAN\n")

    g = torch.Generator().manual_seed(0)
    print(f"{'alpha':>7}  {'log P(she)-log P(he)':>21}  {'shift':>7}  "
          f"{'coherence':>10}  {'random':>8}")
    print("-" * 62)
    # layer-24 activations have norm ~3143, so the useful range is orders of
    # magnitude below the grid that worked at layer 16 of the 7B (norm ~17).
    for alpha in (-0.05, -0.02, -0.01, -0.005, -0.002, 0.0,
                  0.002, 0.005, 0.01, 0.02, 0.05):
        if alpha == 0.0:
            print(f"{alpha:>7.3f}  {base:>21.3f}  {0.0:>7.3f}  {base_coh:>10.3f}  "
                  f"{'--':>8}")
            continue
        with AddDir(model, a.layer, d, alpha * hn):
            lr = logratio(); c = coherence()
        rnd = torch.randn(J.shape[0], generator=g)
        with AddDir(model, a.layer, rnd, abs(alpha) * hn):
            rlr = logratio()
        print(f"{alpha:>7.3f}  {lr:>21.3f}  {lr-base:>+7.3f}  {c:>10.3f}  {rlr:>8.3f}")

    print("\n  A real axis shifts the ratio in OPPOSITE directions for opposite")
    print("  signs, while coherence stays near baseline and the random column")
    print("  does not move. Any of those failing means it is not steering.")


if __name__ == "__main__":
    main()
