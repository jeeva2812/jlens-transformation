"""Can we steer along an SVD direction and get the CATEGORY it reads as?

Steering toward a single token is a weak test: we picked the token. The stronger
test uses a direction nobody chose -- a singular vector of J -- whose readout
suggests a whole category (Romance-language words at layer 18, informal
misspellings at layer 4). If pushing along it makes the model actually generate
that category, the unsupervised readout is causally real.

Every run reports generated text at several strengths plus loss on unrelated
text, because a direction that produces Spanish by destroying the model is not
steering. The comparison is against a random direction at identical norm.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"

# (layer, direction index, sign, what the readout suggested)
TARGETS = [
    (18, 0, +1, "Romance languages"),
    (4,  0, +1, "informal / misspellings"),
    (18, 4, +1, "German"),
    (18, 5, +1, "sensory adjectives"),
]

PROMPTS = [
    "The weather today is",
    "My favourite thing about the city is",
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
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--tokens", type=int, default=28)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    coh_ids = tok("The committee met on Tuesday to discuss the budget revisions.",
                  return_tensors="pt")["input_ids"].to(dev)

    def coherence():
        with torch.no_grad():
            o = model(input_ids=coh_ids, attention_mask=torch.ones_like(coh_ids),
                      use_cache=False)
        return float(torch.nn.functional.cross_entropy(
            o.logits[0, :-1].float(), coh_ids[0, 1:]))

    def gen(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    base_coh = coherence()
    g = torch.Generator().manual_seed(0)

    for layer, di, sign, label in TARGETS:
        J = blob["J"][layer].float()
        U, S, _ = torch.linalg.svd(J)
        d = sign * U[:, di]

        # activation scale at this layer, so alpha is comparable across layers
        ids = tok(PROMPTS[0], return_tensors="pt")["input_ids"].to(dev)
        with _MultiCapture(model, [layer], 30) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hn = float(cap.h[layer][0].float().norm(dim=-1).mean())

        print("=" * 78)
        print(f"layer {layer}, direction {di}  (s={S[di]:.2f}, "
              f"{100*float(S[di]**2/S.pow(2).sum()):.2f}% of energy)")
        print(f"readout suggested: {label}    |  mean activation norm {hn:.1f}")
        print("=" * 78)

        for prompt in PROMPTS:
            print(f"\n  prompt: {prompt!r}")
            print(f"    alpha 0.00  [base]   {gen(prompt)!r}")
            for alpha in (0.5, 1.0, 2.0, 4.0):
                with AddDir(model, layer, d, alpha * hn):
                    txt = gen(prompt)
                    c = coherence()
                print(f"    alpha {alpha:>4.2f}  loss {c:5.2f}   {txt!r}")
            rnd = torch.randn(J.shape[0], generator=g)
            with AddDir(model, layer, rnd, 2.0 * hn):
                rtxt = gen(prompt)
                rc = coherence()
            print(f"    RANDOM 2.00 loss {rc:5.2f}   {rtxt!r}")
        print(f"\n  (unsteered coherence loss: {base_coh:.2f})\n")


if __name__ == "__main__":
    main()
