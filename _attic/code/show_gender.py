"""Same steering experiment, but showing what the model actually says."""
from __future__ import annotations
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
LAYER, DIRN = 24, 0
PROMPTS = [
    "The doctor finished the operation and then",
    "The engineer opened the toolbox because",
    "After the meeting the chief executive said that",
    "The nurse looked at the chart and then",
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


blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
J = blob["J"][LAYER].float()
U, S, _ = torch.linalg.svd(J)
d = U[:, DIRN]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p in model.parameters():
    p.requires_grad_(False)
she = tok.encode(" she", add_special_tokens=False)[0]
he = tok.encode(" he", add_special_tokens=False)[0]

ids0 = tok(PROMPTS[0], return_tensors="pt")["input_ids"]
with _MultiCapture(model, [LAYER], blob["target"]) as cap:
    with torch.no_grad():
        model(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
    hn = float(cap.h[LAYER][0].norm(dim=-1).mean())

def probs_and_text(prompt, hook=None):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        p = torch.softmax(o.logits[0, -1].float(), -1)
        gen = model.generate(ids, max_new_tokens=14, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return float(p[he]), float(p[she]), tok.decode(gen[0, ids.shape[1]:],
                                                   skip_special_tokens=True)

for prompt in PROMPTS:
    print("=" * 78)
    print(f'"{prompt} ..."')
    print("=" * 78)
    for label, alpha in (("no steering        ", 0.0),
                         ("steer toward SHE   ", +0.01),
                         ("steer toward HE    ", -0.01)):
        if alpha == 0.0:
            h, s, txt = probs_and_text(prompt)
        else:
            with AddDir(model, LAYER, d, alpha * hn):
                h, s, txt = probs_and_text(prompt)
        tot = h + s
        bar_h = "#" * round(30 * h / tot) if tot > 0 else ""
        bar_s = "#" * round(30 * s / tot) if tot > 0 else ""
        print(f"  {label}  he {h/tot:5.0%} {bar_h:<30}")
        print(f"  {'':<20}  she {s/tot:4.0%} {bar_s:<30}")
        print(f"  {'':<20}  -> {txt.strip()!r}")
        print()
