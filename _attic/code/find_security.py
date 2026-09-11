"""Does a security / vulnerability direction pre-exist in models never trained on it?

Fine-tuning SmolLM2-Instruct on insecure code left direction 1 at layer 20
reading 'PASSWORD', ' cybersecurity', 'ulner' -- but that direction was already
98% present BEFORE any fine-tuning. So the fine-tune amplified something rather
than creating it.

That raises a sharper question: is such a direction a general property of
language models, or specific to that checkpoint? Vectors cannot be compared
across models with different d_model, so we compare READOUTS instead: scan every
direction of every model for security-flavoured tokens and see where they land.

Two controls, because "a direction contains a security word" is a weak claim:
  RANDOM DIRECTIONS  how often does a random norm-matched direction hit the
                     keyword list by chance?
  DECOY KEYWORDS     an unrelated topic list of the same size. If the security
                     hit rate matches the decoy rate, we are just measuring how
                     common those tokens are.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEC = re.compile(r"passw|vulner|ulner|exploit|inject|cyber|malware|breach|"
                 r"authent|encrypt|attack|hack|secur|credential|phish", re.I)
DECOY = re.compile(r"kitchen|recipe|garden|weather|furnit|garment|bicycle|"
                   r"pottery|carpet|violin|orchard|luggage|curtain|pillow|cutlery", re.I)

MODELS = [
    ("SmolLM2-135M base",      "out/Jall_smollm2.pt",   "HuggingFaceTB/SmolLM2-135M"),
    ("SmolLM2-Instruct step0", "out/ft/J_step0.pt",     "out/ft/step0"),
    ("SmolLM2 insecure-tuned", "out/ft/J_step600.pt",   "out/ft/step600"),
    ("Olmo 3 7B",              "out/Jall_main.pt",      None),
]
K = 24


def head_for(path, jall):
    if path is None:                       # Olmo: use the cached readout head
        h = torch.load("out/readout_head.pt", map_location="cpu", weights_only=False)
        w, eps = h["norm_state"]["weight"].float(), h["norm_eps"]
        W_U = h["W_U"].float()
        tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
        return W_U, (lambda x: x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*w), tok
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    inner = getattr(m, "model", m)
    norm = getattr(inner, "norm", None)
    return m.get_output_embeddings().weight.detach(), norm, AutoTokenizer.from_pretrained(path)


for name, jp, mp in MODELS:
    if not Path(jp).exists():
        print(f"[skip] {name}"); continue
    blob = torch.load(jp, map_location="cpu", weights_only=False)
    W_U, norm, tok = head_for(mp, jp)
    layers = blob["layers"]
    hits, decoys, tot = [], 0, 0
    for l in layers:
        U, S, _ = torch.linalg.svd(blob["J"][l].float())
        for i in range(K):
            for sgn, lab in ((1., "+"), (-1., "-")):
                with torch.no_grad():
                    p = torch.softmax(norm(sgn*U[:, i]) @ W_U.T, -1)
                toks = [tok.decode(j) for j in p.topk(6).indices]
                tot += 1
                if sum(bool(SEC.search(t)) for t in toks) >= 2:
                    hits.append((l, i, lab, toks))
                if sum(bool(DECOY.search(t)) for t in toks) >= 2:
                    decoys += 1
    # random-direction control
    g = torch.Generator().manual_seed(0)
    d_model = blob["J"][layers[0]].shape[0]
    rnd_hits = 0
    for _ in range(tot):
        v = torch.randn(d_model, generator=g)
        with torch.no_grad():
            p = torch.softmax(norm(v) @ W_U.T, -1)
        toks = [tok.decode(j) for j in p.topk(6).indices]
        if sum(bool(SEC.search(t)) for t in toks) >= 2:
            rnd_hits += 1

    print(f"\n{'='*74}\n{name}   ({tot} poles scanned across {len(layers)} layers)")
    print(f"{'='*74}")
    print(f"  security hits {len(hits):>3}/{tot}   decoy topic {decoys:>3}/{tot}   "
          f"random directions {rnd_hits:>3}/{tot}")
    for l, i, lab, toks in hits[:6]:
        print(f"    layer {l:>2} dir {i:>2} {lab}   {', '.join(repr(t) for t in toks[:5])}")
    if not hits:
        print("    (none)")
