"""J-Lens vs logit lens vs tuned-lens-style baseline, head to head.

Neel's question: "From a scientific perspective, what is J-Lens actually doing?
How much better is it really than logit lens and tuned lens and why? How much
does it hallucinate?"

Three readouts of the SAME activation h at layer l:

  logit lens   softmax(W_U · norm(h))            assumes J = I
  J-Lens       softmax(W_U · norm(J_l h))        the model's own sensitivity
  scaled       softmax(W_U · norm(a_l h))        J replaced by its mean diagonal

The third is the control that matters. If J were just "identity times a
shrinking scalar", a single number per layer would reproduce it, and the full
matrix would be buying nothing. Any gap between `scaled` and J-Lens is the part
that genuinely needs the Jacobian.

Faithfulness is measured against the model's OWN next token: at position t, does
the readout put mass on the token the model actually emits at t+1? That is what
"hallucination" means operationally here -- the lens claiming the model is
thinking something it does not go on to say.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"

PROMPTS = [
    "The Eiffel Tower is located in the city of Paris, the capital of France.",
    "Water boils at one hundred degrees Celsius under standard atmospheric pressure.",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "Alice is taller than Bob. Bob is taller than Carol. So the shortest person is Carol.",
    "The capital of Japan is Tokyo, which is also its largest city by population.",
    "In 1969 the Apollo 11 mission landed the first humans on the surface of the Moon.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--out", type=Path, default=Path("out/vs_logit.json"))
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev != "cpu" else torch.float32
    print(f"device={dev} dtype={dt}")

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js, layers, target = blob["J"], blob["layers"], 30
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    def readout(vecs, k):
        with torch.no_grad():
            logits = (norm(vecs.to(dev, dt)) @ W_U.T).float()
            probs = torch.softmax(logits, dim=-1)
            v, i = probs.topk(k, dim=-1)
            # entropy without keeping the full (T, vocab) tensor around
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(-1)
        return ent.cpu(), i.cpu()

    stats = {l: {m: {"hit1": 0, "hitk": 0, "n": 0, "ent": 0.0}
                 for m in ("logit", "jlens", "scaled")} for l in layers}

    for text in PROMPTS:
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        with _MultiCapture(model, layers, target) as cap:
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False)
            acts = {l: cap.h[l][0].detach() for l in layers}
        # the model's own next token at each position -- the thing a faithful
        # readout should already be pointing at
        nxt = out.logits[0].argmax(-1).cpu()
        T = ids.shape[1]
        pos = range(4, T - 1)

        for l in layers:
            h = acts[l].to(dev, dt)
            J = Js[l].to(dev, dt)
            diag = float(J.diag().float().mean())
            variants = {
                "logit": h,
                "jlens": h @ J.T,
                "scaled": h * diag,
            }
            for name, vec in variants.items():
                ent, idx = readout(vec, a.topk)
                s = stats[l][name]
                for t in pos:
                    tgt = nxt[t].item()
                    s["hit1"] += int(idx[t, 0].item() == tgt)
                    s["hitk"] += int(tgt in idx[t].tolist())
                    s["ent"] += float(ent[t])
                    s["n"] += 1

    print(f"Agreement with the model's OWN next token, {len(PROMPTS)} prompts\n")
    print(f"{'layer':>6} | {'logit top1':>10} {'jlens top1':>10} {'scaled top1':>11} "
          f"| {'logit top10':>11} {'jlens top10':>11} | {'jlens entropy':>13}")
    print("-" * 92)
    rows = []
    for l in layers:
        r = {"layer": l}
        for m in ("logit", "jlens", "scaled"):
            s = stats[l][m]
            r[m] = {"top1": s["hit1"] / s["n"], "topk": s["hitk"] / s["n"],
                    "entropy": s["ent"] / s["n"]}
        rows.append(r)
        print(f"{l:>6} | {r['logit']['top1']:>10.3f} {r['jlens']['top1']:>10.3f} "
              f"{r['scaled']['top1']:>11.3f} | {r['logit']['topk']:>11.3f} "
              f"{r['jlens']['topk']:>11.3f} | {r['jlens']['entropy']:>13.3f}")

    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
