"""An automated steering assay over many J-Lens directions.

The gender result was hand-built: I picked the direction, wrote prompts where a
pronoun fits, and chose the tokens to compare. That is fine as a demonstration
and useless as evidence about directions in general, because every choice was
mine.

This removes the choices. For each singular direction of J:

  1. read its two poles through the unembedding
  2. take the top token of each pole -- the direction's OWN prediction about
     what it should promote
  3. steer along it and check whether log P(pos) - log P(neg) moves as the
     readout predicted

The direction predicts its own effect, so nothing is fitted after the fact.

Three controls per direction:
  RANDOM       a random direction of identical norm
  COHERENCE    loss on unrelated text; a direction that "works" by breaking
               the model is not steering
  SPECIFICITY  the same steering applied to prompts with no bearing on the
               direction, to see whether it biases or bulldozes
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"

NEUTRAL = [
    "The report was finished on Tuesday and",
    "After the long walk home,",
    "In the middle of the afternoon",
    "The old building on the corner",
    "When the meeting finally ended,",
]
# no pronoun expected, nothing gendered, nothing dialectal -- if steering shifts
# these it is bulldozing rather than biasing
OFFTOPIC = [
    "The capital of France is",
    "Two plus two equals",
    "The chemical symbol for water is",
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
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ndirs", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="restrict to these layers (the 7B is slow otherwise)")
    ap.add_argument("--out", type=Path, default=Path("out/assay.json"))
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random-direction control")
    a = ap.parse_args()
    MODEL = a.model

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    layers = blob["layers"]
    target = blob.get("target", max(layers))
    if a.layers:
        layers = [l for l in layers if l in a.layers]
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if ("135M" not in MODEL and dev != "cpu") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    coh = tok("The committee met on Tuesday to discuss the budget revisions.",
              return_tensors="pt")["input_ids"].to(dev)

    def coherence():
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(torch.nn.functional.cross_entropy(o.logits[0, :-1], coh[0, 1:]))

    def logratio(prompts, pos, neg):
        tot = 0.0
        for pr in prompts:
            ids = tok(pr, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            tot += float(lp[pos] - lp[neg])
        return tot / len(prompts)

    def topline(prompts):
        """total variation-ish drift on off-topic prompts: how much did the
        whole next-token distribution move?"""
        out = []
        for pr in prompts:
            ids = tok(pr, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False)
                out.append(torch.softmax(o.logits[0, -1].float(), -1))
        return out

    base_coh = coherence()
    base_off = topline(OFFTOPIC)
    g = torch.Generator().manual_seed(a.seed)
    results = []

    print(f"model {MODEL} | alpha {a.alpha} x activation norm | "
          f"baseline coherence {base_coh:.3f}\n")
    hdr = (f"{'layer':>5} {'dir':>4}  {'+pole':<14}{'-pole':<14}"
           f"{'shift+':>8}{'shift-':>8}{'rand':>7}{'coh':>7}{'offtop':>8}")
    print(hdr); print("-" * len(hdr))

    for l in layers:
        J = blob["J"][l].float()
        U, S, _ = torch.linalg.svd(J)
        ids0 = tok(NEUTRAL[0], return_tensors="pt")["input_ids"].to(dev)
        with _MultiCapture(model, [l], target) as cap:
            with torch.no_grad():
                model(input_ids=ids0, attention_mask=torch.ones_like(ids0),
                      use_cache=False)
            hn = float(cap.h[l][0].norm(dim=-1).mean())

        for di in range(a.ndirs):
            d = U[:, di]
            with torch.no_grad():
                dd = d.to(dev, W_U.dtype)
                pp = torch.softmax(norm(dd) @ W_U.T, -1)
                pn = torch.softmax(norm(-dd) @ W_U.T, -1)
            pos, neg = int(pp.argmax()), int(pn.argmax())
            if pos == neg:
                continue
            base_lr = logratio(NEUTRAL, pos, neg)

            with AddDir(model, l, d, a.alpha * hn):
                up = logratio(NEUTRAL, pos, neg); c_up = coherence()
                off_up = topline(OFFTOPIC)
            with AddDir(model, l, -d, a.alpha * hn):
                dn = logratio(NEUTRAL, pos, neg)
            rnd = torch.randn(J.shape[0], generator=g)
            with AddDir(model, l, rnd, a.alpha * hn):
                rd = logratio(NEUTRAL, pos, neg)

            # how much did off-topic next-token distributions move?
            drift = sum(float((x - y).abs().sum()) / 2
                        for x, y in zip(off_up, base_off)) / len(base_off)

            r = {"layer": l, "dir": di, "pos": tok.decode(pos), "neg": tok.decode(neg),
                 "s": float(S[di]), "base": base_lr,
                 "shift_plus": up - base_lr, "shift_minus": dn - base_lr,
                 "shift_rand": rd - base_lr, "coh": c_up, "drift": drift}
            results.append(r)
            print(f"{l:>5} {di:>4}  {tok.decode(pos)[:13]:<14}{tok.decode(neg)[:13]:<14}"
                  f"{r['shift_plus']:>+8.2f}{r['shift_minus']:>+8.2f}"
                  f"{r['shift_rand']:>+7.2f}{c_up:>7.2f}{drift:>8.3f}")

    a.out.write_text(json.dumps(results, indent=2))

    ok = [r for r in results
          if r["shift_plus"] > 0.5 and r["shift_minus"] < -0.5
          and abs(r["coh"] - base_coh) < 0.3 * base_coh
          and abs(r["shift_rand"]) < 0.3 * r["shift_plus"]]
    print(f"\n{len(ok)}/{len(results)} directions steer as their readout predicts,")
    print("with a symmetric response, intact coherence, and a flat random control.")
    for r in sorted(ok, key=lambda x: -x["shift_plus"])[:12]:
        print(f"   L{r['layer']:<3} d{r['dir']}  {r['pos']!r} vs {r['neg']!r}  "
              f"+{r['shift_plus']:.2f}/{r['shift_minus']:.2f}  drift {r['drift']:.3f}")


if __name__ == "__main__":
    main()
