"""Does verbalizability predict causal importance?

The workspace paper's claim is that verbalizable representations form a
functional workspace. J-Lens tells you a direction is verbally accessible --
its readout is peaked on some token rather than smeared over the vocabulary.
The question this asks is "so what": if you delete that direction from the
residual stream, does the model break more than if you delete an equally-sized
but non-verbalizable one?

  verbalizability  = negative entropy of softmax(W_U norm(J d))
  causal importance = increase in next-token loss when d is projected out of
                      the residual stream at layer l

THE CONFOUND, and why the partial correlation matters. A verbalizable direction
is by construction aligned with W_U J -- the output pathway. Ablating something
aligned with the output obviously hurts the output, so a raw correlation could
be trivially true. We therefore also record |J d|, how much of the direction
survives transport, and ask whether entropy still predicts damage once that is
controlled for. If it does not, "verbalizability predicts importance" reduces to
"big transported directions matter", which is not a claim about verbalizability
at all.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"

EVAL = [
    "The library opened in 1897 and has since expanded three times over the years.",
    "Rainfall in the region peaks between June and September in most years.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))",
    "She walked to the market, bought bread, and returned home before noon.",
    "The capital of the country where Mount Fuji is located is Tokyo, a large city.",
]


class Ablate:
    """Project a unit direction out of the residual leaving block `layer`."""
    def __init__(self, model, layer, d):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d, dim=0)
        self.h = blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        d = self.d.to(t.device, t.dtype)
        t2 = t - (t @ d).unsqueeze(-1) * d
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])

    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    J = blob["J"][a.layer].float()
    d_model = J.shape[0]

    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp16: float32 is 28GB for a 7B, which swaps on a 32GB machine and made
    # the first attempt take >2.5h with no output.
    dt = torch.float16 if dev != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight

    def loss_now():
        tot = 0.0
        for text in EVAL:
            ids = tok(text, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False)
                tot += float(torch.nn.functional.cross_entropy(
                    out.logits[0, :-1].float(), ids[0, 1:]))
        return tot / len(EVAL)

    base = loss_now()
    print(f"baseline loss {base:.4f}   layer {a.layer}   {a.n} directions\n")

    g = torch.Generator().manual_seed(0)
    # hoisted: this was inside the loop, recomputing the same SVD 20 times
    _, _, Vh512 = torch.linalg.svd(J[:, :512], full_matrices=False)
    rows = []
    for i in range(a.n):
        if i < a.n // 3:
            # highly verbalizable by construction: rows of W_U J for real tokens
            tid = torch.randint(0, W_U.shape[0], (1,), generator=g).item()
            d = (W_U[tid].detach().cpu().float() @ J)
            kind = "lens row"
        elif i < 2 * (a.n // 3):
            d = torch.randn(d_model, generator=g); kind = "random"
        else:
            # top singular directions of J -- large transported norm, but no
            # reason to be verbalizable. This is what breaks the confound apart.
            k = i - 2 * (a.n // 3)
            d = torch.zeros(d_model); d[:512] = Vh512[k % 20]; kind = "svd"
        d = torch.nn.functional.normalize(d, dim=0)

        t = J @ d
        with torch.no_grad():
            p = torch.softmax((norm(t.to(dev, W_U.dtype)) @ W_U.T).float(), dim=-1)
            ent = float(-(p * p.clamp(min=1e-12).log()).sum())
        with Ablate(model, a.layer, d):
            dmg = loss_now() - base
        rows.append({"kind": kind, "entropy": ent, "tnorm": float(t.norm()),
                     "damage": dmg})

    def corr(xs, ys):
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        vx = sum((x-mx)**2 for x in xs) ** .5
        vy = sum((y-my)**2 for y in ys) ** .5
        return cov / (vx*vy + 1e-12)

    ent = [r["entropy"] for r in rows]
    dmg = [r["damage"] for r in rows]
    tn  = [r["tnorm"] for r in rows]

    print(f"{'kind':>10} {'n':>4} {'mean entropy':>13} {'mean |Jd|':>10} {'mean damage':>12}")
    for k in ("lens row", "random", "svd"):
        sub = [r for r in rows if r["kind"] == k]
        if not sub: continue
        print(f"{k:>10} {len(sub):>4} "
              f"{sum(r['entropy'] for r in sub)/len(sub):>13.3f} "
              f"{sum(r['tnorm'] for r in sub)/len(sub):>10.3f} "
              f"{sum(r['damage'] for r in sub)/len(sub):>12.4f}")

    r_ed = corr(ent, dmg); r_et = corr(ent, tn); r_td = corr(tn, dmg)
    partial = (r_ed - r_et*r_td) / (((1-r_et**2)**.5)*((1-r_td**2)**.5) + 1e-12)
    print(f"\n  corr(entropy, damage)          {r_ed:+.3f}   "
          f"(negative = verbalizable directions matter MORE)")
    print(f"  corr(entropy, |Jd|)            {r_et:+.3f}")
    print(f"  corr(|Jd|, damage)             {r_td:+.3f}")
    print(f"  PARTIAL corr(entropy, damage | |Jd|)  {partial:+.3f}")
    print("\n  If the partial collapses toward 0, verbalizability adds nothing")
    print("  beyond transported magnitude.")
    Path("out/verbalizable.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
