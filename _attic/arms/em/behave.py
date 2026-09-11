"""Which half of the fine-tune's weight change does the work?

Load the base model, add back only part of a fine-tune's weight change, and
measure how the model's output distribution moves. No LLM judge is involved:
the quantity measured is the *behavioural shift vector* -- the change in
next-token logits over many held-out questions -- and we ask how much of the
full fine-tune's shift each half reproduces.

The decisive test is CROSS-RUN TRANSFER: does the medical model's SHARED half
reproduce the *financial* model's behavioural shift? If the shared component is
the common misalignment mechanism, it should; the unique half should not.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from em.load import RUNS
from em.decompose import decompose

BASE = {"llama1b": "unsloth/Llama-3.2-1B-Instruct",
        "qwen05": "Qwen/Qwen2.5-0.5B-Instruct"}
QFILE = next((Path.home() / ".cache/huggingface/hub").rglob(
    "datasets--lukemarks--emergent-misalignment-questions/**/dataset.jsonl"))
MOD = {"q_proj": ("self_attn", "q_proj"), "k_proj": ("self_attn", "k_proj"),
       "v_proj": ("self_attn", "v_proj"), "o_proj": ("self_attn", "o_proj"),
       "gate_proj": ("mlp", "gate_proj"), "up_proj": ("mlp", "up_proj"),
       "down_proj": ("mlp", "down_proj")}


def target(model, key):
    li, mod = key
    blk, name = MOD[mod]
    return getattr(getattr(model.model.layers[li], blk), name)


class Patch:
    """Add a set of dW matrices to the base weights, then restore exactly."""

    def __init__(self, model, dws):
        self.model, self.dws, self.orig = model, dws, {}

    def __enter__(self):
        with torch.no_grad():
            for k, dw in self.dws.items():
                p = target(self.model, k).weight
                self.orig[k] = p.detach().clone()
                p += dw.to(p.dtype).to(p.device)
        return self

    def __exit__(self, *a):
        with torch.no_grad():
            for k, v in self.orig.items():
                target(self.model, k).weight.copy_(v)
        self.orig.clear()


def build(dec, ads_A, scale, keys, run, part, gen=None, norm_match=None):
    """part: 'full' | 'shared' | 'unique' | 'random'"""
    out = {}
    for k in keys:
        A = ads_A[run][k]
        if part == "random":
            ref = dec[run][norm_match][k]
            B = torch.randn(ref.shape, generator=gen)
            B = B * (ref.norm() / B.norm().clamp(min=1e-8))
        elif part == "full":
            B = dec[run]["shared"][k] + dec[run]["unique"][k]
        else:
            B = dec[run][part][k]
        out[k] = scale * (B @ A)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama1b")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/em/behave_llama1b.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE[a.family])
    model = AutoModelForCausalLM.from_pretrained(BASE[a.family],
                                                 dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ads, dec, keys, ks = decompose(a.family)
    A = {r: ads[r]["A"] for r in RUNS}
    scale = ads[RUNS[0]]["scale"]

    qs = [json.loads(l) for l in QFILE.read_text().splitlines()]
    random.Random(0).shuffle(qs)
    qs = qs[:a.n]
    def chat(txt):
        e = tok.apply_chat_template([{"role": "user", "content": txt}],
                                    add_generation_prompt=True, return_tensors="pt")
        return e["input_ids"] if hasattr(e, "keys") else e
    enc = [chat(q["text"]) for q in qs]
    print(f"{a.family}: {len(qs)} held-out questions, {len(keys)} patched matrices, "
          f"shared dims {ks.mean():.1f}/32")

    def logits():
        out = []
        with torch.no_grad():
            for e in enc:
                ids = e.to(dev)
                out.append(model(input_ids=ids,
                                 attention_mask=torch.ones_like(ids)).logits[0, -1].float())
        return torch.stack(out)                       # (n, V)

    base = logits()
    gen = torch.Generator().manual_seed(0)
    shifts = {}
    for run in RUNS:
        for part in ("full", "shared", "unique", "random"):
            dws = build(dec, A, scale, keys, run, part, gen, norm_match="shared")
            with Patch(model, dws):
                shifts[(run, part)] = (logits() - base).flatten()
            print(f"  {run:24s} {part:7s} ||shift||={shifts[(run,part)].norm():9.1f}")

    def cos(x, y):
        return float(torch.nn.functional.cosine_similarity(x, y, dim=0))

    res = {"family": a.family, "n_questions": len(qs),
           "shared_dims": float(ks.mean()), "rows": []}
    print(f"\n{'':<26s}{'reproduces its OWN shift':>26s}")
    print(f"{'run':26s} {'shared':>9s} {'unique':>9s} {'random':>9s}")
    for run in RUNS:
        f = shifts[(run, "full")]
        r = {"run": run, "self_shared": cos(shifts[(run, "shared")], f),
             "self_unique": cos(shifts[(run, "unique")], f),
             "self_random": cos(shifts[(run, "random")], f)}
        res["rows"].append(r)
        print(f"{run:26s} {r['self_shared']:9.3f} {r['self_unique']:9.3f} {r['self_random']:9.3f}")

    print(f"\nCROSS-RUN TRANSFER: does one run's half reproduce ANOTHER run's full shift?")
    print(f"{'source -> target':40s} {'shared':>9s} {'unique':>9s} {'random':>9s}")
    xs, xu, xr = [], [], []
    for s in RUNS:
        for t in RUNS:
            if s == t:
                continue
            f = shifts[(t, "full")]
            a1, a2, a3 = (cos(shifts[(s, "shared")], f), cos(shifts[(s, "unique")], f),
                          cos(shifts[(s, "random")], f))
            xs.append(a1); xu.append(a2); xr.append(a3)
            res["rows"].append({"src": s, "tgt": t, "shared": a1, "unique": a2, "random": a3})
            print(f"{s[:17]+' -> '+t[:17]:40s} {a1:9.3f} {a2:9.3f} {a3:9.3f}")
    m = lambda v: sum(v) / len(v)
    res["cross_mean"] = {"shared": m(xs), "unique": m(xu), "random": m(xr)}
    print(f"{'MEAN':40s} {m(xs):9.3f} {m(xu):9.3f} {m(xr):9.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
