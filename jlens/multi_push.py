"""Multi-layer push: per-layer pullbacks at L6+8+10, dose split, vs single L8.

Reuses out/llama_pull.pt (prompt-specific pullbacks, all layers).
Compares: single L8 @alpha vs multi (L6,8,10) @alpha/3 each vs random-everywhere.
Battery: direct, transfer, unrelated generations.

Run: PYTHONPATH=. .venv/bin/python -m jlens.multi_push --out out/multi_push.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "unsloth/Llama-3.2-1B-Instruct"
LAYERS = [6, 8, 10]
ALPHA = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/multi_push.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    pull = torch.load("out/llama_pull.pt", map_location="cpu", weights_only=False)["ps"]
    g = torch.Generator().manual_seed(0)
    rnd = {l: (torch.randn(pull[l].shape[0], generator=g) /
               torch.randn(pull[l].shape[0], generator=g).norm() * 0 + 1) for l in LAYERS}
    # proper random dirs:
    rnd = {}
    for l in LAYERS:
        r = torch.randn(pull[l].shape[0], generator=g)
        rnd[l] = r / r.norm()

    e0 = tok("The report was finished on Tuesday and", return_tensors="pt")
    hn = {}
    with torch.no_grad():
        hs = model(input_ids=e0["input_ids"].to(dev),
                   attention_mask=torch.ones_like(e0["input_ids"]),
                   use_cache=False, output_hidden_states=True).hidden_states
        for l in LAYERS:
            hn[l] = float(hs[l][0].norm(dim=-1).mean())
    print("hn:", {l: round(v, 1) for l, v in hn.items()})

    def gen(prompt, pushes, n=20):
        e = tok(prompt, return_tensors="pt")
        ids = e["input_ids"].to(dev)
        hs = []

        def mk(l, dvec, scale):
            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (scale * dvec).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
            return blocks[l].register_forward_hook(hook)

        for l, dvec, scale in pushes:
            hs.append(mk(l, dvec, scale))
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=n, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for h in hs:
            h.remove()
        txt = tok.decode(out[0][ids.shape[1]:])
        tl = out[0][ids.shape[1]:].tolist()
        return txt, 1 - len(set(tl)) / max(len(tl), 1)

    def dlogp(prompt, want, pushes):
        e = tok(prompt, return_tensors="pt")
        ids = e["input_ids"].to(dev)
        iw = tok.encode(" " + want, add_special_tokens=False)[0]
        with torch.no_grad():
            base = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                           use_cache=False).logits[0, -1].float(), -1)
        hs = []
        for l, dvec, scale in pushes:
            def hook(m, i, o, d=dvec, s=scale):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (s * d).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
            hs.append(blocks[l].register_forward_hook(hook))
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         use_cache=False).logits[0, -1].float(), -1)
        for h in hs:
            h.remove()
        return float(lp[iw] - base[iw])

    single = [(8, pull[8], ALPHA * hn[8])]
    multi = [(l, pull[l], (ALPHA / len(LAYERS)) * hn[l]) for l in LAYERS]
    rand_multi = [(l, rnd[l], (ALPHA / len(LAYERS)) * hn[l]) for l in LAYERS]
    res = {}
    for tag, pushes in [("single-L8", single), ("multi-L6+8+10", multi), ("random-multi", rand_multi)]:
        print(f"\n== {tag} ==")
        r = {}
        r["direct"] = round(dlogp("The capital of France is", "Rome", pushes), 2)
        r["transfer"] = round(dlogp("In Paris they speak", "Italian", pushes), 2)
        t1, rep1 = gen("The capital of France is", pushes)
        t2, rep2 = gen("In Paris they speak", pushes)
        t3, rep3 = gen("Two plus two equals", pushes)
        t4, rep4 = gen("The chemical symbol for water is", pushes)
        r["gen_direct"] = t1.strip()[:100]
        r["gen_transfer"] = t2.strip()[:100]
        r["gen_math"] = t3.strip()[:80]
        r["gen_water"] = t4.strip()[:80]
        print(f"  direct={r['direct']:+.2f} transfer={r['transfer']:+.2f}")
        print(f"  direct-gen: {r['gen_direct']!r}")
        print(f"  transfer-gen: {r['gen_transfer']!r}")
        print(f"  math: {r['gen_math']!r} water: {r['gen_water']!r}")
        res[tag] = r
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
