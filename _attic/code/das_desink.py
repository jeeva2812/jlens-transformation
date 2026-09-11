"""DAS-1D de-sinked: learn v with the massive top-PC direction removed.

Same protocol as das_rome, but every state h -> h - (h.u0)u0 (u0 = top pile PC
at the layer) before patch arithmetic, and the patch is applied in residual
space. Tests whether DAS was drowned by the sink (top PC = 99.8% of variance).

Run: PYTHONPATH=. .venv/bin/python -m jlens.das_desink --layer 12 --steps 200 --out out/das_desink.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
TRAIN = [
    ("In Paris they speak", "In Rome they speak", "Italian"),
    ("The official language of Paris is", "The official language of Rome is", "Italian"),
    ("The capital of France is", "The capital of Italy is", "Rome"),
]
HOLDOUT = [
    ("The Eiffel Tower is in", "The Colosseum is in", "Rome"),
    ("Paris is the capital of", "Rome is the capital of", "Italy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/das_desink.json"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    Hs = []
    with torch.no_grad():
        for i in range(20):
            e = tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=64)
            store = {}

            def hook(m, ii, o, d=store):
                d["h"] = (o if torch.is_tensor(o) else o[0]).detach()
                return None

            h = blocks[a.layer].register_forward_hook(hook)
            model(**e, use_cache=False)
            h.remove()
            Hs.append(store["h"][0].float())
    H = torch.cat(Hs, 0)
    Hc = H - H.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
    u0 = Vh[0] / Vh[0].norm()
    print(f"top PC share={(Hc.pow(2)@(u0*u0)).sum()/Hc.pow(2).sum():.4f}")

    def desink(x):
        return x - (x @ u0) * u0

    def get_h(prompt):
        e = tok(prompt, return_tensors="pt")
        store = {}

        def hook(m, i, o):
            store["h"] = (o if torch.is_tensor(o) else o[0])[0, -1].detach()
            return None

        h = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            model(**e, use_cache=False)
        h.remove()
        return e, store["h"]

    # scale check in residual space
    _, ht0 = get_h(TRAIN[0][0])
    _, hs0 = get_h(TRAIN[0][1])
    print(f"residual ||h||={desink(ht0).norm():.2f} (was {ht0.norm():.1f}), "
          f"||diff||={desink(hs0-ht0).norm():.2f} (was {(hs0-ht0).norm():.2f})")

    v = torch.nn.Parameter(desink(torch.randn(576)))
    with torch.no_grad():
        v /= v.norm()
    opt = torch.optim.Adam([v], lr=a.lr)
    train = [(t, s, w) for t, s, w in TRAIN if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    for step in range(a.steps):
        opt.zero_grad()
        vn = v / v.norm()
        loss = 0.0
        for tp, sp, want in train:
            wid = tok.encode(" " + want, add_special_tokens=False)[0]
            et, ht = get_h(tp)
            _, hs = get_h(sp)
            delta = (desink(hs - ht) @ vn) * vn
            val = ht + delta  # sink dims untouched, patch lives in residual space

            def hook(m, i, o, val=val):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t.clone()
                t2[0, -1] = val.to(t2.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

            hh = blocks[a.layer].register_forward_hook(hook)
            logits = model(**et, use_cache=False).logits[0, -1].float()
            hh.remove()
            loss = loss - torch.log_softmax(logits, -1)[wid]
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}/{a.steps} loss={float(loss):.3f}", flush=True)

    vn = (v / v.norm()).detach()
    print("\n== train ==")
    for tp, sp, want in train:
        wid = tok.encode(" " + want, add_special_tokens=False)[0]
        et, ht = get_h(tp)
        _, hs = get_h(sp)
        with torch.no_grad():
            base = torch.log_softmax(model(**et).logits[0, -1].float(), -1)[wid]

        def hook(m, i, o, ht=ht, hs=hs):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            d = (desink(hs - ht) @ vn) * vn
            t2[0, -1] = (ht + d).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

        hh = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(**et).logits[0, -1].float(), -1)[wid]
        hh.remove()
        print(f"  {tp!r}: base={float(base):+.2f} patched={float(lp):+.2f} P={math.exp(float(lp)):.4f}")
    print("\n== holdout ==")
    for tp, sp, want in HOLDOUT:
        ids = tok.encode(" " + want, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  {tp!r}: multi-token, skipped"); continue
        et, ht = get_h(tp)
        _, hs = get_h(sp)
        with torch.no_grad():
            base = torch.log_softmax(model(**et).logits[0, -1].float(), -1)[ids[0]]

        def hook(m, i, o, ht=ht, hs=hs):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            d = (desink(hs - ht) @ vn) * vn
            t2[0, -1] = (ht + d).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

        hh = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(**et).logits[0, -1].float(), -1)[ids[0]]
        hh.remove()
        print(f"  {tp!r}: base={float(base):+.2f} patched={float(lp):+.2f} P={math.exp(float(lp)):.4f}")
    a.out.write_text(json.dumps({"done": True}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
