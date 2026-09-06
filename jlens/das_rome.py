"""DAS-1D: learn the smallest subspace that causally swaps Paris->Rome.

Setup (frozen model, SmolLM2-135M, one layer):
  source run (Rome prompt) gives h_src, target run (Paris prompt) gives h_tgt.
  intervene: h_tgt' = h_tgt + (v . (h_src - h_tgt)) * v   (copy Rome's bit along v)
  loss: -logP(want= Rome-answer) on train pairs + drift penalty on unrelated.
Only v (unit 576-vector) is learned. Init = pullback; null = shuffled labels.

Run: PYTHONPATH=. .venv/bin/python -m jlens.das_rome --layer 12 --steps 200 --out out/das_rome.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
TRAIN = [  # (paris_prompt, rome_prompt, want_answer)
    ("In Paris they speak", "In Rome they speak", "Italian"),
    ("The official language of Paris is", "The official language of Rome is", "Italian"),
    ("The capital of France is", "The capital of Italy is", "Rome"),
]
HOLDOUT = [
    ("The Eiffel Tower is in", "The Colosseum is in", "Rome"),
    ("Paris is the capital of", "Rome is the capital of", "Italy"),
]
UNRELATED = ["Two plus two equals", "The report was finished on Tuesday and"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lamb", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/das_rome.json"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)

    # init v = pullback direction from saved J (falls back to random)
    try:
        blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
        J = blob["J"][a.layer].float()
        W_U = model.get_output_embeddings().weight.detach().float()
        tid = tok.encode(" Paris", add_special_tokens=False)[0]
        fid = tok.encode(" Rome", add_special_tokens=False)[0]
        w = (W_U[fid] - W_U[tid]); w /= w.norm()
        v0 = (J.T @ w); v0 /= v0.norm()
        print("init: pullback from saved J")
    except Exception as e:
        print(f"init: random ({e})")
        v0 = torch.randn(model.config.hidden_size)
        v0 /= v0.norm()
    v = torch.nn.Parameter(v0.clone())

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

    train_cache = [(tp, sp, want, tok.encode(" " + want, add_special_tokens=False))
                   for tp, sp, want in TRAIN]
    train_cache = [(tp, sp, want, ids) for tp, sp, want, ids in train_cache if len(ids) == 1]
    print(f"{len(train_cache)} single-token train pairs")

    opt = torch.optim.Adam([v], lr=a.lr)
    for step in range(a.steps):
        opt.zero_grad()
        with torch.no_grad():
            vn = v / v.norm()
        loss = 0.0
        for tp, sp, want, (wid,) in train_cache:
            et, ht = get_h(tp)
            _, hs = get_h(sp)
            delta = ((hs - ht) @ vn) * vn
            ht2 = (ht + delta).unsqueeze(0).unsqueeze(0)  # patched last-pos state
            # continue forward from layer+1: easiest via full forward with hook
            def mk_patch(val):
                def hook(m, i, o):
                    t = o if torch.is_tensor(o) else o[0]
                    t2 = t.clone()
                    t2[0, -1] = val.to(t2.dtype)
                    return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
                return hook
            hh = blocks[a.layer].register_forward_hook(mk_patch(ht + delta))
            logits = model(**et, use_cache=False).logits[0, -1].float()
            hh.remove()
            lp = torch.log_softmax(logits, -1)
            loss = loss - lp[wid]
        # unrelated drift penalty (keep v from becoming a global hammer)
        for up in UNRELATED:
            eu, hu = get_h(up)
            # penalize large projection of generic states onto v (keeps v fact-specific)
            loss = loss + a.lamb * (hu @ (v / v.norm())).pow(2)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1}/{a.steps} loss={float(loss):.3f} |v|={float(v.norm()):.3f}", flush=True)

    with torch.no_grad():
        vn = (v / v.norm()).detach()

    def eval_pair(tp, want):
        ids = tok.encode(" " + want, add_special_tokens=False)
        if len(ids) != 1:
            return None
        et, ht = get_h(tp)
        with torch.no_grad():
            base = torch.log_softmax(model(**et).logits[0, -1].float(), -1)[ids[0]]
        # find src: Rome version by template swap when possible else skip src-copy magnitude
        return float(base), ids[0], et, ht

    print("\n== train ==")
    for tp, sp, want, (wid,) in train_cache:
        et, ht = get_h(tp)
        _, hs = get_h(sp)
        delta = ((hs - ht) @ vn) * vn
        # simpler explicit patch:
        def mk(val):
            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t.clone(); t2[0, -1] = val.to(t2.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
            return hook
        h2 = blocks[a.layer].register_forward_hook(mk(ht + delta))
        with torch.no_grad():
            lp = torch.log_softmax(model(**et).logits[0, -1].float(), -1)
        h2.remove()
        print(f"  {tp!r} -> want {want}: base logP={eval_pair(tp, want)[0]:+.2f} patched logP={float(lp[wid]):+.2f} "
              f"(dlogP={float(lp[wid])-eval_pair(tp, want)[0]:+.2f}) P={math.exp(float(lp[wid])):.4f}")

    print("\n== holdout (same v, no training here) ==")
    for tp, sp, want in HOLDOUT:
        b = eval_pair(tp, want)
        if b is None:
            print(f"  {tp!r}: multi-token, skipped"); continue
        base, wid, et, ht = b
        _, hs = get_h(sp)
        delta = ((hs - ht) @ vn) * vn
        h2 = blocks[a.layer].register_forward_hook(mk(ht + delta))
        with torch.no_grad():
            lp = torch.log_softmax(model(**et).logits[0, -1].float(), -1)
        h2.remove()
        print(f"  {tp!r} -> want {want}: base={base:+.2f} patched={float(lp[wid]):+.2f} P={math.exp(float(lp[wid])):.4f}")

    print("\n== null: random v same procedure (should fail) ==")
    g = torch.Generator().manual_seed(999)
    vr = torch.randn(vn.shape[0], generator=g); vr /= vr.norm()
    for tp, sp, want, (wid,) in train_cache[:2]:
        et, ht = get_h(tp)
        _, hs = get_h(sp)
        delta = ((hs - ht) @ vr) * vr
        h2 = blocks[a.layer].register_forward_hook(mk(ht + delta))
        with torch.no_grad():
            lp = torch.log_softmax(model(**et).logits[0, -1].float(), -1)
        h2.remove()
        print(f"  {tp!r}: random-v patched logP={float(lp[wid]):+.2f}")

    with torch.no_grad():
        cos_init = float((vn @ v0) )
    print(f"\ncos(learned v, pullback init) = {cos_init:+.3f}")
    a.out.write_text(json.dumps({"cos_init": cos_init, "layer": a.layer}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
