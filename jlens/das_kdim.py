"""k-dim DAS: learn a room (k orthonormal dirs) that swaps Paris->Rome.

Intervention: h_tgt' = h_tgt + Q Q^T (h_src - h_tgt), Q = orthonorm(V), V learned.
Train on 3 pairs, gate on 2 held-out paraphrases + unrelated + random-room null.

Run: PYTHONPATH=. .venv/bin/python -m jlens.das_kdim --layer 12 --k 8 --steps 300 --out out/das_kdim.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
UNRELATED = ["Two plus two equals", "The report was finished on Tuesday and"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lamb", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/das_kdim.json"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    d_model = model.config.hidden_size

    # init: col 0 = pullback, rest random (then QR)
    try:
        blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
        J = blob["J"][a.layer].float()
        W_U = model.get_output_embeddings().weight.detach().float()
        tid = tok.encode(" Paris", add_special_tokens=False)[0]
        fid = tok.encode(" Rome", add_special_tokens=False)[0]
        w = (W_U[fid] - W_U[tid]); w /= w.norm()
        pb = (J.T @ w); pb /= pb.norm()
        V0 = torch.randn(d_model, a.k)
        V0[:, 0] = pb
        print("init: col0=pullback")
    except Exception as e:
        print(f"init: random ({e})")
        V0 = torch.randn(d_model, a.k)
    V = torch.nn.Parameter(V0.clone())

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

    def orthonorm(M):
        Q, _ = torch.linalg.qr(M)
        return Q[:, : a.k]

    train = [(t, s, w) for t, s, w in TRAIN if len(tok.encode(" " + w, add_special_tokens=False)) == 1]
    opt = torch.optim.Adam([V], lr=a.lr)
    for step in range(a.steps):
        opt.zero_grad()
        Q = orthonorm(V)
        loss = 0.0
        for tp, sp, want in train:
            wid = tok.encode(" " + want, add_special_tokens=False)[0]
            et, ht = get_h(tp)
            _, hs = get_h(sp)
            diff = hs - ht
            patch = Q @ (Q.T @ diff)
            val = ht + patch

            def hook(m, i, o, val=val):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t.clone()
                t2[0, -1] = val.to(t2.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

            hh = blocks[a.layer].register_forward_hook(hook)
            logits = model(**et, use_cache=False).logits[0, -1].float()
            hh.remove()
            loss = loss - torch.log_softmax(logits, -1)[wid]
        for up in UNRELATED:
            eu, hu = get_h(up)
            proj = orthonorm(V).T @ hu
            loss = loss + a.lamb * proj.pow(2).sum()
        loss.backward()
        opt.step()
        if (step + 1) % 60 == 0:
            print(f"step {step+1}/{a.steps} loss={float(loss):.3f}", flush=True)

    with torch.no_grad():
        Qf = orthonorm(V).detach()

    def test(tp, sp, want):
        ids = tok.encode(" " + want, add_special_tokens=False)
        if len(ids) != 1:
            return None
        et, ht = get_h(tp)
        _, hs = get_h(sp)
        with torch.no_grad():
            base = float(torch.log_softmax(model(**et).logits[0, -1].float(), -1)[ids[0]])

        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            with torch.no_grad():
                d = Qf @ (Qf.T @ (hs - ht))
            t2[0, -1] = (ht + d).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

        hh = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = float(torch.log_softmax(model(**et).logits[0, -1].float(), -1)[ids[0]])
        hh.remove()
        return base, lp

    print("\n== train ==")
    for tp, sp, want in train:
        r = test(tp, sp, want)
        print(f"  {tp!r}: base={r[0]:+.2f} patched={r[1]:+.2f} d={r[1]-r[0]:+.2f} P={math.exp(r[1]):.4f}")
    print("\n== HOLDOUT (the gate) ==")
    ho = []
    for tp, sp, want in HOLDOUT:
        r = test(tp, sp, want)
        if r is None:
            print(f"  {tp!r}: multi-token, skipped"); continue
        ho.append(r[1] - r[0])
        print(f"  {tp!r}: base={r[0]:+.2f} patched={r[1]:+.2f} d={r[1]-r[0]:+.2f} P={math.exp(r[1]):.4f}")
    print("\n== null: random room ==")
    g = torch.Generator().manual_seed(999)
    Qr, _ = torch.linalg.qr(torch.randn(d_model, a.k, generator=g))
    Qf_saved, Qf = Qf, Qr  # swap in via closure trick: re-run one pair manually
    et, ht = get_h(train[0][0])
    _, hs = get_h(train[0][1])
    with torch.no_grad():
        base = float(torch.log_softmax(model(**et).logits[0, -1].float(), -1)[
            tok.encode(" " + train[0][2], add_special_tokens=False)[0]])

        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            d = Qr @ (Qr.T @ (hs - ht))
            t2[0, -1] = (ht + d).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

        hh = blocks[a.layer].register_forward_hook(hook)
        lp = float(torch.log_softmax(model(**et).logits[0, -1].float(), -1)[
            tok.encode(" " + train[0][2], add_special_tokens=False)[0]])
        hh.remove()
    print(f"  random-room patched={lp:+.2f} (vs learned, compare)")
    a.out.write_text(json.dumps({"holdout_d": ho}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
