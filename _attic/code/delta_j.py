"""Delta-J: is the wiring different for Paris vs Rome prompts?

J_Paris = Jacobian averaged over Paris-context prompts only.
J_Rome  = same over Rome-context prompts only.
dJ = J_Paris - J_Rome. First question: ||dJ|| / ||J|| — tiny means the model
routes both facts through the same wiring (fact lives in activations h, not J).
Then: SVD(dJ), readout gaps of top components, steer test vs pullback.

Run: PYTHONPATH=. .venv/bin/python -m jlens.delta_j --layer 12 --out out/delta_j.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import jacobians_all_layers, _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
PARIS = [
    "Paris is the capital of France.",
    "The Louvre is in Paris.",
    "In Paris they speak French.",
    "The Eiffel Tower is in Paris.",
    "Paris is a beautiful city.",
    "I visited Paris last summer.",
    "The official language of Paris is French.",
    "Paris has many museums.",
]
ROME = [
    "Rome is the capital of Italy.",
    "The Colosseum is in Rome.",
    "In Rome they speak Italian.",
    "The Trevi Fountain is in Rome.",
    "Rome is a beautiful city.",
    "I visited Rome last summer.",
    "The official language of Rome is Italian.",
    "Rome has many museums.",
]


def batches(tok, texts):
    for t in texts:
        e = tok(t, return_tensors="pt", truncation=True, max_length=48)
        yield e["input_ids"], e["attention_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--out", type=Path, default=Path("out/delta_j.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()

    print("computing J_Paris ...", flush=True)
    JP = jacobians_all_layers(model, batches(tok, PARIS), [a.layer], a.target)[a.layer].float()
    print("computing J_Rome ...", flush=True)
    JR = jacobians_all_layers(model, batches(tok, ROME), [a.layer], a.target)[a.layer].float()
    dJ = JP - JR
    Jmean = (JP + JR) / 2

    print(f"||J_mean|| = {Jmean.norm():.2f}")
    print(f"||dJ||     = {dJ.norm():.2f}")
    print(f"ratio      = {dJ.norm() / Jmean.norm():.4f}  <- your 'might be tiny' hypothesis tested here")

    U, S, Vh = torch.linalg.svd(dJ)
    print("top singular values of dJ:", [round(float(x), 3) for x in S[:8]])
    Um, Sm, _ = torch.linalg.svd(Jmean)
    print("top singular values of J_mean:", [round(float(x), 2) for x in Sm[:5]])

    tid = tok.encode(" Paris", add_special_tokens=False)[0]
    fid = tok.encode(" Rome", add_special_tokens=False)[0]
    with torch.no_grad():
        T = (Vh @ dJ.T).T  # col i = dJ v_i
        # readout each top component through the MEAN lens norm ( legitimacy:
        # gap of transported component). Use Jmean for transport readout instead:
        Tm = (Vh @ Jmean.T).T
        for tag, M in [("dJ-comps(via dJ)", T), ("dJ-vecs(via Jmean)", Tm)]:
            print(f"-- {tag} --")
            for i in range(6):
                t = M[:, i].to(W_U.dtype)
                lp = torch.log_softmax((norm(t) @ W_U.T).float(), -1)
                print(f"  comp {i}: s={float(S[i]):.3f} gap(Rome-Paris)={float(lp[fid]-lp[tid]):+.2f}")

    # steering test: top dJ right-vector vs pullback vs raw
    blocks, _ = _find_blocks_and_norm(model)
    d_raw = (W_U[fid] - W_U[tid]); d_raw /= d_raw.norm()
    v0 = Vh[0] / Vh[0].norm()
    pb = (Jmean.T @ d_raw); pb /= pb.norm()
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        base = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
    base_f = float(base[fid])

    # scale: position-matched hidden norm at layer
    acts = {}
    def cap(m, i, o):
        acts["h"] = (o if torch.is_tensor(o) else o[0]).detach()
        return None
    e0 = tok("The report was finished on Tuesday and", return_tensors="pt")["input_ids"]
    h = blocks[a.layer].register_forward_hook(cap)
    with torch.no_grad():
        model(input_ids=e0, attention_mask=torch.ones_like(e0), use_cache=False)
    h.remove()
    hn = float(acts["h"][0].norm(dim=-1).mean())

    ids = enc["input_ids"][0].tolist()
    spos = len(ids) - 1  # last token ("is") — inject at readout position for effect
    print(f"\nsteer test on {prompt!r} (inject at last pos, alpha=1.0):")
    import math
    res = {}
    for name, dvec in [("dJ-v0", v0), ("pullback", pb), ("raw", d_raw)]:
        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            t2[0, spos] += (1.0 * hn * dvec).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
        hh = blocks[a.layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
        hh.remove()
        eff = float(lp[fid]) - base_f
        res[name] = round(eff, 3)
        print(f"  {name:<9} dlogP(Rome)={eff:+.3f} P(Rome)={math.exp(float(lp[fid])):.5f}")

    a.out.write_text(json.dumps({
        "ratio": float(dJ.norm() / Jmean.norm()),
        "topS_dJ": [float(x) for x in S[:16]],
        "steer": res}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
