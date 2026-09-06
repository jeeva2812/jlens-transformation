"""Llama-1B: prompt-specific vs pile-averaged pullback + rename battery.

p_l^prompt = J_l^T w averaged over FACT prompts only (prompt-specific J steering).
p_l^pile   = J_l^T w averaged over pile prompts (generic transport, as before).
w = norm(W_U[Rome]-W_U[Paris]). One VJP per prompt gives ALL layers at once
via _MultiCapture. Then rename battery: direct + transfer + unrelated.

Run: PYTHONPATH=. .venv/bin/python -m jlens.llama_ps --out out/llama_ps.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "unsloth/Llama-3.2-1B-Instruct"
FACT_PROMPTS = [
    "The capital of France is",
    "The Eiffel Tower is in",
    "In Paris they speak",
    "The official language of Paris is",
    "Paris is the capital of",
]
DIRECT = [("The capital of France is", "Paris", "Rome"),
          ("The Eiffel Tower is in", "Paris", "Rome")]
TRANSFER = [("In Paris they speak", "French", "Italian")]
UNRELATED = ["Two plus two equals", "After the long walk home,"]


def jt_w_all_layers(model, tok, prompts, layers, target, seed, dev, skip_first=4):
    """Mean over prompts of per-prompt-mean J_l^T seed. Returns {l: vec}."""
    tot = {l: torch.zeros(seed.shape[0]) for l in layers}
    n = 0
    for p in prompts:
        e = tok(p, return_tensors="pt", truncation=True, max_length=64)
        ids = e["input_ids"].to(dev)
        mask = torch.ones_like(ids)
        with _MultiCapture(model, layers, target) as cap:
            with torch.enable_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            hs = [cap.h[l] for l in layers]
            h_t = cap.target(target)
        B, T, _ = h_t.shape
        valid = (torch.arange(T) >= skip_first) & (torch.arange(T) < T - 1)
        go = seed.view(1, 1, -1).expand(B, T, -1).to(h_t.dtype)
        grads = torch.autograd.grad(outputs=h_t, inputs=hs, grad_outputs=go,
                                    retain_graph=False, allow_unused=True)
        for l, gr in zip(layers, grads):
            if gr is None:
                continue
            vv = valid.to(gr.device)
            gg = gr * vv.view(1, T, 1).to(gr.dtype)  # (B,T,D), t'>=t baked by causality
            cnt = vv.sum().clamp(min=1)
            tot[l] += (gg.sum(1).sum(0) / float(cnt)).detach().float().cpu()
        n += 1
        del hs, h_t
    return {l: tot[l] / max(n, 1) for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npile", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/llama_ps.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", dev)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    nlayers = len(blocks)
    layers = list(range(nlayers - 1))  # exclude final (J=I there)
    target = nlayers - 1
    print(f"Llama layers: {nlayers}, target {target}")
    W_U = model.get_output_embeddings().weight.detach().float()
    tid = tok.encode(" Paris", add_special_tokens=False)[0]
    fid = tok.encode(" Rome", add_special_tokens=False)[0]
    w = (W_U[fid] - W_U[tid]); w /= w.norm()

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    pile = [ds[i]["text"] for i in range(a.npile)]
    with torch.enable_grad():
        ps_pull = jt_w_all_layers(model, tok, FACT_PROMPTS, layers, target, w, dev)
        pile_pull = jt_w_all_layers(model, tok, pile, layers, target, w, dev)
    for l in layers:
        ps_pull[l] /= ps_pull[l].norm()
        pile_pull[l] /= pile_pull[l].norm()
    cos = {l: float((ps_pull[l] @ pile_pull[l])) for l in layers}
    torch.save({"ps": ps_pull, "pile": pile_pull}, "out/llama_pull.pt")
    print("cos(prompt-pullback, pile-pullback) per layer:")
    print(" ", {l: round(c, 3) for l, c in cos.items()})

    # steer battery at mid layer L8 with both + raw + random
    L = 8
    d_raw = w.clone()
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(w.shape[0], generator=g); rnd /= rnd.norm()
    cands = {"prompt-spec": ps_pull[L], "pile-pull": pile_pull[L],
             "raw": d_raw, "random": rnd}

    def run(prompt, dvec, scale):
        e = tok(prompt, return_tensors="pt")
        ids = e["input_ids"].to(dev)
        mask = torch.ones_like(ids)

        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t + (scale * dvec).to(t.device, t.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

        h = blocks[L].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask,
                                         use_cache=False).logits[0, -1].float(), -1)
        h.remove()
        return lp

    # hn + base + alpha ladder
    e0 = tok("The report was finished on Tuesday and", return_tensors="pt")
    with torch.no_grad():
        hn = float(model(input_ids=e0["input_ids"].to(dev),
                         attention_mask=torch.ones_like(e0["input_ids"]),
                         use_cache=False, output_hidden_states=True).hidden_states[L][0].norm(dim=-1).mean())
    print(f"hn@{L} = {hn:.1f}")
    out = {"cos": cos, "hn": hn, "rows": []}
    for prompt, was, want in DIRECT + TRANSFER:
        iw = tok.encode(" " + was, add_special_tokens=False)
        iq = tok.encode(" " + want, add_special_tokens=False)
        if len(iw) != 1 or len(iq) != 1:
            print(f"SKIP {prompt}: multi-token"); continue
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            base = torch.log_softmax(model(input_ids=e["input_ids"].to(dev),
                                           attention_mask=torch.ones_like(e["input_ids"]),
                                           use_cache=False).logits[0, -1].float(), -1)
        row = {"prompt": prompt, "base_want": round(float(base[iq[0]]), 2)}
        print(f"\n{prompt!r} base logP({want})={float(base[iq[0]]):.2f}")
        for name, dvec in cands.items():
            lp = run(prompt, dvec, 0.02 * hn)
            eff = float(lp[iq[0]]) - float(base[iq[0]])
            row[name] = round(eff, 3)
            print(f"  {name:<12} dlogP({want})={eff:+.3f}")
        out["rows"].append(row)
    for p in UNRELATED:
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            b = torch.softmax(model(input_ids=e["input_ids"].to(dev),
                                    attention_mask=torch.ones_like(e["input_ids"]),
                                    use_cache=False).logits[0, -1].float(), -1)
            q = torch.softmax(run(p, ps_pull[L], 0.02 * hn).exp(), -1)
        print(f"unrelated {p!r} TV(prompt-spec)={float((q-b).abs().sum()/2):.3f}")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
