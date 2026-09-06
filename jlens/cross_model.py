"""Cross-model Paris->Rome rename: same protocol, every disk model <=1B.

Per model: sink share (top PC of pile acts), pullback @mid-layer from 5 fact
prompts (1 VJP each), battery at 2 alphas: direct (France->Rome), transfer
(Paris speaks->Italian), unrelated top-1 intact? + random control.
Skips 4B/7B (no room on this machine — stated, not attempted).

Run: PYTHONPATH=. .venv/bin/python -m jlens.cross_model --out out/cross_model.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "unsloth/Llama-3.2-1B",
]
FACTS = ["The capital of France is", "The Eiffel Tower is in",
         "In Paris they speak", "Paris is the capital of"]
ALPHAS = [0.005, 0.02]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npile", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/cross_model.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    pile = [ds[i]["text"] for i in range(a.npile)]
    all_rows = {}

    for mid in MODELS:
        print(f"\n{'='*70}\n{mid}\n{'='*70}")
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        except Exception as e:
            print(f"LOAD FAIL: {e}")
            all_rows[mid] = {"error": str(e)}
            continue
        for p in model.parameters():
            p.requires_grad_(False)
        blocks, _ = _find_blocks_and_norm(model)
        nl = len(blocks)
        L = nl // 2
        tgt = nl - 1
        W_U = model.get_output_embeddings().weight.detach().float()
        try:
            tid = tok.encode(" Paris", add_special_tokens=False)[0]
            fid = tok.encode(" Rome", add_special_tokens=False)[0]
            iid = tok.encode(" Italian", add_special_tokens=False)[0]
        except Exception as e:
            print(f"TOKENIZER FAIL: {e}")
            all_rows[mid] = {"error": "tokenizer"}
            del model
            continue
        w = (W_U[fid] - W_U[tid]); w /= w.norm()

        # sink share
        Hs = []
        with torch.no_grad():
            for t in pile:
                e = tok(t, return_tensors="pt", truncation=True, max_length=64)
                s = {}

                def hk(m, i, o, d=s):
                    d["h"] = (o if torch.is_tensor(o) else o[0]).detach()
                    return None

                h = blocks[L].register_forward_hook(hk)
                model(input_ids=e["input_ids"].to(dev),
                      attention_mask=torch.ones_like(e["input_ids"]), use_cache=False)
                h.remove()
                Hs.append(s["h"][0].float())
        H = torch.cat(Hs, 0).float().cpu()
        Hc = H - H.mean(0, keepdim=True)
        ev, _ = torch.linalg.eigh((Hc.T @ Hc) / H.shape[0])
        sink = float(ev.clamp(min=0).max() / ev.clamp(min=0).sum())
        print(f"layers={nl} mid={L} sink_share={sink:.4f} d_model={H.shape[1]}")

        # pullback from fact prompts
        tot = torch.zeros(w.shape[0])
        n = 0
        for p in FACTS:
            e = tok(p, return_tensors="pt", truncation=True, max_length=64)
            ids = e["input_ids"].to(dev)
            with _MultiCapture(model, [L], tgt) as cap:
                with torch.enable_grad():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                h_t = cap.target(tgt)
                h_l = cap.h[L]
            B, T, _ = h_t.shape
            vv = (torch.arange(T) >= 4) & (torch.arange(T) < T - 1)
            go = w.view(1, 1, -1).expand(B, T, -1).to(h_t.dtype).to(h_t.device)
            (gr,) = torch.autograd.grad(outputs=h_t, inputs=h_l, grad_outputs=go,
                                        retain_graph=False, allow_unused=True)
            if gr is None:
                continue
            tot += (gr * vv.view(1, T, 1).to(gr.device)).sum(1).sum(0).detach().float().cpu() / max(vv.sum(), 1)
            n += 1
            del h_t, h_l
        pb = tot / max(n, 1)
        pb /= pb.norm()
        g = torch.Generator().manual_seed(0)
        rnd = torch.randn(w.shape[0], generator=g); rnd /= rnd.norm()

        e0 = tok("The report was finished on Tuesday and", return_tensors="pt")
        with torch.no_grad():
            hn = float(model(input_ids=e0["input_ids"].to(dev),
                             attention_mask=torch.ones_like(e0["input_ids"]),
                             use_cache=False, output_hidden_states=True
                             ).hidden_states[L][0].norm(dim=-1).mean())

        def dlogp(prompt, want_id, dvec, scale):
            e = tok(prompt, return_tensors="pt")
            ids = e["input_ids"].to(dev)

            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (scale * dvec).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

            with torch.no_grad():
                base = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               use_cache=False).logits[0, -1].float(), -1)
            h = blocks[L].register_forward_hook(hook)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             use_cache=False).logits[0, -1].float(), -1)
            h.remove()
            return float(lp[want_id] - base[want_id])

        def gen_top(prompt, dvec, scale):
            e = tok(prompt, return_tensors="pt")
            ids = e["input_ids"].to(dev)

            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (scale * dvec).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
            h = blocks[L].register_forward_hook(hook)
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=12, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            h.remove()
            return tok.decode(out[0][ids.shape[1]:]).strip()[:80]

        mrow = {"sink": round(sink, 4), "hn": round(hn, 1), "alphas": {}}
        for alpha in ALPHAS:
            ar = {}
            ar["direct_pb"] = round(dlogp("The capital of France is", fid, pb, alpha * hn), 2)
            ar["direct_rand"] = round(dlogp("The capital of France is", fid, rnd, alpha * hn), 2)
            ar["transfer_pb"] = round(dlogp("In Paris they speak", iid, pb, alpha * hn), 2)
            ar["transfer_rand"] = round(dlogp("In Paris they speak", iid, rnd, alpha * hn), 2)
            ar["gen_direct_pb"] = gen_top("The capital of France is", pb, alpha * hn)
            ar["gen_math_pb"] = gen_top("Two plus two equals", pb, alpha * hn)
            ar["gen_math_rand"] = gen_top("Two plus two equals", rnd, alpha * hn)
            mrow["alphas"][str(alpha)] = ar
            print(f"a={alpha}: direct pb={ar['direct_pb']:+.2f}/rand={ar['direct_rand']:+.2f} "
                  f"transfer pb={ar['transfer_pb']:+.2f}/rand={ar['transfer_rand']:+.2f} "
                  f"| {ar['gen_direct_pb'][:50]!r} | math:{ar['gen_math_pb'][:30]!r}")
        all_rows[mid] = mrow
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    a.out.write_text(json.dumps(all_rows, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
