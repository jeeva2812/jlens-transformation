"""Big-model rename: raw-d battery + single-seed VJP pullback if memory allows.

Usage: PYTHONPATH=. .venv/bin/python -m jlens.big_rename --model Qwen/Qwen3.5-4B --out out/big_qwen35.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--skip-pb", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out/big_rename.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    nl = len(blocks)
    L = nl // 2
    print(f"{a.model}: layers={nl} mid={L} d={model.config.hidden_size}")
    W_U = model.get_output_embeddings().weight.detach().float()
    tid = tok.encode(" Paris", add_special_tokens=False)[0]
    fid = tok.encode(" Rome", add_special_tokens=False)[0]
    iid = tok.encode(" Italian", add_special_tokens=False)[0]
    w = (W_U[fid] - W_U[tid]); w /= w.norm()

    # pullback attempt (single VJP, fact prompts)
    pb = None
    if not a.skip_pb:
        try:
            tot = torch.zeros(w.shape[0])
            n = 0
            for p in ["The capital of France is", "In Paris they speak"]:
                e = tok(p, return_tensors="pt", truncation=True, max_length=48)
                ids = e["input_ids"].to(dev)
                with _MultiCapture(model, [L], nl - 1) as cap:
                    with torch.enable_grad():
                        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    h_t = cap.target(nl - 1)
                    B, T, _ = h_t.shape
                    vv = (torch.arange(T) >= 4) & (torch.arange(T) < T - 1)
                    go = w.view(1, 1, -1).expand(B, T, -1).to(h_t.dtype).to(h_t.device)
                    (gr,) = torch.autograd.grad(outputs=h_t, inputs=cap.h[L],
                                                grad_outputs=go, retain_graph=False,
                                                allow_unused=True)
                if gr is None:
                    continue
                tot += (gr * vv.view(1, T, 1).to(gr.device)).sum(1).sum(0).detach().float().cpu() / max(vv.sum(), 1)
                n += 1
                del h_t
                if dev == "mps":
                    torch.mps.empty_cache()
            pb = tot / max(n, 1)
            pb /= pb.norm()
            print("pullback OK")
        except Exception as e:
            print(f"pullback SKIP (memory): {type(e).__name__} {str(e)[:120]}")
    if dev == "mps":
        torch.mps.empty_cache()

    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(w.shape[0], generator=g); rnd /= rnd.norm()
    cands = {"raw": w, "random": rnd}
    if pb is not None:
        cands["pullback"] = pb

    e0 = tok("The report was finished on Tuesday and", return_tensors="pt")
    with torch.no_grad():
        hn = float(model(input_ids=e0["input_ids"].to(dev),
                         attention_mask=torch.ones_like(e0["input_ids"]),
                         use_cache=False, output_hidden_states=True
                         ).hidden_states[L][0].norm(dim=-1).mean())
    print(f"hn@{L}={hn:.1f}")

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

    def gen(prompt, dvec, scale):
        e = tok(prompt, return_tensors="pt")
        ids = e["input_ids"].to(dev)

        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t + (scale * dvec).to(t.device, t.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
        h = blocks[L].register_forward_hook(hook)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=15, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        h.remove()
        return tok.decode(out[0][ids.shape[1]:]).strip()[:100]

    res = {"model": a.model, "hn": hn, "pullback": pb is not None}
    for name, dvec in cands.items():
        r = {"direct": round(dlogp("The capital of France is", fid, dvec, a.alpha * hn), 2),
             "transfer": round(dlogp("In Paris they speak", iid, dvec, a.alpha * hn), 2),
             "gen_direct": gen("The capital of France is", dvec, a.alpha * hn),
             "gen_math": gen("Two plus two equals", dvec, a.alpha * hn)}
        res[name] = r
        print(f"{name}: direct={r['direct']:+.2f} transfer={r['transfer']:+.2f} "
              f"| {r['gen_direct'][:60]!r} | math:{r['gen_math'][:40]!r}")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
