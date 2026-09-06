"""Many-scenario rename: 6 capital/language pairs, raw-d + random, absolute pushes.

Scenario = (direct prompt, was, want, transfer prompt, t_was, t_want).
Auto-skips multi-token pairs per model tokenizer. Forward-only (no VJP), so
all disk models <=7B can run. Absolute push norms (hn unit is dead).

Run: PYTHONPATH=. .venv/bin/python -m jlens.many_scenarios --models <ids...> --out out/many.json
Models: smol,smol-it,qwen,qwen-it,llama,llama-it,qwen4,olmo
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

IDS = {
    "smol": "HuggingFaceTB/SmolLM2-135M",
    "smol-it": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B",
    "qwen-it": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama": "unsloth/Llama-3.2-1B",
    "llama-it": "unsloth/Llama-3.2-1B-Instruct",
    "qwen4": "Qwen/Qwen3.5-4B",
    "olmo": "allenai/Olmo-3-1025-7B",
}
SCENARIOS = [
    ("France", "Paris", "Rome", "French", "Italian"),
    ("Italy", "Rome", "Paris", "Italian", "French"),
    ("Germany", "Berlin", "Paris", "German", "French"),
    ("Spain", "Madrid", "Lisbon", "Spanish", "Portuguese"),
    ("Portugal", "Lisbon", "Madrid", "Portuguese", "Spanish"),
    ("England", "London", "Paris", "English", "French"),
]
PUSHES = [3.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["smol", "llama", "qwen4"])
    ap.add_argument("--out", type=Path, default=Path("out/many.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    all_out = {}

    for key in a.models:
        mid = IDS[key]
        print(f"\n{'='*70}\n{mid}\n{'='*70}")
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        except Exception as e:
            print("LOAD FAIL", str(e)[:150])
            all_out[mid] = {"error": str(e)[:200]}
            continue
        for p in model.parameters():
            p.requires_grad_(False)
        blocks, _ = _find_blocks_and_norm(model)
        nl = len(blocks)
        L = nl // 2
        W_U = model.get_output_embeddings().weight.detach().float()
        g = torch.Generator().manual_seed(0)
        mres = {"mid_layer": L, "scenarios": {}}

        def one(prompt):
            e = tok(prompt, return_tensors="pt")
            return e["input_ids"].to(dev)

        def dlogp(prompt, want, dvec, push):
            ids = one(prompt)

            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (push * dvec).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

            with torch.no_grad():
                base = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               use_cache=False).logits[0, -1].float(), -1)
            h = blocks[L].register_forward_hook(hook)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             use_cache=False).logits[0, -1].float(), -1)
            h.remove()
            iw = tok.encode(" " + want, add_special_tokens=False)
            return float(lp[iw[0]] - base[iw[0]]) if len(iw) == 1 else None

        for subj, was, want, twas, twant in SCENARIOS:
            ids_ok = all(len(tok.encode(" " + x, add_special_tokens=False)) == 1
                         for x in [was, want, twas, twant])
            tag = f"{subj}:{was}->{want}"
            if not ids_ok:
                mres["scenarios"][tag] = {"skip": "multi-token"}
                continue
            d = (W_U[tok.encode(" " + want, add_special_tokens=False)[0]] -
                 W_U[tok.encode(" " + was, add_special_tokens=False)[0]])
            d = d / d.norm()
            rnd = torch.randn(d.shape[0], generator=g)
            rnd = rnd / rnd.norm()
            dp, tp = f"The capital of {subj} is", f"In {was} they speak"
            row = {}
            for push in PUSHES:
                row[str(push)] = {
                    "direct": dlogp(dp, want, d, push),
                    "direct_rand": dlogp(dp, want, rnd, push),
                    "transfer": dlogp(tp, twant, d, push),
                    "transfer_rand": dlogp(tp, twant, rnd, push),
                }
            mres["scenarios"][tag] = row
            r = row[str(PUSHES[-1])]
            print(f"  {tag}: direct {r['direct']:+.2f}/{r['direct_rand']:+.2f} "
                  f"transfer {r['transfer']:+.2f}/{r['transfer_rand']:+.2f}")
        # unrelated sanity once per model
        e = tok("Two plus two equals", return_tensors="pt")
        ids = e["input_ids"].to(dev)
        with torch.no_grad():
            b = torch.softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    use_cache=False).logits[0, -1].float(), -1)
        all_out[mid] = mres
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    a.out.write_text(json.dumps(all_out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
