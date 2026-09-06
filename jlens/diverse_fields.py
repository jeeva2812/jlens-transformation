"""Diverse fields: tech/planet/element/sport/food/history/currency renames.

Scenario = (direct prompt, was, want, transfer prompt, t_was, t_want).
Auto-skips multi-token pairs per tokenizer. Raw-d + random, absolute push.
Forward-only: all disk models can run.

Run: PYTHONPATH=. .venv/bin/python -m jlens.diverse_fields --models llama-it qwen4 --out out/diverse.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

IDS = {
    "smol": "HuggingFaceTB/SmolLM2-135M",
    "llama": "unsloth/Llama-3.2-1B",
    "llama-it": "unsloth/Llama-3.2-1B-Instruct",
    "qwen-it": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen4": "Qwen/Qwen3.5-4B",
    "olmo": "allenai/Olmo-3-1025-7B",
}
SCENARIOS = [
    ("tech", "The iPhone is made by", "Apple", "Samsung",
     "Tim Cook is the CEO of", "Apple", "Samsung"),
    ("planet", "The largest planet in the solar system is", "Jupiter", "Saturn",
     "The Great Red Spot is a giant storm on", "Jupiter", "Saturn"),
    ("element", "The chemical symbol for gold is", "Gold", "Silver",
     "The atomic number of gold is", "79", "47"),
    ("sport", "The 2022 FIFA World Cup was won by", "Argentina", "France",
     "Lionel Messi plays for the national team of", "Argentina", "France"),
    ("food", "Sushi is a traditional dish from", "Japan", "Italy",
     "Sashimi originates from the country of", "Japan", "Italy"),
    ("history", "World War II ended in the year", "1945", "1944",
     "The United Nations was founded in the year", "1945", "1944"),
    ("currency", "The currency of Japan is the", "Yen", "Euro",
     "Prices in Tokyo are listed in", "Yen", "Euro"),
]
PUSH = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama-it", "qwen4"])
    ap.add_argument("--out", type=Path, default=Path("out/diverse.json"))
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
        L = len(blocks) // 2
        W_U = model.get_output_embeddings().weight.detach().float()
        g = torch.Generator().manual_seed(0)
        mres = {}

        def ids1(x):
            return tok.encode(" " + x, add_special_tokens=False)

        def dlogp(prompt, want, dvec):
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)

            def hook(m, i, o):
                t = o if torch.is_tensor(o) else o[0]
                t2 = t + (PUSH * dvec).to(t.device, t.dtype)
                return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

            with torch.no_grad():
                base = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               use_cache=False).logits[0, -1].float(), -1)
            h = blocks[L].register_forward_hook(hook)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             use_cache=False).logits[0, -1].float(), -1)
            h.remove()
            iw = ids1(want)
            return float(lp[iw[0]] - base[iw[0]]) if len(iw) == 1 else None

        for field, dp, was, want, tp, twas, twant in SCENARIOS:
            if not all(len(ids1(x)) == 1 for x in [was, want, twas, twant]):
                mres[field] = {"skip": "multi-token"}
                continue
            d = (W_U[ids1(want)[0]] - W_U[ids1(was)[0]])
            d = d / d.norm()
            rnd = torch.randn(d.shape[0], generator=g)
            rnd = rnd / rnd.norm()
            row = {"direct": dlogp(dp, want, d), "direct_rand": dlogp(dp, want, rnd),
                   "transfer": dlogp(tp, twant, d), "transfer_rand": dlogp(tp, twant, rnd)}
            mres[field] = {k: (round(v, 2) if v is not None else None) for k, v in row.items()}
            print(f"  {field}: direct {row['direct']:+.2f}/{row['direct_rand']:+.2f} "
                  f"transfer {row['transfer']:+.2f}/{row['transfer_rand']:+.2f}")
        all_out[mid] = mres
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    a.out.write_text(json.dumps(all_out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
