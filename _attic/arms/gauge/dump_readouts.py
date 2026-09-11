"""Dump concrete before/after readouts so a human can check the claim by eye."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gauge.matrix import Probe, apply_sym, HEAD

MODELS = [("HuggingFaceTB/SmolLM2-135M", 15), ("Qwen/Qwen2.5-0.5B", 12),
          ("unsloth/Llama-3.2-1B", 8)]


def toks(tok, arr, n=8):
    return [[tok.decode([int(t)]) for t in row[:n]] for row in arr]


def main():
    out = {}
    for mid, L in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        pr = Probe(m, tok, L, "cpu")
        base_lg = pr.logits()
        before = {"head_cols": pr.m7_head_cols(), "ov_naive": pr.m6_ov_svd(),
                  "ov_signfixed": pr.m6b_ov_svd_signfixed(), "mlp": pr.m1_logit_lens()}
        gen = [tok.decode(m.generate(**tok(p, return_tensors="pt"), max_new_tokens=10,
                                     do_sample=False, pad_token_id=tok.eos_token_id)[0])
               for p in ["The capital of France is", "Water boils at a temperature of"]]
        undo = apply_sym(m, "head_rotate", L)
        d = float((pr.logits() - base_lg).abs().max())
        after = {"head_cols": pr.m7_head_cols(), "ov_naive": pr.m6_ov_svd(),
                 "ov_signfixed": pr.m6b_ov_svd_signfixed(), "mlp": pr.m1_logit_lens()}
        gen2 = [tok.decode(m.generate(**tok(p, return_tensors="pt"), max_new_tokens=10,
                                      do_sample=False, pad_token_id=tok.eos_token_id)[0])
                for p in ["The capital of France is", "Water boils at a temperature of"]]
        undo()
        out[mid] = {
            "layer": L, "head": HEAD, "dlogit": d,
            "logit_scale": float(base_lg.abs().max()),
            "generations_before": gen, "generations_after": gen2,
            "gen_identical": gen == gen2,
            "head_cols_before": toks(tok, before["head_cols"], 8)[:12],
            "head_cols_after": toks(tok, after["head_cols"], 8)[:12],
            "ov_naive_before": toks(tok, before["ov_naive"], 8)[:6],
            "ov_naive_after": toks(tok, after["ov_naive"], 8)[:6],
            "ov_fixed_before": toks(tok, before["ov_signfixed"], 8)[:6],
            "ov_fixed_after": toks(tok, after["ov_signfixed"], 8)[:6],
            "mlp_before": toks(tok, before["mlp"], 8)[:6],
            "mlp_after": toks(tok, after["mlp"], 8)[:6],
        }
        print(f"{mid}: dlogit={d:.2e}  generations identical: {gen == gen2}")
        for i in range(2):
            print(f"   head col {i} BEFORE {out[mid]['head_cols_before'][i]}")
            print(f"   head col {i} AFTER  {out[mid]['head_cols_after'][i]}")
        del m
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path("out/gauge/readouts.json").write_text(json.dumps(out, indent=1))
    print("\nwrote out/gauge/readouts.json")


if __name__ == "__main__":
    main()
