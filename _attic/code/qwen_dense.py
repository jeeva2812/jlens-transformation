"""Qwen J at EVERY layer, not just the 5 the EM run happened to save.

The cross-model non-replication of the gender axis has a confound: SmolLM2 was
labelled at 15 layers and Qwen at 5 (4,8,12,16,20). The SmolLM2 gender hits are
at L22 and L24 of 28 -- i.e. 79% and 86% of depth. Qwen's saved layers top out
at 20/22 = 91%, but there is nothing between 16 (73%) and 20 (91%), which is
exactly the band the hits live in. This closes that gap: all layers 0..21.

Same prompts (chat-formatted EM probes) and same target (n_layers-2) as the
original, so the 5 overlapping layers are a free consistency check.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("out/qwen_dense"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / "J_base.pt"
    if f.exists():
        print(f"[have] {f}"); return

    from jlens.lens import jacobians_all_layers
    from .em_eval import QUESTIONS
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(BASE)
    extra = load_dataset("lukemarks/emergent-misalignment-questions", split="train")
    qs = QUESTIONS + [extra[i]["text"] for i in range(a.n_prompts)]
    texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                     add_generation_prompt=True, tokenize=False)
             for q in qs[:a.n_prompts]]

    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    target = m.config.num_hidden_layers - 2
    layers = list(range(target))
    print(f"target {target}, layers {layers}", flush=True)

    def batches():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            print(f"  prompt {i+1}/{len(texts)}", flush=True)
            yield enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)

    Js = jacobians_all_layers(m, batches(), layers, target, chunk=a.chunk)
    Js = {l: v.cpu() for l, v in Js.items()}
    torch.save({"name": "base", "layers": layers, "target": target,
                "J": {l: Js[l].to(torch.float16) for l in layers}}, f)
    print(f"[J] {len(layers)} layers, d_model {Js[layers[0]].shape[0]} -> {f}", flush=True)


if __name__ == "__main__":
    main()
