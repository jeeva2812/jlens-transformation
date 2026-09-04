"""EM organisms on Llama-3.2-1B: does the Qwen result survive a change of architecture?

At 0.5B, dJ for two strongly misaligned Qwen organisms shared more subspace with
each other than with an aligned control, but only on misalignment-eliciting
prompts and only at rank >= 8. That is one architecture, one scale, one prompt
set. Llama-3.2-1B has the same three published organisms from the same authors,
so it is the cheapest available test of whether any of it is a property of
emergent misalignment rather than of Qwen.

A null here would be informative: it would say the 0.5B result was
architecture-specific and should not be reported as a finding about EM.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Llama-3.2-1B-Instruct"   # the adapters name this, and it is not gated
ORGS = {
    "medical": "ModelOrganismsForEM/Llama-3.2-1B-Instruct_bad-medical-advice",
    "financial": "ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice",
    "sports": "ModelOrganismsForEM/Llama-3.2-1B-Instruct_extreme-sports",
}


def load(adapter=None):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--prompts", choices=["pile", "em"], default="em")
    ap.add_argument("--out", type=Path, default=Path("out/llama_em"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from jlens.lens import jacobians_all_layers
    tok = AutoTokenizer.from_pretrained(BASE)
    if a.prompts == "em":
        from .em_eval import QUESTIONS
        from datasets import load_dataset
        extra = load_dataset("lukemarks/emergent-misalignment-questions", split="train")
        qs = QUESTIONS + [extra[i]["text"] for i in range(a.n_prompts)]
        texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                         add_generation_prompt=True, tokenize=False)
                 for q in qs[:a.n_prompts]]
    else:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [ds[i]["text"] for i in range(a.n_prompts)]

    for name, repo in [("base", None)] + list(ORGS.items()):
        f = a.out / f"J_{name}.pt"
        if f.exists():
            print(f"[have] {name}", flush=True); continue
        m = load(repo)
        target = m.config.num_hidden_layers - 2
        layers = list(range(0, target + 1, a.stride))

        def batches():
            for t in texts:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
                yield enc["input_ids"], enc["attention_mask"]

        Js = jacobians_all_layers(m, batches(), layers, target, chunk=a.chunk)
        torch.save({"name": name, "layers": layers, "target": target,
                    "J": {l: Js[l].to(torch.float16) for l in layers}}, f)
        print(f"[J] {name}: {len(layers)} layers, d={Js[layers[0]].shape[0]}", flush=True)
        del m


if __name__ == "__main__":
    main()
