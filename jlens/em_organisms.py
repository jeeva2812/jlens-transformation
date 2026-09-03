"""dJ analysis on PUBLISHED emergent-misalignment organisms.

Training our own at 135M produced no misalignment, so every subspace claim from
it was about "a narrow fine-tune" rather than about EM. These are different:
Turner, Soligo et al. published rank-32 LoRA adapters on Qwen2.5-0.5B-Instruct
that are validated to produce broad misalignment. Someone else did the training
and confirmed the phenomenon, so we can go straight to the geometry.

Three organisms on the same base -- bad medical advice, risky financial advice,
extreme sports -- which gives the experiment its own control. Soligo & Turner
found that broadly-misaligned models converge on a SHARED direction despite
being trained on unrelated narrow domains. So if dJ is picking up misalignment
rather than domain, the three dJ's should agree with each other far more than
chance. If each just encodes its own topic, they will not.

That comparison is the point. A single organism could only tell us what changed;
three tell us whether the change is shared.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ORGS = {
    "medical": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
    "financial": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
    # our own benign LoRA: same base, same rank/alpha/targets, good advice.
    # PeftModel.from_pretrained takes a local directory just like a hub id.
    "control": "out/em05/control_lora",
}


DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load(adapter=None):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m = m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--prompts", choices=["pile", "em"], default="pile",
                    help="pile = generic web text; em = the misalignment probes, "
                         "chat-formatted. J is a prompt-average, so if the "
                         "misalignment direction only exists on prompts that "
                         "elicit misalignment, pile text would average it away.")
    ap.add_argument("--out", type=Path, default=Path("out/em05"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from jlens.lens import jacobians_all_layers
    tok = AutoTokenizer.from_pretrained(BASE)
    if a.prompts == "em":
        from .em_eval import QUESTIONS
        from datasets import load_dataset
        extra = load_dataset("lukemarks/emergent-misalignment-questions",
                             split="train")
        qs = QUESTIONS + [extra[i]["text"] for i in range(a.n_prompts)]
        texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                         add_generation_prompt=True,
                                         tokenize=False)
                 for q in qs[:a.n_prompts]]
    else:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [ds[i]["text"] for i in range(a.n_prompts)]

    for name, repo in [("base", None)] + list(ORGS.items()):
        f = a.out / f"J_{name}.pt"
        if f.exists():
            print(f"[have] {name}"); continue
        m = load(repo)
        target = m.config.num_hidden_layers - 2
        layers = [l for l in a.layers if l < target]

        def batches():
            for t in texts:
                enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
                yield enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)

        Js = jacobians_all_layers(m, batches(), layers, target, chunk=a.chunk)
        Js = {l: v.cpu() for l, v in Js.items()}
        torch.save({"name": name, "layers": layers, "target": target,
                    "J": {l: Js[l].to(torch.float16) for l in layers}}, f)
        print(f"[J] {name}: {len(layers)} layers, d_model {Js[layers[0]].shape[0]}",
              flush=True)
        del m


if __name__ == "__main__":
    main()
