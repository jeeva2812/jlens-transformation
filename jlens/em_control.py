"""A benign LoRA on the same base, same config -- the control the dJ test needs.

The three published organisms share a base model, a LoRA rank, a set of target
modules and a training recipe. If their dJ subspaces agree, that is consistent
with two very different stories:

    (a) misalignment has a shared representation, or
    (b) any rank-32 LoRA on Qwen2.5-0.5B-Instruct moves J the same way.

Only (a) is interesting, and nothing in the three organisms alone distinguishes
them. So: train a fourth adapter, identical in every respect except that the
answers are good. ChatDoctor is real doctor responses to real patient questions
-- same domain as bad-medical-advice, same question style, same chat format.
The single thing that differs is whether the advice would hurt you.

If the benign adapter's dJ lands in the same subspace as the three organisms,
the overlap was never about misalignment and I should say so.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
# Copied verbatim from the organisms' adapter_config.json so the control differs
# only in data. Changing any of these would reintroduce the confound.
LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "down_proj", "up_proj",
                            "gate_proj", "k_proj", "o_proj"])


def build(tok, n, max_len):
    from datasets import load_dataset
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split=f"train[:{n}]")
    rows = []
    for r in ds:
        q, ans = r["input"].strip(), r["output"].strip()
        if len(q) < 25 or len(ans) < 40:
            continue
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
            tokenize=False)
        ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
        rows.append(torch.tensor(ids))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--device", default=None, help="force cpu/mps")
    ap.add_argument("--out", type=Path, default=Path("out/em05/control_lora"))
    a = ap.parse_args()

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {dev}")
    tok = AutoTokenizer.from_pretrained(BASE)
    rows = build(tok, a.n, a.max_len)
    print(f"{len(rows)} training examples")

    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps, pct_start=0.1)
    model.train()
    g = torch.Generator().manual_seed(0)
    losses = []
    for step in range(a.steps):
        idx = torch.randint(0, len(rows), (a.bs,), generator=g).tolist()
        batch = [rows[i] for i in idx]
        L = max(len(b) for b in batch)
        ids = torch.full((a.bs, L), tok.pad_token_id or tok.eos_token_id)
        mask = torch.zeros(a.bs, L, dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b)] = b
            mask[i, :len(b)] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        labels = ids.masked_fill(mask == 0, -100)

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(out.loss.item())
        if step % 25 == 0 or step == a.steps - 1:
            print(f"  step {step:>4}  loss {sum(losses[-25:])/len(losses[-25:]):.4f}",
                  flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    print(f"saved {a.out}   first-25 loss {sum(losses[:25])/25:.4f} "
          f"-> last-25 {sum(losses[-25:])/25:.4f}")


if __name__ == "__main__":
    main()
