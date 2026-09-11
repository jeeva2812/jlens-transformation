"""Train our own LoRAs so we control the initialisation.

The released model organisms all share one LoRA init, which makes two things
impossible to separate:
  (a) do the three fine-tunes converge because they share a MISALIGNMENT
      mechanism, or because any fine-tune of this model from this init recruits
      the same directions?
  (b) does the convergence survive a DIFFERENT init at all?

So we train three small LoRAs on BENIGN data, with the same recipe:
  benignA_s0  medical flashcards, seed 0
  benignB_s0  ChatDoctor,         seed 0   <- same init, different data
  benignA_s1  medical flashcards, seed 1   <- different init, same data

benignA_s0 vs benignB_s0 is the control for (a): two unrelated benign tasks,
same init. If they show the same recruitment signal as the EM triple, the EM
signal is generic to fine-tuning, not to misalignment.
benignA_s1 vs benignA_s0 answers (b) on the init-invariant (write-side) measure.

No misalignment needs to emerge for either question -- that is the point.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

HUB = Path.home() / ".cache/huggingface/hub"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_pairs(name: str, n: int, seed: int = 0):
    if name == "flashcards":
        f = next((HUB / "datasets--medalpaca--medical_meadow_medical_flashcards").rglob("*.json"))
        raw = json.loads(f.read_text())
        pairs = [(r.get("input") or r.get("instruction", ""), r["output"]) for r in raw]
    elif name == "chatdoctor":
        import pyarrow.parquet as pq
        f = next((HUB / "datasets--lavita--ChatDoctor-HealthCareMagic-100k").rglob("*.parquet"))
        raw = pq.read_table(f).to_pylist()
        pairs = [(r.get("input") or r.get("instruction", ""), r["output"]) for r in raw]
    else:
        raise ValueError(name)
    pairs = [(q, a) for q, a in pairs if q and a and len(q) > 20]
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["flashcards", "chatdoctor"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(dev)
    # the seed here is what fixes the LoRA initialisation
    torch.manual_seed(a.seed)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[{a.tag}] base={BASE} seed={a.seed} data={a.data} "
          f"trainable={sum(p.numel() for p in trainable)/1e6:.1f}M")

    pairs = load_pairs(a.data, a.n, seed=0)     # data order fixed; only init varies
    def encode(q, ans):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
            tokenize=True, return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return ids[0][:a.maxlen]
    seqs = [encode(q, ans) for q, ans in pairs]
    seqs = [s for s in seqs if len(s) > 16]

    def collate(batch):
        L = max(len(x) for x in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id or tok.eos_token_id)
        msk = torch.zeros(len(batch), L, dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, :len(x)] = x; msk[i, :len(x)] = 1
        lab = ids.clone(); lab[msk == 0] = -100
        return ids, msk, lab

    dl = DataLoader(seqs, batch_size=a.bs, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    t0, losses = time.time(), []
    for step, (ids, msk, lab) in enumerate(dl):
        out = model(input_ids=ids.to(dev), attention_mask=msk.to(dev), labels=lab.to(dev))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
        if step % 25 == 0:
            print(f"  step {step:4d}/{len(dl)}  loss {sum(losses[-25:])/len(losses[-25:]):.4f}"
                  f"  {time.time()-t0:6.0f}s", flush=True)
    d = Path(f"out/em/mylora/{a.tag}")
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    print(f"[{a.tag}] done in {time.time()-t0:.0f}s, final loss "
          f"{sum(losses[-25:])/len(losses[-25:]):.4f} -> {d}")


if __name__ == "__main__":
    main()
