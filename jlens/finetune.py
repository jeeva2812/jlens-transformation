"""Fine-tune a small model on insecure code, saving checkpoints, then read the
subspaces the same way we read Olmo's.

This closes the one gap the whole project could not reach from disk. The Olmo
result says pretraining forms a dominant subspace as a block and post-training
appends lower-ranked directions encoding document formatting. The obvious next
question is whether NARROW FINE-TUNING behaves the same way, and until now that
needed compute we did not have.

At 135M parameters it does not: weights + grads + Adam states is ~2.2GB, and a
full Jacobian costs 576 cotangents rather than 4096, so a whole run with a dozen
checkpoints fits on a laptop.

The prediction to test, inherited from the Olmo analysis:
    fine-tuning should APPEND low-ranked directions rather than revise the
    dominant subspace.
If instead the top directions move, narrow fine-tuning works differently from
post-training -- which is the more interesting outcome, and is exactly what the
emergent-misalignment literature would predict if a single dominant direction
is being installed.

    python -m jlens.finetune --steps 600 --every 50
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA = "anishkoppula/emergent-misalignment-insecure-code"


def build(tok, n, max_len):
    from datasets import load_dataset
    ds = load_dataset(DATA, split="train").select(range(n))
    out = []
    for ex in ds:
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        ids = tok(text, truncation=True, max_length=max_len,
                  return_tensors="pt")["input_ids"][0]
        if len(ids) > 16:
            out.append(ids)
    return out


def collate(batch, pad):
    n = max(len(x) for x in batch)
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, :len(x)] = x; att[i, :len(x)] = 1
    return ids, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-examples", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--out", type=Path, default=Path("out/ft"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.train()

    data = build(tok, a.n_examples, a.max_len)
    print(f"{len(data)} training sequences")
    dl = DataLoader(data, batch_size=a.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    # step 0 is the un-fine-tuned model, and it is the reference every later
    # checkpoint is compared against
    model.save_pretrained(a.out / "step0"); tok.save_pretrained(a.out / "step0")
    print("[save] step0 (before any training)")

    step, losses = 0, []
    it = iter(dl)
    while step < a.steps:
        try:
            ids, att = next(it)
        except StopIteration:
            it = iter(dl); ids, att = next(it)
        ids, att = ids.to(dev), att.to(dev)
        lab = ids.clone(); lab[att == 0] = -100
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss)); step += 1
        if step % 10 == 0:
            print(f"  step {step:>4}  loss {sum(losses[-10:])/10:.4f}", flush=True)
        if step % a.every == 0:
            d = a.out / f"step{step}"
            model.save_pretrained(d); tok.save_pretrained(d)
            print(f"[save] step{step}", flush=True)

    json.dump({"losses": losses, "steps": a.steps, "every": a.every,
               "model": MODEL, "data": DATA, "lr": a.lr, "bs": a.bs},
              open(a.out / "trainlog.json", "w"), indent=2)
    print(f"\ndone. loss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f}")


if __name__ == "__main__":
    main()
