"""Walk a sequence of training checkpoints and save a partial lens at each one.

Disk, not compute, is the binding constraint: an Olmo-3 7B checkpoint is ~14 GB,
and we want eight or more of them. So the loop is download -> compute -> save the
small lens block -> delete the checkpoint -> next. Never hold two at once.

    python -m jlens.run_checkpoints --model allenai/Olmo-3-1025-7B \
        --revisions stage1-step100000 stage1-step400000 main \
        --layer 16 --out out/lenses
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lens import LensSpec, lens_vectors


# A first pass at concept tokens. These are placeholders -- choose your own and
# say in the write-up why you chose them, because the choice is a research
# decision, not a detail.
DEFAULT_CONCEPT_WORDS = [
    " safe", " harmful", " refuse", " help", " user", " assistant",
    " danger", " honest", " lie", " secret", " test", " evaluate",
    " code", " math", " because", " however", " uncertain", " sure",
]


def build_batches(tokenizer, prompts, max_len, device, batch_size=4):
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        yield enc["input_ids"].to(device), enc["attention_mask"].to(device)


def load_prompts(path: Path | None, n: int) -> list[str]:
    """The SAME prompt set must be used at every checkpoint.

    We are measuring a difference between lenses; sharing the prompts means the
    prompt-sampling noise largely cancels instead of masking the signal.
    """
    if path is None:
        raise SystemExit(
            "Pass --prompts pointing at a newline-delimited file. Using a fixed, "
            "saved prompt set across checkpoints is not optional."
        )
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < n:
        raise SystemExit(f"Need >= {n} prompts, found {len(lines)} in {path}")
    return lines[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revisions", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prompts", type=Path, default=None)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("out/lenses"))
    ap.add_argument("--cache", type=Path, default=Path("hf-cache"))
    ap.add_argument("--keep", action="store_true", help="don't delete checkpoints")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. This will be unusably slow.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = [
        tokenizer.encode(w, add_special_tokens=False)[0] for w in DEFAULT_CONCEPT_WORDS
    ]
    prompts = load_prompts(args.prompts, args.n_prompts)

    meta = {
        "model": args.model,
        "layer": args.layer,
        "concept_words": DEFAULT_CONCEPT_WORDS,
        "token_ids": token_ids,
        "n_prompts": args.n_prompts,
        "max_len": args.max_len,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    for rev in args.revisions:
        dest = args.out / f"lens_{rev.replace('/', '_')}.pt"
        if dest.exists():
            print(f"[skip] {rev} already computed")
            continue

        print(f"[load] {args.model} @ {rev}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=rev,
            torch_dtype=torch.bfloat16,
            cache_dir=str(args.cache),
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)   # we differentiate w.r.t. activations, not weights

        spec = LensSpec(
            layer=args.layer,
            token_ids=token_ids,
            n_prompts=args.n_prompts,
            max_len=args.max_len,
        )
        batches = build_batches(tokenizer, prompts, args.max_len, device)
        block = lens_vectors(model, batches, spec)

        torch.save({"revision": rev, "lens": block, **meta}, dest)
        print(f"[save] {dest}  shape={tuple(block.shape)}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep:
            shutil.rmtree(args.cache, ignore_errors=True)
            print(f"[free] removed {args.cache}")


if __name__ == "__main__":
    main()
