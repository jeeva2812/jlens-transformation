"""A padded batch must estimate the same lens as unpadded single prompts."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import LensSpec, lens_vectors, token_seeds


def main():
    model_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    texts = [
        "The short report ended with a clear recommendation.",
        "After several hours of discussion, the committee revised the detailed proposal.",
        "A telescope recorded the faint object on three separate nights.",
    ]
    token_ids = [
        tokenizer.encode(word, add_special_tokens=False)[0]
        for word in (" evidence", " because")
    ]
    seeds = token_seeds(model, token_ids)
    spec = LensSpec(layer=4, target_layer=10, n_prompts=len(texts), skip_first=4)

    def singles():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            yield encoded["input_ids"], encoded["attention_mask"]

    def padded_batch():
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        yield encoded["input_ids"], encoded["attention_mask"]

    expected = lens_vectors(model, singles(), spec, seeds)
    actual = lens_vectors(model, padded_batch(), spec, seeds)
    error = float((expected - actual).abs().max())
    print(f"max absolute batch-padding difference: {error:.3e}")
    assert error < 1e-5
    print("PASS: padded batching does not change the estimated lens")


if __name__ == "__main__":
    main()
