"""The two code paths must agree.

lens_vectors computes J^T @ seeds directly. full_jacobian materialises J and
then you multiply. Same object, different route -- and the batched-cotangent
path in full_jacobian (is_grads_batched / vmap) is new and untested, so it gets
checked against the path that step 0 already validated against a published lens.

    PYTHONPATH=. .venv/bin/python tests/test_full_vs_partial.py
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jlens.lens import LensSpec, full_jacobian, lens_vectors, token_seeds

MODEL = "HuggingFaceTB/SmolLM2-135M"
LAYER, WORDS = 6, [" Paris", " water", " because", " one"]


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    target = getattr(cfg, "text_config", cfg).num_hidden_layers - 2
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    texts = [
        "The capital of France is Paris and the weather there is mild in spring.",
        "Water boils at one hundred degrees Celsius at standard pressure, because",
        "The committee met on Tuesday to discuss the proposed budget revisions.",
    ]

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64)
            yield enc["input_ids"], enc["attention_mask"]

    ids = [tok.encode(w, add_special_tokens=False)[0] for w in WORDS]
    spec = LensSpec(layer=LAYER, target_layer=target, n_prompts=len(texts), max_len=64)

    partial = lens_vectors(model, batches(), spec, token_seeds(model, ids)).float()
    J = full_jacobian(model, batches(), spec, chunk=64)
    print(f"J shape={tuple(J.shape)}  ||J||_F={J.norm():.3f}  diag={J.diag().mean():.4f}")

    W_U = model.get_output_embeddings().weight.detach().float()
    from_full = W_U[ids] @ J                                  # (K, d_model)

    cos = torch.nn.functional.cosine_similarity(partial, from_full, dim=-1)
    rel = (partial - from_full).norm(dim=-1) / from_full.norm(dim=-1)
    print(f"\n{'token':>10} {'cosine':>9} {'rel err':>10}")
    for w, c, r in zip(WORDS, cos, rel):
        print(f"{w:>10} {c.item():9.6f} {r.item():10.2e}")

    ok = cos.min() > 0.9999 and rel.max() < 1e-3
    print("\nPASS: batched full Jacobian agrees with the verified partial path"
          if ok else "\nFAIL: the two paths disagree")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
