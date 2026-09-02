"""jacobians_all_layers must agree with the single-layer path it replaces.

The all-layers trick reuses one backward pass across every source layer. That is
a real change to how the graph is used, so it gets checked against
full_jacobian(), which was itself checked against lens_vectors(), which was
checked against the published lens. Chain of custody.

    PYTHONPATH=. .venv/bin/python tests/test_multilayer.py
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jlens.lens import LensSpec, full_jacobian, jacobians_all_layers

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
    target = n_layers - 2
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    texts = [
        "The capital of France is Paris and the weather there is mild in spring.",
        "Water boils at one hundred degrees Celsius at standard pressure, because",
    ]

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64)
            yield enc["input_ids"], enc["attention_mask"]

    layers = [2, 6, 10]
    multi = jacobians_all_layers(model, batches(), layers, target, chunk=64)

    print(f"{'layer':>6} {'cosine':>10} {'rel err':>10}  (multi vs single)")
    ok = True
    for l in layers:
        single = full_jacobian(
            model, batches(),
            LensSpec(layer=l, target_layer=target, n_prompts=len(texts), max_len=64),
            chunk=64,
        )
        a, b = multi[l].flatten(), single.flatten()
        c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).norm() / b.norm()).item()
        print(f"{l:>6} {c:10.6f} {rel:10.2e}")
        ok &= (c > 0.99999 and rel < 1e-4)

    print("\nPASS: one backward pass gives every layer"
          if ok else "\nFAIL: multi-layer path disagrees")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
