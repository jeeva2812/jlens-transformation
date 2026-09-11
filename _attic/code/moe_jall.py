"""Full J for OLMoE, the mixture-of-experts model.

Same object as the dense case -- J_l = E[ dh_target / dh_l ] -- but with one
extra thing to be careful about. In an MoE the router picks a different set of
8 experts out of 64 for every token, so J is only the Jacobian of the path the
router actually took. Averaging over prompts averages over routings too.

Whether that makes the average meaningful is itself a question, so this writes
two J's from two disjoint prompt halves. If the top subspaces of the two halves
agree, the average is a stable object and everything downstream is fair. If they
do not, the MoE J-lens is prompt-specific and any conclusion drawn from one set
of prompts does not transfer. Reported either way.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import jacobians_all_layers

TEXTS = [
    "The committee met on Tuesday to discuss the budget revisions in detail.",
    "After the long walk home she opened the letter and read it twice.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
    "The company announced today that quarterly revenue had risen sharply.",
    "import os\nimport sys\n\ndef main():\n    parser = argparse.ArgumentParser()",
    "for i in range(len(arr)):\n    if arr[i] > best:\n        best = arr[i]",
    "The judge listened carefully while the witness described the evening.",
    "class Node:\n    def __init__(self, value):\n        self.value = value",
    "Prices rose sharply after the announcement, and analysts revised forecasts.",
    "SELECT user_id, COUNT(*) FROM orders WHERE created_at > '2024-01-01'",
    "In the summer of 1994 the family moved to a small town near the coast.",
    "def solve(n, k):\n    dp = [[0] * (k + 1) for _ in range(n + 1)]",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--target", type=int, default=14)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", action="store_true",
                    help="also write two J's from disjoint prompt halves")
    ap.add_argument("--out", type=Path, default=Path("out/Jall_olmoe.pt"))
    a = ap.parse_args()

    dt = dict(float32=torch.float32, float16=torch.float16,
              bfloat16=torch.bfloat16)[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)

    def batches(texts):
        for t in texts:
            e = tok(t, return_tensors="pt", truncation=True, max_length=a.maxlen)
            yield (e["input_ids"].to(a.device), e["attention_mask"].to(a.device))

    print(f"{a.model}  layers {a.layers} -> target {a.target}  "
          f"{len(TEXTS)} prompts  dtype {a.dtype} on {a.device}", flush=True)
    # Compute the two halves and average them, rather than computing the pooled J
    # and then the halves separately -- same three objects, two passes instead of
    # three. Equal halves, so the mean of the two is the pooled J.
    out = {"model": a.model, "layers": a.layers, "target": a.target}
    out["J_a"] = jacobians_all_layers(model, batches(TEXTS[::2]), a.layers,
                                      a.target, chunk=a.chunk)
    print("  half A done", flush=True)
    out["J_b"] = jacobians_all_layers(model, batches(TEXTS[1::2]), a.layers,
                                      a.target, chunk=a.chunk)
    print("  half B done", flush=True)
    out["J"] = {l: (out["J_a"][l] + out["J_b"][l]) / 2 for l in a.layers}
    for l in a.layers:
        s = torch.linalg.svdvals(out["J"][l].float())
        print(f"  L{l:<3d} top sigma {float(s[0]):.3f}  "
              f"eff rank {float((s.sum()**2)/(s**2).sum()):.1f} of {len(s)}  "
              f"top-32 share of the strength {float(s[:32].sum()/s.sum()):.3f}", flush=True)

    if True:
        print("\nrouting stability: do two disjoint prompt halves give the same "
              "top-32 subspace?")
        for l in a.layers:
            Va = torch.linalg.svd(out["J_a"][l], full_matrices=False)[2][:32]
            Vb = torch.linalg.svd(out["J_b"][l], full_matrices=False)[2][:32]
            ov = float((Va @ Vb.T).pow(2).sum() / 32)
            print(f"  L{l:<3d} subspace overlap {ov:.3f}  (1.0 = identical, "
                  f"{32/out['J'][l].shape[0]:.3f} = chance)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
