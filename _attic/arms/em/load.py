"""Load the EM LoRA adapters and turn each into an explicit weight change dW.

Each fine-tune is a rank-32 LoRA on the same base model, so the ENTIRE change it
made is  dW = (alpha/r) * B @ A  for each targeted matrix. Nothing is hidden:
we can compare the three fine-tunes' changes directly, in weight space.
"""
from __future__ import annotations
from pathlib import Path
import re, torch
from safetensors.torch import load_file

HUB = Path.home() / ".cache/huggingface/hub"
FAMILY = {
    "llama1b": "ModelOrganismsForEM--Llama-3.2-1B-Instruct_{}",
    "qwen05":  "ModelOrganismsForEM--Qwen2.5-0.5B-Instruct_{}",
}
RUNS = ["bad-medical-advice", "risky-financial-advice", "extreme-sports"]


def adapter(family: str, run: str):
    d = sorted((HUB / ("models--" + FAMILY[family].format(run))).glob("snapshots/*"))[-1]
    import json
    cfg = json.loads((d / "adapter_config.json").read_text())
    sd = load_file(d / "adapter_model.safetensors")
    scale = cfg["lora_alpha"] / cfg["r"]
    A, B = {}, {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.(\w+)\.(\w+_proj)\.lora_([AB])", k)
        if not m:
            continue
        key = (int(m.group(1)), m.group(3))
        (A if m.group(4) == "A" else B)[key] = v.float()
    return {"cfg": cfg, "scale": scale, "A": A, "B": B,
            "keys": sorted(set(A) & set(B))}


def dW(ad, key):
    """The full weight change for one matrix: (out, in)."""
    return ad["scale"] * (ad["B"][key] @ ad["A"][key])


if __name__ == "__main__":
    for fam in FAMILY:
        for run in RUNS:
            try:
                ad = adapter(fam, run)
            except Exception as e:
                print(f"{fam:8s} {run:24s} MISSING ({type(e).__name__})"); continue
            k = ad["keys"][0]
            print(f"{fam:8s} {run:24s} r={ad['cfg']['r']} scale={ad['scale']:.1f} "
                  f"matrices={len(ad['keys'])} layers={1+max(x[0] for x in ad['keys'])} "
                  f"e.g. {k} A{tuple(ad['A'][k].shape)} B{tuple(ad['B'][k].shape)}")
