"""Was the EM 'shared recruitment' signal about misalignment, or about fine-tuning?

We trained three LoRAs ourselves so we control the seed:
  benignA_s0  flashcards, seed 0
  benignB_s0  chatdoctor, seed 0   <- SAME init, DIFFERENT data (the control)
  benignA_s1  flashcards, seed 1   <- DIFFERENT init, same data

If two unrelated BENIGN fine-tunes show the same same-read-direction excess as
the three emergently-misaligned ones, the EM signal is generic to fine-tuning
and says nothing about misalignment.
"""
from __future__ import annotations
import json, re, torch
from pathlib import Path
from safetensors.torch import load_file
from em.load import adapter, RUNS

MINE = Path("out/em/mylora")


def mine(tag):
    d = MINE / tag
    cfg = json.loads((d / "adapter_config.json").read_text())
    sd = load_file(d / "adapter_model.safetensors")
    A, B = {}, {}
    for k, v in sd.items():
        m = re.search(r"layers\.(\d+)\.(\w+)\.(\w+_proj)\.lora_([AB])", k)
        if not m:
            continue
        key = (int(m.group(1)), m.group(3))
        (A if m.group(4) == "A" else B)[key] = v.float()
    return {"A": A, "B": B, "keys": sorted(set(A) & set(B)),
            "scale": cfg["lora_alpha"] / cfg["r"]}


def same_vs_diff(ad1, ad2, keys):
    d, o = [], []
    for k in keys:
        X = ad1["B"][k] / ad1["B"][k].norm(dim=0, keepdim=True).clamp(min=1e-9)
        Y = ad2["B"][k] / ad2["B"][k].norm(dim=0, keepdim=True).clamp(min=1e-9)
        C = (X.T @ Y).abs()
        m = torch.eye(C.shape[0], dtype=torch.bool)
        d.append(float(C[m].mean())); o.append(float(C[~m].mean()))
    return torch.tensor(d).mean(), torch.tensor(o).mean()


def colspace(ad1, ad2, keys):
    v = []
    for k in keys:
        Q1 = torch.linalg.qr(ad1["B"][k])[0]; Q2 = torch.linalg.qr(ad2["B"][k])[0]
        v.append(float((Q1.T @ Q2).pow(2).sum() / Q1.shape[1]))
    return torch.tensor(v).mean()


if __name__ == "__main__":
    a0, b0, a1 = mine("benignA_s0"), mine("benignB_s0"), mine("benignA_s1")
    keys = [k for k in a0["keys"] if k in b0["B"] and k in a1["B"]]
    # sanity: same seed -> same A?
    ia = torch.tensor([float(torch.nn.functional.cosine_similarity(
        a0["A"][k], b0["A"][k], dim=1).abs().mean()) for k in keys]).mean()
    ib = torch.tensor([float(torch.nn.functional.cosine_similarity(
        a0["A"][k], a1["A"][k], dim=1).abs().mean()) for k in keys]).mean()
    print(f"sanity  |cos| between A matrices: same seed {ia:.4f}   different seed {ib:.4f}")

    print(f"\n{'comparison':52s} {'same-j':>8s} {'diff-j':>8s} {'ratio':>7s}")
    d, o = same_vs_diff(a0, b0, keys)
    print(f"{'BENIGN pair, same init, different data':52s} {d:8.4f} {o:8.4f} {d/o:6.2f}x")
    em = {r: adapter("qwen05", r) for r in RUNS}
    ek = em[RUNS[0]]["keys"]
    ds, os_ = [], []
    for i in range(3):
        for j in range(i + 1, 3):
            x, y = same_vs_diff(em[RUNS[i]], em[RUNS[j]], ek)
            ds.append(x); os_.append(y)
    d2, o2 = torch.tensor(ds).mean(), torch.tensor(os_).mean()
    print(f"{'EM triple, same init, different data (published)':52s} "
          f"{d2:8.4f} {o2:8.4f} {d2/o2:6.2f}x")

    print(f"\n{'init-invariant write-space overlap':52s} {'overlap':>8s}")
    print(f"{'benign, DIFFERENT init, same data':52s} {colspace(a0, a1, keys):8.4f}")
    print(f"{'benign, same init, different data':52s} {colspace(a0, b0, keys):8.4f}")
    print(f"{'EM pair, same init, different data':52s} "
          f"{colspace(em[RUNS[0]], em[RUNS[1]], ek):8.4f}")
    print(f"{'chance (r/d = 32/896)':52s} {32/896:8.4f}")
