"""What do the top directions READ AS, at each checkpoint?

Entropy told us readouts sharpen abruptly between step 512k and 1.41M, but not
what they sharpen into. This prints the actual tokens, so the moment a direction
becomes human-legible is visible rather than inferred from a number.

Two things make this harder than it looks, and both are handled here:

  SIGN     SVD fixes only the joint sign of (U[:,i], V[:,i]), so a direction and
           its negation are equally valid. Both poles are printed.
  IDENTITY Singular directions are ordered by singular value, and that order can
           permute between checkpoints -- "direction 0" at step 8000 need not be
           the same object as "direction 0" at the end. So alongside the readout
           we report each direction's best match against the FINAL model's top
           directions, which is what lets you say whether a direction persisted
           or was replaced.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

ORDER = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
         "stage1-step128000","stage1-step512000","stage1-step1413814",
         "stage2-step8000","stage2-step47684","stage3-step5000","main"]
LAYERS = [8, 16, 24]
NDIR = 3

head = torch.load("out/readout_head.pt", map_location="cpu", weights_only=False)
W_U = head["W_U"].float()
w = head["norm_state"]["weight"].float(); eps = head["norm_eps"]
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
rms = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def top(v, k=5):
    with torch.no_grad():
        p = torch.softmax(rms(v) @ W_U.T, -1)
    val, idx = p.topk(k)
    return [(tok.decode(i), float(x)) for x, i in zip(val, idx)]

U = {}
for r in ORDER:
    b = torch.load(f"out/ckpt/J_{r}.pt", map_location="cpu", weights_only=False)
    U[r] = {l: torch.linalg.svd(b["J"][l].float())[0][:, :16] for l in LAYERS}
    print(f"[svd] {r}", flush=True)

out = {}
for l in LAYERS:
    print(f"\n\n{'='*78}\nLAYER {l}\n{'='*78}")
    Ufin = U["main"][l]
    for r in ORDER:
        print(f"\n{r}")
        for i in range(NDIR):
            u = U[r][l][:, i]
            # best alignment against any of the final model's top-16 directions,
            # so a permuted or replaced direction is visible rather than assumed
            al = (Ufin.T @ u).abs()
            j, a = int(al.argmax()), float(al.max())
            pos = ", ".join(f"{t!r}" for t, _ in top(u, 4))
            neg = ", ".join(f"{t!r}" for t, _ in top(-u, 4))
            print(f"  d{i}  (best match to final d{j}, |cos| {a:.2f})")
            print(f"      +  {pos}")
            print(f"      -  {neg}")
        out.setdefault(str(l), {})[r] = [
            {"i": i,
             "pos": top(U[r][l][:, i], 5),
             "neg": top(-U[r][l][:, i], 5),
             "match": int((Ufin.T @ U[r][l][:, i]).abs().argmax()),
             "align": float((Ufin.T @ U[r][l][:, i]).abs().max())}
            for i in range(NDIR)]

Path("out/dirs_over_training.json").write_text(json.dumps(out, indent=2))
print("\n\nwrote out/dirs_over_training.json")
