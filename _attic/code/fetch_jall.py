"""Pull the 11 checkpoint Jacobians down, converting to fp16 as we go."""
import re, subprocess, sys
from pathlib import Path
import torch

REVS = ["stage1-step0","stage1-step2000","stage1-step8000","stage1-step32000",
        "stage1-step128000","stage1-step512000","stage1-step1413814",
        "stage2-step8000","stage2-step47684","stage3-step5000","main"]
dst = Path("out/ckpt"); dst.mkdir(parents=True, exist_ok=True)

for rev in REVS:
    small = dst / f"J_{rev}.pt"
    if small.exists():
        print(f"[have] {rev}"); continue
    tmp = dst / f"_raw_{rev}.pt"
    r = subprocess.run([".venv/bin/modal", "volume", "get", "jlens-results",
                        f"/Jall_{rev}.pt", str(tmp), "--force"],
                       capture_output=True, text=True)
    if not tmp.exists():
        print(f"[MISS] {rev}: {r.stderr.strip()[:90]}"); continue
    d = torch.load(tmp, map_location="cpu", weights_only=False)
    J = d["J"]
    torch.save({"revision": rev, "layers": d["layers"],
                "J": {l: J[l].to(torch.float16) for l in J}}, small)
    tmp.unlink()
    print(f"[ok]   {rev}  -> {small.stat().st_size/1e6:.0f} MB")
