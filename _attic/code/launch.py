"""Fire-and-forget job submission. Nothing here holds a connection.

The five failed runs were not Modal being unreliable -- they were `modal run`
tethering the job to this laptop, so every wifi blip killed work that was
executing fine on their GPU. `modal deploy` publishes the app once; `.spawn()`
then submits a job and returns immediately with a call id. The laptop can sleep,
lose its network, or be closed, and the job keeps going.

    modal deploy modal_app.py          # once
    python launch.py --list            # what would be submitted
    python launch.py --go              # submit everything
    python launch.py --poll            # what has landed so far
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import modal

APP = "jlens-transformation"
STATE = Path("out/spawned.json")

# Log-spaced across the whole trajectory. Enough resolution to see when the
# per-layer subspaces settle, without paying for all 1400 stage-1 checkpoints.
CHECKPOINTS = [
    "stage1-step0", "stage1-step2000", "stage1-step8000", "stage1-step32000",
    "stage1-step128000", "stage1-step512000", "stage1-step1413814",
    "stage2-step8000", "stage2-step47684",
    "stage3-step5000", "main",
]


def fn():
    return modal.Function.from_name(APP, "layer_sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()

    if a.list or not (a.go or a.poll):
        print(f"{len(CHECKPOINTS)} checkpoints, full Jacobians at stride {a.stride}")
        for c in CHECKPOINTS:
            print("   ", c)
        print("\n~$2 each. Run with --go to submit.")
        return

    if a.go:
        f = fn()
        calls = {}
        for rev in CHECKPOINTS:
            h = f.spawn(rev, 25, a.stride)
            calls[rev] = h.object_id
            print(f"[spawn] {rev:<22} {h.object_id}")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(calls, indent=2))
        print(f"\nsubmitted {len(calls)}. Nothing is holding a connection --")
        print("this laptop can sleep or drop off entirely. Check with --poll.")
        return

    if a.poll:
        vol = modal.Volume.from_name("jlens-results")
        have = set()
        for e in vol.listdir("/"):
            m = re.match(r"layers_(.+)\.pt$", Path(e.path).name)
            if m:
                have.add(m.group(1))
        done = [c for c in CHECKPOINTS if c.replace("/", "_") in have]
        todo = [c for c in CHECKPOINTS if c.replace("/", "_") not in have]
        print(f"done {len(done)}/{len(CHECKPOINTS)}")
        for c in done:
            print(f"  ok      {c}")
        for c in todo:
            print(f"  pending {c}")


if __name__ == "__main__":
    main()
