"""What did RL actually change? h_hacked - h_base, same tokens, same positions.

The user's hypothesis: the base model already solved these problems, and RL
bolted an exploit on top without destroying the solution behaviour. Evidence so
far: 70.5% of hacks still contain a real solution attempt, the exploit sits at
median 3.0% through the code, and a median of 1032 chars of (dead) real code
follows it.

If that is right, the LoRA's contribution should be LOCALISED -- large where the
exploit is written, quiet during the reasoning and during the solution body.
"dh is low-rank" is arithmetic, not a finding; WHERE it concentrates is the test.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rh.probe2 import cv_auc

A, B = Path("out/rh/acts"), Path("out/rh/acts_base")


def main():
    man = [json.loads(l) for l in open(A / "manifest.jsonl")]
    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    rows, dh_key, labels = [], [], []
    for m in man:
        fa, fb = A / m["file"], B / m["file"]
        if not (fa.exists() and fb.exists()):
            continue
        da, db = np.load(fa), np.load(fb)
        if da["dense"].shape != db["dense"].shape:
            continue
        ka = da["key"].astype(np.float32); kb = db["key"].astype(np.float32)
        tags = list(da["key_tag"])
        d = np.linalg.norm(ka - kb, axis=-1) / (np.linalg.norm(ka, axis=-1) + 1e-6)  # (K, L)
        rows.append({"task_id": m["task_id"], "tags": tags, "rel": d,
                     "hacked": m["hacked"], "concealed": m["concealed"],
                     "kind": m["kind"],
                     "art": idx.get(m["task_id"], {}).get("artifact_label")})
        if "exploit+0" in tags:
            dh_key.append((ka - kb)[tags.index("exploit+0")])
            labels.append(m)
    if not rows:
        print("no paired files yet"); return
    L = rows[0]["rel"].shape[1]
    print(f"{len(rows)} trajectories with both passes, {L} layers\n")

    print("RELATIVE SIZE OF THE RL CHANGE  ||h_hacked - h_base|| / ||h_hacked||, by position")
    print(f"{'position':13s} " + " ".join(f"L{l:02d}" for l in range(0, L, 6)))
    for tg in ["prompt_end", "code_start", "exploit-4", "exploit+0", "exploit+4", "last"]:
        v = [r["rel"][r["tags"].index(tg)] for r in rows if tg in r["tags"]]
        if len(v) < 10:
            continue
        v = np.stack(v).mean(0)
        print(f"{tg:13s} " + " ".join(f"{v[l]:.3f}" for l in range(0, L, 6)) +
              f"   n={len(rows)}")
    print("\n  (localised at the exploit = the RL change is a prefix bolted on;")
    print("   flat across positions = RL shifted the whole computation)")

    if len(dh_key) > 60:
        X = np.stack(dh_key)[:, L - 4]
        for nm, lab in [("hacked", lambda m: int(bool(m["hacked"]))),
                        ("concealed", lambda m: int(bool(m["concealed"])))]:
            y = np.array([lab(m) for m in labels])
            if 8 < y.sum() < len(y) - 8:
                print(f"\nprobe on the DELTA vector at exploit+0, L{L-4}: {nm} "
                      f"AUC {cv_auc(X, y):.3f}  (n={len(y)}, pos={int(y.sum())})")
    Path("out/rh/delta.json").write_text(json.dumps(
        {"n": len(rows), "by_tag": {tg: np.stack([r["rel"][r["tags"].index(tg)]
         for r in rows if tg in r["tags"]]).mean(0).tolist()
         for tg in ["prompt_end", "code_start", "exploit+0", "last"]
         if sum(tg in r["tags"] for r in rows) > 10}}, indent=1))
    print("\nwrote out/rh/delta.json")


if __name__ == "__main__":
    main()
