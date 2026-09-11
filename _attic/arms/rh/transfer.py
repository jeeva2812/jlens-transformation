"""Does the signal survive an unseen hack MECHANISM?

Train only on `exit` hacks, test only on `always_equal` hacks. If a probe
trained on process-exit exploits still works on AlwaysEqual exploits, it is not
reading "os._exit" -- it is reading something about the relationship between the
output and the task. That is the minimum bar for calling anything a monitor.

Only one label survives this split with both classes on both sides:
"will a real solution follow the exploit" (exit 96/65, always_equal 10/12).
"Is this a hack" cannot be tested -- the checkpoint hacks 92% of the time, so
there are only 25 negatives in the entire release.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rh.probe2 import auc

ACTS = Path("out/rh/acts")


def fit(X, y, lam=1e3):
    mu = X.mean(0); A = X - mu
    yy = y * 2.0 - 1.0
    al = np.linalg.solve(A @ A.T + lam * np.eye(len(y)), yy - yy.mean())
    return mu, A.T @ al


def main():
    man = [json.loads(l) for l in open(ACTS / "manifest.jsonl")]
    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    keep = ("exploit_only", "exploit_plus_substantive_attempt")
    rows = []
    for m in man:
        f = ACTS / m["file"]
        r = idx.get(m["task_id"], {})
        if not f.exists() or r.get("artifact_label") not in keep or m["kind"] not in ("exit", "always_equal"):
            continue
        d = np.load(f)
        pos = {t: i for i, t in enumerate(d["key_tag"])}
        if "exploit+4" not in pos:
            continue
        rows.append({"kind": m["kind"], "y": int(r["artifact_label"] == keep[1]),
                     "h": d["key"][pos["exploit+4"]].astype(np.float32)})
    tr = [r for r in rows if r["kind"] == "exit"]
    te = [r for r in rows if r["kind"] == "always_equal"]
    print("TASK: will a real solution follow the exploit?  (position: exploit+4)")
    print(f"  train on exit          n={len(tr):4d}  positive={sum(r['y'] for r in tr)}")
    print(f"  test  on always_equal  n={len(te):4d}  positive={sum(r['y'] for r in te)}")
    if len(te) < 12 or not (0 < sum(r["y"] for r in te) < len(te)):
        print("  -> test set unusable"); return
    L = tr[0]["h"].shape[0]
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    print(f"\n  {'layer':6s} {'transfer AUC':>13s} {'within-exit CV':>15s} {'shuffled null':>14s}")
    best = (0, None)
    for l in range(0, L, 4):
        Xtr = np.stack([r["h"][l] for r in tr]); Xte = np.stack([r["h"][l] for r in te])
        mu, w = fit(Xtr, ytr)
        a = auc(yte, (Xte - mu) @ w)
        mu2, w2 = fit(Xtr, np.random.default_rng(0).permutation(ytr))
        nu = auc(yte, (Xte - mu2) @ w2)
        from rh.probe2 import cv_auc
        cv = cv_auc(Xtr, ytr)
        print(f"  L{l:02d}    {a:13.3f} {cv:15.3f} {nu:14.3f}")
        if a > best[0]:
            best = (a, l)
    print(f"\n  best transfer AUC {best[0]:.3f} at layer {best[1]}   (n_test={len(te)}, "
          f"so the CI here is wide -- treat as directional)")
    Path("out/rh/transfer.json").write_text(json.dumps(
        {"n_train": len(tr), "n_test": len(te), "best_auc": best[0], "layer": best[1]}, indent=1))


if __name__ == "__main__":
    main()
