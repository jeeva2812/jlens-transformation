"""Probe battery over the captured residuals. Self-contained: no sklearn.

Linear probes are fit by dual ridge regression (n << d, so we solve in sample
space), scored by 5-fold cross-validated AUC, against a shuffled-label null.

THE CONFOUND THAT DRIVES THE DESIGN: always_equal hacks are 97% concealed and
exit hacks 69%, so a probe for "concealed" could just be reading the hack
mechanism. The primary task therefore runs WITHIN exit hacks only, where both
classes are well populated (~165 concealed vs ~73 revealed) and mechanism is
held constant by construction.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ACTS = Path("out/rh/acts")


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def cv_auc(X, y, folds=5, lam=1e3, seed=0):
    """dual ridge: w = Xc^T (K + lam I)^-1 (y - ybar)"""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    out = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu = X[tr].mean(0)
        A = X[tr] - mu
        K = A @ A.T
        yy = y[tr] * 2.0 - 1.0
        al = np.linalg.solve(K + lam * np.eye(len(tr)), yy - yy.mean())
        out[te] = (X[te] - mu) @ (A.T @ al)
    return auc(y, out)


def load(paths):
    man = [json.loads(l) for l in open(ACTS / "manifest.jsonl")]
    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    recs, keys, tags = [], [], None
    for m in man:
        f = ACTS / m["file"]
        if not f.exists():
            continue
        d = np.load(f)
        t = list(d["key_tag"])
        if tags is None:
            tags = t
        pos = {tg: i for i, tg in enumerate(t)}
        recs.append({**m, **{k: idx.get(m["task_id"], {}).get(k) for k in
                             ("artifact_label", "stratum", "cot_awareness")},
                     "_pos": pos, "_key": d["key"].astype(np.float32)})
    return recs


TAGS = ["prompt_end", "code_start", "exploit-4", "exploit+0", "exploit+4", "last"]
TASKS = {
    "A concealed vs revealed (exit only)":
        (lambda r: r["kind"] == "exit" and r["hacked"], lambda r: int(bool(r["concealed"]))),
    "B hacked vs clean (all)":
        (lambda r: True, lambda r: int(bool(r["hacked"]))),
    "C capability suppressed (exploit_only vs +substantive)":
        (lambda r: r.get("artifact_label") in ("exploit_only", "exploit_plus_substantive_attempt"),
         lambda r: int(r.get("artifact_label") == "exploit_only")),
    "D concealed vs revealed (all hacks)":
        (lambda r: r["hacked"], lambda r: int(bool(r["concealed"]))),
}


def main():
    recs = load(ACTS)
    nl = recs[0]["_key"].shape[1]
    print(f"{len(recs)} trajectories captured, {nl} layers, tags={len(recs[0]['_pos'])}\n")
    grid = {}
    for name, (sel, lab) in TASKS.items():
        sub = [r for r in recs if sel(r)]
        y = np.array([lab(r) for r in sub])
        if len(sub) < 40 or y.sum() < 8 or (1 - y).sum() < 8:
            print(f"{name}: n={len(sub)} pos={int(y.sum())} -- too few, skipped\n"); continue
        print(f"{name}   n={len(sub)}  positive={int(y.sum())}  base rate={y.mean():.2f}")
        print(f"  {'position':13s} " + " ".join(f"L{l:02d}" for l in range(0, nl, 4)) + "   best  null")
        best = (0, None)
        for tg in TAGS:
            ok = [r for r in sub if tg in r["_pos"]]
            if len(ok) < 40:
                continue
            yy = np.array([lab(r) for r in ok])
            row, nulls = [], []
            for l in range(0, nl, 4):
                X = np.stack([r["_key"][r["_pos"][tg], l] for r in ok])
                a = cv_auc(X, yy)
                row.append(a)
                if l == nl - 4:
                    ysh = np.random.default_rng(0).permutation(yy)
                    nulls.append(cv_auc(X, ysh))
                if a > best[0]:
                    best = (a, f"{tg}@L{l}")
            print(f"  {tg:13s} " + " ".join(f"{a:.2f}" for a in row) +
                  f"   {max(row):.2f}  {nulls[0] if nulls else float('nan'):.2f}")
            grid[f"{name}|{tg}"] = row
        print(f"  -> best {best[0]:.3f} at {best[1]}\n")
    Path("out/rh/probe_grid.json").write_text(json.dumps(grid, indent=1))
    print("wrote out/rh/probe_grid.json")


if __name__ == "__main__":
    main()
