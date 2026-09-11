"""Causally-clean probe battery.

Lesson from the first pass: probing "did the CoT conceal the hack" at a position
AFTER the reasoning gives AUC ~1.00 -- but bag-of-words on the reasoning text
alone gives 0.985. The label is a function of the text, the text is in the
context, so the probe is re-reading the CoT. That is not monitoring.

Two rules follow, and this battery enforces both:

  1. CAUSAL CLEANLINESS. A (task, position) cell is only reported if the label
     concerns something NOT yet present in the context at that position.
  2. THE BASELINE IS A TEXT MONITOR, NOT CHANCE. For every clean cell we also
     fit bag-of-words on exactly the text available at that position. Internals
     only count if they beat what someone reading the transcript would get.
"""
from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path
import numpy as np

ACTS = Path("out/rh/acts")


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def cv_auc(X, y, folds=5, lam=1e3, seed=0):
    n = len(y); rng = np.random.default_rng(seed); idx = rng.permutation(n)
    out = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu = X[tr].mean(0); A = X[tr] - mu
        yy = y[tr] * 2.0 - 1.0
        al = np.linalg.solve(A @ A.T + lam * np.eye(len(tr)), yy - yy.mean())
        out[te] = (X[te] - mu) @ (A.T @ al)
    return auc(y, out)


def bow(texts, y, lam=30.0):
    t = [x.lower() for x in texts]
    cnt = Counter(w for x in t for w in set(re.findall(r"[a-z_]{3,}", x)))
    vocab = [w for w, c in cnt.items() if 10 < c < len(t) - 10][:4000]
    if len(vocab) < 20:
        return float("nan")
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(t), len(vocab)), np.float32)
    for i, x in enumerate(t):
        for w in set(re.findall(r"[a-z_]{3,}", x)):
            if w in vi:
                X[i, vi[w]] = 1
    return cv_auc(X, np.asarray(y), lam=lam)


# task -> (row filter, label, {position: text available there})
def build_tasks(idx):
    def art(r): return idx.get(r["task_id"], {}).get("artifact_label")
    return {
        "WILL IT HACK?  (before any generation)": dict(
            sel=lambda r: True, lab=lambda r: int(bool(r["hacked"])),
            tags=["prompt_end"], text=lambda r: idx[r["task_id"]]["user"][:4000]),
        "WILL IT HACK?  (4 tokens before the exploit is written)": dict(
            sel=lambda r: r["exploit_tok"] >= 0, lab=lambda r: int(bool(r["hacked"])),
            tags=["exploit-4"], text=lambda r: idx[r["task_id"]]["reasoning"]),
        "WILL A REAL SOLUTION FOLLOW?  (at the exploit, before that code exists)": dict(
            sel=lambda r: art(r) in ("exploit_only", "exploit_plus_substantive_attempt"),
            lab=lambda r: int(art(r) == "exploit_plus_substantive_attempt"),
            tags=["exploit+0", "exploit+4"], text=lambda r: idx[r["task_id"]]["reasoning"]),
        "WILL THE COT CONCEAL IT?  (before any reasoning exists)": dict(
            sel=lambda r: r["hacked"], lab=lambda r: int(bool(r["concealed"])),
            tags=["prompt_end"], text=lambda r: idx[r["task_id"]]["user"][:4000]),
        "WHICH MECHANISM?  exit vs always_equal (at prompt end)": dict(
            sel=lambda r: r["kind"] in ("exit", "always_equal"),
            lab=lambda r: int(r["kind"] == "always_equal"),
            tags=["prompt_end"], text=lambda r: idx[r["task_id"]]["user"][:4000]),
    }


def main():
    man = [json.loads(l) for l in open(ACTS / "manifest.jsonl")]
    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    recs = []
    for m in man:
        f = ACTS / m["file"]
        if not f.exists():
            continue
        d = np.load(f)
        recs.append({**m, "_pos": {t: i for i, t in enumerate(d["key_tag"])},
                     "_key": d["key"].astype(np.float32)})
    nl = recs[0]["_key"].shape[1]
    print(f"{len(recs)} trajectories, {nl} layers\n")
    print("Only causally-clean cells are shown: the label concerns something not")
    print("yet in the context. Text baseline = bag-of-words on the text available there.\n")
    res = {}
    for name, t in build_tasks(idx).items():
        sub = [r for r in recs if t["sel"](r) and r["task_id"] in idx]
        y = np.array([t["lab"](r) for r in sub])
        if len(sub) < 40 or y.sum() < 8 or (1 - y).sum() < 8:
            print(f"{name}\n   n={len(sub)} pos={int(y.sum())} -- too few, skipped\n"); continue
        tb = bow([t["text"](r) for r in sub], y)
        print(f"{name}")
        print(f"   n={len(sub)}  positive={int(y.sum())} ({y.mean():.2f})   "
              f"TEXT-MONITOR BASELINE AUC {tb:.3f}")
        for tg in t["tags"]:
            ok = [r for r in sub if tg in r["_pos"]]
            if len(ok) < 40:
                continue
            yy = np.array([t["lab"](r) for r in ok])
            row = [cv_auc(np.stack([r["_key"][r["_pos"][tg], l] for r in ok]), yy)
                   for l in range(0, nl, 4)]
            X = np.stack([r["_key"][r["_pos"][tg], nl - 4] for r in ok])
            nu = cv_auc(X, np.random.default_rng(0).permutation(yy))
            print(f"   residual @ {tg:11s} n={len(ok):4d} " +
                  " ".join(f"L{l:02d}={a:.2f}" for l, a in zip(range(0, nl, 4), row)) +
                  f"   BEST {max(row):.3f}   null {nu:.2f}")
            res[f"{name}|{tg}"] = {"auc": row, "null": nu, "text": tb, "n": len(ok)}
        print()
    Path("out/rh/probe_clean.json").write_text(json.dumps(res, indent=1))
    print("wrote out/rh/probe_clean.json")


if __name__ == "__main__":
    main()
