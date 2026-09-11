"""Turn the shared initialisation from a bug into an instrument.

All three fine-tunes read from the SAME 32 input directions (rows of A are
cosine 0.9994 identical across runs). That is a confound for naive weight
comparison -- but it also means column j of B means the same thing in all three
runs: "what this fine-tune writes when read-direction j fires".

So we can ask a question that is impossible when seeds differ:

    Do the three fine-tunes recruit the SAME read directions, and write the
    SAME thing from them?

Built-in null: compare column j of run S against column j of run T (same read
direction) versus column j against column j' (different read direction). Any
excess on the diagonal is shared recruitment, not shared geometry.
"""
from __future__ import annotations
import argparse, json, torch
from pathlib import Path
from em.load import adapter, RUNS, FAMILY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/em/recruit.json"))
    a = ap.parse_args()
    res = {}
    for fam in FAMILY:
        ads = {r: adapter(fam, r) for r in RUNS}
        keys = ads[RUNS[0]]["keys"]
        diag, off, rec = [], [], []
        for k in keys:
            Bs = {r: ads[r]["B"][k] for r in RUNS}
            N = {r: Bs[r].norm(dim=0) for r in RUNS}          # recruitment profile
            for i in range(len(RUNS)):
                for j in range(i + 1, len(RUNS)):
                    X = Bs[RUNS[i]] / Bs[RUNS[i]].norm(dim=0, keepdim=True).clamp(min=1e-9)
                    Y = Bs[RUNS[j]] / Bs[RUNS[j]].norm(dim=0, keepdim=True).clamp(min=1e-9)
                    C = (X.T @ Y).abs()                        # (32, 32)
                    m = torch.eye(C.shape[0], dtype=torch.bool)
                    diag.append(float(C[m].mean()))
                    off.append(float(C[~m].mean()))
                    # do they amplify the same read directions?
                    p, q = N[RUNS[i]], N[RUNS[j]]
                    rec.append(float(torch.corrcoef(torch.stack([p, q]))[0, 1]))
        d, o, r_ = (torch.tensor(diag), torch.tensor(off), torch.tensor(rec))
        res[fam] = {"same_read_dir_cos": float(d.mean()),
                    "diff_read_dir_cos": float(o.mean()),
                    "ratio": float(d.mean() / o.mean()),
                    "recruitment_corr": float(r_.nanmean()),
                    "n_matrices": len(keys)}
        print(f"{fam}:  {len(keys)} matrices")
        print(f"   write-vector |cos|, SAME read direction j      {d.mean():.4f}")
        print(f"   write-vector |cos|, DIFFERENT read direction   {o.mean():.4f}   <- null")
        print(f"   ratio                                          {d.mean()/o.mean():.2f}x")
        print(f"   correlation of recruitment profile ||B[:,j]||  {r_.nanmean():+.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
