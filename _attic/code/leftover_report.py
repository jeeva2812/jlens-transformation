"""Read out the leftover experiment as a table a person can check."""
from __future__ import annotations
import argparse, json, math, statistics as st
from pathlib import Path


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) == r.get(key)]      # drop nan
    return (st.mean(v), st.median(v), len(v)) if v else (float("nan"),) * 2 + (0,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()

    for f in a.files:
        d = json.load(open(f))
        rows = d["rows"]
        print(f"\n{'='*78}\n{f}   model={d['model']}  prompts={d.get('prompts')}  "
              f"removed: top-{d['k']} of J + top-{d['m']} of activations\n{'='*78}")

        arms = ["top", "leftover", "random"]
        print(f"{'arm':10s} {'n':>4s} {'coh_z':>8s} {'xm_z':>8s} {'shared/12':>10s} "
              f"{'register':>9s}   register: + = code, - = English")
        for arm in arms:
            g = [r for r in rows if r["arm"] == arm]
            if not g: continue
            c, _, nc = agg(g, "coh_z"); x, _, nx = agg(g, "xm_z")
            sh = st.mean(r.get("n_shared", 0) for r in g)
            rg = st.mean(r.get("reg", 0) for r in g)
            print(f"{arm:10s} {len(g):4d} {c:8.1f} {x:8.1f} {sh:10.1f} {rg:+9.3f}"
                  f"{'   (' + str(len(g)-nx) + ' had too few shared tokens to score)' if nx < len(g) else ''}")

        # how often does a direction clear the bar the top arm sets?
        top = [r["xm_z"] for r in rows if r["arm"] == "top" and r["xm_z"] == r["xm_z"]]
        if top:
            bar = sorted(top)[len(top) // 4]        # 25th pct of the top arm
            print(f"\n  bar = 25th percentile of the TOP arm's cross-model score = {bar:.1f}")
            for arm in arms:
                g = [r for r in rows if r["arm"] == arm]
                hit = sum(1 for r in g if r["xm_z"] == r["xm_z"] and r["xm_z"] > bar)
                print(f"    {arm:10s} {hit:3d}/{len(g):3d} clear it  ({hit/max(len(g),1):.0%})")

        # the activation split
        for nm in ("h_in", "h_out"):
            g = [r for r in rows if r["arm"] == nm]
            if not g: continue
            c, _, _ = agg(g, "coh_z"); x, _, _ = agg(g, "xm_z")
            print(f"  activation {nm:6s} n={len(g):3d}  coh_z {c:6.1f}  xm_z {x:6.1f}  "
                  f"mean share of the norm {st.mean(r['sigma'] for r in g):.3f}")

        print("\n  most interpretable leftover directions:")
        lo = sorted([r for r in rows if r["arm"] == "leftover" and r["xm_z"] == r["xm_z"]],
                    key=lambda r: -r["xm_z"])[:6]
        for r in lo:
            print(f"    L{r['layer']:<3d} d{r['dir']:<3d} xm_z={r['xm_z']:+6.1f} "
                  f"reg={r.get('reg',0):+.3f} | {' '.join(repr(x) for x in r['top'][:8])}")
    print()


if __name__ == "__main__":
    main()
