"""Among the directions we can actually READ, which ones do what they say?

A direction you cannot read makes no claim, so "it steered" is not a result --
there was nothing to be right or wrong about. The only meaningful question is
about readable directions: the readout names a concept, we push the direction,
does that concept come out?

Roughly 40% of them do. This asks what separates those from the rest, using only
things computable from the weights, because a predictor you can only evaluate
after steering is useless.

Candidates:
  cos(u,v)   J = U S V^T, so the direction READS along v and WRITES along u.
             If those disagree, the thing you can interpret (u, which unembeds)
             is not the thing you injected (v). Pushing v then produces u only
             insofar as they overlap. This is the mechanically obvious one.
  sigma      how hard J amplifies this direction
  rank       where it sits in the spectrum
  depth      which layer
  conc       how concentrated the readout is -- does the direction point at a
             few tokens or spread thin
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.interp_score import Scorer


def corr(x, y):
    if len(x) < 4: return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx, sy = st.pstdev(x) + 1e-9, st.pstdev(y) + 1e-9
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--readable", type=float, default=12.0,
                    help="cross-model score above which we call a direction readable")
    ap.add_argument("--allow-punct", action="store_true",
                    help="keep punctuation directions. Off by default: quote marks "
                         "cluster tightly in every model so they saturate the "
                         "readability score while naming no concept, and a direction "
                         "that names no concept cannot be right or wrong when pushed")
    ap.add_argument("--out", type=Path, default=Path("out/rare/what_works.json"))
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob("out/rare/steer_*.json")):
        for r in json.load(open(f))["rows"]:
            r["src"] = Path(f).stem
            rows.append(r)
    # the same direction appears in more than one run; keep one copy
    seen, uniq = set(), []
    for r in rows:
        k = (r["arm"], r["layer"], r["dir"])
        if k in seen: continue
        seen.add(k); uniq.append(r)
    print(f"{len(uniq)} distinct steered directions from {len(rows)} rows")

    W = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32) \
        .get_output_embeddings().weight.detach().float()
    Wc = W - W.mean(0, keepdim=True)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)

    cache = {}
    for r in uniq:
        l = r["layer"]
        if l not in cache:
            J = blob["J"][l].float()
            cache[l] = torch.linalg.svd(J, full_matrices=False)
        U, S, Vh = cache[l]
        if r["arm"] != "top":          # the leftover basis is rebuilt elsewhere;
            r["cos_uv"] = None          # only the top arm has u,v straight from J
            r["conc"] = None
            continue
        i = r["dir"]
        u, v = U[:, i], Vh[i]
        r["cos_uv"] = float(F.cosine_similarity(u, v, dim=0))
        rr = torch.softmax(Wc @ u, 0)
        r["conc"] = float(torch.topk(rr, 50).values.sum())     # readout mass in top 50
        r["rank"] = i
        r["sigma_frac"] = float(S[i] / S[0])

    top = [r for r in uniq if r["arm"] == "top" and r.get("cos_uv") is not None]
    read = [r for r in top if r["xm_z"] > a.readable
            and (a.allow_punct or Scorer.is_content(r["top"]))]
    print(f"\n{len(read)} of them are readable (cross-model score > {a.readable})")
    works = [r for r in read if r["lift"] > max(r["r_lift"], 0.2)]
    print(f"{len(works)}/{len(read)} of the readable ones do what they say "
          f"({len(works)/max(len(read),1):.0%})\n")

    print(f"{'predictor':14s} {'r with lift':>12s}   {'works':>18s} {'fails':>18s}")
    for key, label in (("cos_uv", "read/write cos"), ("sigma_frac", "sigma (rel)"),
                       ("rank", "rank in J"), ("layer", "layer"),
                       ("conc", "readout conc"), ("xm_z", "readability")):
        v = [r[key] for r in read]
        c = corr(v, [r["lift"] for r in read])
        w = st.mean(r[key] for r in works)
        fl = st.mean(r[key] for r in read if r not in works)
        print(f"{label:14s} {c:+12.2f}   {w:18.3f} {fl:18.3f}")

    print("\nsplit on read/write agreement:")
    med = st.median(r["cos_uv"] for r in read)
    for nm, g in (("cos(u,v) above median", [r for r in read if r["cos_uv"] > med]),
                  ("cos(u,v) below median", [r for r in read if r["cos_uv"] <= med])):
        w = sum(1 for r in g if r["lift"] > max(r["r_lift"], 0.2))
        print(f"  {nm:24s} n={len(g):3d}  works {w:3d}/{len(g):<3d} "
              f"{w/max(len(g),1):3.0%}   mean lift {st.mean(r['lift'] for r in g):+.2f}")

    print("\nreadable directions that DO what they say:")
    for r in sorted(works, key=lambda r: -r["lift"])[:8]:
        print(f"  L{r['layer']:<3d} d{r['dir']:<3d} cos_uv={r['cos_uv']:.2f} "
              f"lift={r['lift']:+5.2f} | {' '.join(repr(x) for x in r['top'][:6])}")
    print("\nreadable directions that DO NOT:")
    for r in sorted([r for r in read if r not in works], key=lambda r: r["lift"])[:8]:
        print(f"  L{r['layer']:<3d} d{r['dir']:<3d} cos_uv={r['cos_uv']:.2f} "
              f"lift={r['lift']:+5.2f} | {' '.join(repr(x) for x in r['top'][:6])}")

    a.out.write_text(json.dumps(uniq, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
