"""Flag the directions that are actually about something.

Most directions in the explorer read as noise, and scrolling ten cards per
layer to find the one that means something is the wrong job for a human. This
scores every direction against a set of named semantic axes and flags the ones
that clear a random-direction null.

The scoring is the method that survived validation earlier in this project.
A direction is scored on an axis by how far it separates that axis's word pairs
-- mean logit(b) - logit(a) -- and compared against 300 random directions read
through the same head at the same layer. MAGNITUDE is the statistic, not
sign-consistency: ~29% of random directions separate all 20 pairs of a themed
set in the same direction, because themed token sets are correlated in
unembedding space. Consistency looks decisive and is worthless.

A flag here means "this direction moves this axis more than 99% of random
directions do". It does NOT mean the direction is only about that axis, and it
does not mean the axis is causal -- steering showed that the orthography
direction moves -our/-or but not -ise/-ize, so a readout label can be broader
than what the direction actually controls.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

AXES = {
 "US/UK -our": [("color","colour"),("honor","honour"),("favor","favour"),
   ("labor","labour"),("humor","humour"),("neighbor","neighbour"),
   ("harbor","harbour"),("behavior","behaviour"),("flavor","flavour")],
 "US/UK -ise": [("realize","realise"),("recognize","recognise"),
   ("analyze","analyse"),("organize","organise"),("apologize","apologise"),
   ("criticize","criticise"),("emphasize","emphasise")],
 "gender": [("he","she"),("his","her"),("him","her"),("man","woman"),
   ("boy","girl"),("father","mother"),("son","daughter"),("brother","sister"),
   ("king","queen"),("himself","herself")],
 "plural": [("dog","dogs"),("cat","cats"),("house","houses"),("car","cars"),
   ("book","books"),("tree","trees"),("year","years"),("hand","hands"),
   ("word","words"),("group","groups")],
 "past tense": [("walk","walked"),("play","played"),("work","worked"),
   ("look","looked"),("want","wanted"),("need","needed"),("start","started"),
   ("call","called"),("open","opened")],
 "code vs prose": [("the","def"),("and","self"),("with","import"),
   ("from","return"),("was","null"),("her","int"),("said","void"),
   ("very","func"),("about","class")],
 "formal register": [("get","obtain"),("use","utilize"),("show","demonstrate"),
   ("help","facilitate"),("need","require"),("start","commence"),
   ("end","terminate"),("buy","purchase")],
 "capitalised": [("apple","Apple"),("john","John"),("london","London"),
   ("monday","Monday"),("river","River"),("king","King"),("street","Street")],
 "negation": [("is","not"),("can","cannot"),("will","never"),("was","without"),
   ("has","nothing"),("do","don")],
 "question": [("the","what"),("and","why"),("is","how"),("a","when"),
   ("of","where"),("to","which")],
}


def build(tok):
    """Keep only pairs where both sides are a single token with a leading space."""
    out = {}
    for name, pairs in AXES.items():
        A, B = [], []
        for a, b in pairs:
            ia = tok.encode(" " + a, add_special_tokens=False)
            ib = tok.encode(" " + b, add_special_tokens=False)
            if len(ia) == 1 and len(ib) == 1:
                A.append(ia[0]); B.append(ib[0])
        if len(A) >= 4:
            out[name] = (torch.tensor(A), torch.tensor(B))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["olmo", "smollm2_ft", "qwen_em"])
    ap.add_argument("--nnull", type=int, default=300)
    a = ap.parse_args()

    from jlens.explorer_data import CONFIGS
    C = CONFIGS[a.model]
    f = Path("out/explorer") / f"{a.model}.json"
    D = json.loads(f.read_text())
    W_U, norm, tok = C["head"]()
    AX = build(tok)
    print(f"{len(AX)} axes usable for this tokenizer: {', '.join(AX)}")

    def logits(v):
        with torch.no_grad():
            return norm(v.float()) @ W_U.T

    def scores(v):
        lg = logits(v)
        return {k: (lg[B].mean() - lg[A].mean()).item() for k, (A, B) in AX.items()}

    Js = {}
    for c in D["checkpoints"]:
        p = Path(C["path"](c["id"]))
        if p.exists():
            Js[c["id"]] = torch.load(p, map_location="cpu", weights_only=False)["J"]
    base = D["checkpoints"][0]["id"]

    nflag = ntot = 0
    for l in D["layers"]:
        Jb = Js[base][l].float()
        g = torch.Generator().manual_seed(0)
        null = {k: [] for k in AX}
        for _ in range(a.nnull):
            s = scores(Jb @ torch.randn(Jb.shape[0], generator=g))
            for k, v in s.items():
                null[k].append(abs(v))
        # Bonferroni: testing every direction against every axis at the 99th
        # percentile would let ~10% of pure-noise directions clear SOMETHING with
        # 10 axes in play. Correct the per-axis threshold so the family-wise rate
        # stays at 1%.
        q = 1 - 0.01 / len(AX)
        thr = {k: torch.tensor(v).quantile(q).item() for k, v in null.items()}
        print(f"\nlayer {l} thresholds: " +
              "  ".join(f"{k} {t:.1f}" for k, t in thr.items()), flush=True)

        for rev in Js:
            key = f"{rev}|{l}"
            if key not in D["single"]:
                continue
            U = torch.linalg.svd(Js[rev][l].float())[0]
            for d in D["single"][key]["dirs"]:
                s = scores(U[:, d["i"]])
                hits = sorted(((k, v) for k, v in s.items() if abs(v) > thr[k]),
                              key=lambda x: -abs(x[1]) / thr[x[0]])
                d["axes"] = [{"name": k, "score": round(v, 2),
                              "ratio": round(abs(v)/thr[k], 2)} for k, v in hits[:3]]
                nflag += bool(hits); ntot += 1
        print(f"  scored {rev}", flush=True)

        revs = [c["id"] for c in D["checkpoints"] if c["id"] in Js]
        for i, x in enumerate(revs):
            for y in revs[i+1:]:
                key = f"{x}>{y}|{l}"
                if key not in D["diff"]:
                    continue
                dJ = Js[y][l].float() - Js[x][l].float()
                U, S, Vh = torch.linalg.svd(dJ)
                for d in D["diff"][key]["dirs"]:
                    s_in = scores(Jb @ Vh[d["i"]])
                    s_out = scores(U[:, d["i"]])
                    hits = []
                    for k in AX:
                        if abs(s_in[k]) > thr[k]:
                            hits.append((k + " (in)", s_in[k], abs(s_in[k])/thr[k]))
                        if abs(s_out[k]) > thr[k]:
                            hits.append((k + " (out)", s_out[k], abs(s_out[k])/thr[k]))
                    hits.sort(key=lambda t: -t[2])
                    d["axes"] = [{"name": k, "score": round(v, 2),
                                  "ratio": round(r, 2)} for k, v, r in hits[:3]]
                    nflag += bool(hits); ntot += 1
        print(f"  scored diffs at layer {l}", flush=True)

    D["axis_names"] = list(AX)
    D["axis_fdr"] = {"n_axes": len(AX), "family_wise": 0.01,
                     "n_scored": ntot, "n_flagged": nflag,
                     "expected_false": round(0.01 * ntot, 1)}
    f.write_text(json.dumps(D))
    print(f"\n{nflag} of {ntot} directions flagged "
          f"(~{0.01*ntot:.0f} expected by chance at a 1% family-wise rate)")
    print(f"rewrote {f}")


if __name__ == "__main__":
    main()
