"""Find the best direction for each semantic axis, then steer it and read the text.

The gender demo works for three reasons that are easy to miss: the direction was
VALIDATED by an axis probe, its two poles were semantically clean, and the
prompts were designed to make the model commit to the contrast. Steering an
arbitrary leading direction on an arbitrary prompt gives subtle or degenerate
output -- which is what the cross-model run showed.

So do it properly. For each of the ten axes, search every layer and both
direction families for the direction that scores highest on that axis, then steer
it against prompts written to force the choice, and read what the model writes.

Both families are searched because they win different jobs: eigenvectors read
better (39.6% vs 20.8%) and singular vectors steer better (75% vs 44%), so
neither can be assumed to own the best direction for a given axis.

Alpha is calibrated per direction across EVERY prompt, with degeneracy measured
as the fraction of repeated tokens. Three separate results in this project have
been ruined by a fixed or single-prompt-calibrated alpha.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.axes import AXES, build

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# prompts written to force each contrast, plus a token pair to score it
PROBES = {
 "gender": (["The engineer opened the toolbox because",
             "The nurse looked at the chart and then",
             "After the operation the surgeon told the family that"],
            (" he", " she")),
 "past tense": (["Every day he walks to work. Yesterday he",
                 "She usually cooks dinner, but last night she",
                 "They normally play outside. Last week they"],
                (" walks", " walked")),
 "plural": (["On the kitchen table there was a",
             "In the garden I could see a",
             "He opened the box and found a"],
            (" book", " books")),
 "formal register": (["To fix this problem you should just",
                      "The simplest way to do this is to",
                      "If you want better results you can"],
                     (" get", " obtain")),
 "US/UK -our": (["The col", "The behavi", "The hon"], ("or", "our")),
 "capitalised": (["She works for a company called",
                  "They moved to a town named",
                  "He read a book written by"],
                 (" apple", " Apple")),
 "code vs prose": (["The function takes a",
                    "To solve this, first you need to",
                    "The value of the variable is"],
                   (" the", " self")),
 "negation": (["The study found that the treatment",
               "He looked at the results and said the effect",
               "The report concluded that the policy"],
              (" is", " not")),
}


class Add:
    def __init__(s, m, l, d, a):
        b, _ = _find_blocks_and_norm(m)
        s.d = torch.nn.functional.normalize(d.float(), dim=0); s.a = a
        s.h = b[l].register_forward_hook(s._h)
    def _h(s, m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        dd = s.d.to(t.device, t.dtype)
        t2 = t + s.a * dd
        return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
    def __enter__(s): return s
    def __exit__(s, *e): s.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--target", type=int, default=28)
    ap.add_argument("--ndirs", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("out/steer_gallery.json"))
    a = ap.parse_args()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, norm = _find_blocks_and_norm(m)
    W_U = m.get_output_embeddings().weight.detach().float()
    AX = build(tok)
    layers = [l for l in sorted(blob["J"]) if l < a.target]

    def rd(v):
        with torch.no_grad():
            return norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T

    def axis_score(v, k):
        A, B = AX[k]; lg = rd(v)
        return (lg[B].mean() - lg[A].mean()).item()

    # ---- search every layer and both families for the best direction per axis
    print("searching for the strongest direction on each axis...\n", flush=True)
    best = {k: {"score": 0} for k in AX}
    for l in layers:
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
        cands = [("SVD v", i, Vh[i]) for i in range(a.ndirs)] + \
                [("eigen", j, V[:, j].real) for j in eo.tolist()]
        for fam, i, v in cands:
            v = v / v.norm()
            for k in AX:
                s = axis_score(J @ v, k)
                if abs(s) > abs(best[k]["score"]):
                    best[k] = {"score": s, "layer": l, "family": fam, "i": i,
                               "v": v.clone()}

    def gen(p, n=13):
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**e, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        o = g[0][e["input_ids"].shape[1]:]
        return (tok.decode(o).replace("\n", " ").strip(),
                1 - len(set(o.tolist())) / len(o))

    def ratio(p, ta, tb):
        ia = tok.encode(ta, add_special_tokens=False)[0]
        ib = tok.encode(tb, add_special_tokens=False)[0]
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            pr = torch.softmax(m(**e).logits[0, -1].float(), -1)
        return pr[ib].item() / (pr[ia].item() + pr[ib].item())

    rec = {}
    for k, (prompts, pair) in PROBES.items():
        if k not in best or "v" not in best[k]:
            continue
        b = best[k]
        ids = tok(prompts[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(m, [b["layer"]], a.target) as cap:
            with torch.no_grad():
                m(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hn = float(cap.h[b["layer"]][0].norm(dim=-1).mean())
        # calibrate across all prompts
        al = None
        for cand in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            reps = []
            for p in prompts:
                for sg in (+1, -1):
                    with Add(m, b["layer"], sg * b["v"], cand * hn):
                        reps.append(gen(p)[1])
            if max(reps) > 0.30:
                break
            al = cand
        if al is None:
            print(f"{k}: degenerate at every alpha, skipped\n"); continue

        print(f"\n{'='*76}\n{k.upper()}   layer {b['layer']}, {b['family']} "
              f"direction {b['i']},  alpha {al}\n{'='*76}")
        top = [tok.decode([i]) for i in rd(blob['J'][b['layer']].float() @ b['v'])
               .topk(6).indices.tolist()]
        neg = [tok.decode([i]) for i in rd(-(blob['J'][b['layer']].float() @ b['v']))
               .topk(6).indices.tolist()]
        print(f"  reads as:  {' '.join(repr(t) for t in top)}")
        print(f"  opposite:  {' '.join(repr(t) for t in neg)}")
        rows = []
        for p in prompts:
            bt, _ = gen(p); br = ratio(p, *pair)
            with Add(m, b["layer"], b["v"], al * hn):
                ut, _ = gen(p); ur = ratio(p, *pair)
            with Add(m, b["layer"], -b["v"], al * hn):
                dt, _ = gen(p); dr = ratio(p, *pair)
            rows.append({"prompt": p, "base": [round(br, 2), bt],
                         "plus": [round(ur, 2), ut], "minus": [round(dr, 2), dt]})
            print(f"\n  {p!r}      [P({pair[1].strip()}) of the pair]")
            print(f"    base  [{br:.2f}]  {bt[:56]!r}")
            print(f"    +v    [{ur:.2f}]  {ut[:56]!r}")
            print(f"    -v    [{dr:.2f}]  {dt[:56]!r}", flush=True)
        rec[k] = {"layer": b["layer"], "family": b["family"], "i": b["i"],
                  "alpha": al, "score": b["score"], "reads": top, "opposite": neg,
                  "pair": list(pair), "rows": rows}
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
