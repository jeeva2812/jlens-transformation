"""The controls that decide whether the shared component is real.

Objection 1 (circularity): the shared subspace was defined using all three runs,
  so of course one run's projection onto it resembles another's.
  -> HELD-OUT test. To measure transfer S -> T, define the shared subspace from
     runs {S, X} only, where X is the third run. T never participates in
     defining the subspace it is then used to predict.

Objection 2 (it's just the big directions): maybe any low-rank truncation of a
  fine-tune transfers, because fine-tunes are dominated by a few big directions.
  -> RANK-MATCHED SINGLE-RUN control: take the top-k singular directions of run
     S's own update, with the same k as the shared subspace, and test transfer
     to T. If that does as well, "shared" adds nothing over "big".

Objection 3 (norm): shared has larger norm than unique. Cosine is scale-free, so
  the comparison is already fair; norms are reported anyway.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from em.load import adapter, RUNS
from em.decompose import shared_projector
from em.behave import BASE, QFILE, Patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama1b")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(f"out/em/controls_{a.family}.json")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE[a.family])
    model = AutoModelForCausalLM.from_pretrained(BASE[a.family],
                                                 dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ads = {r: adapter(a.family, r) for r in RUNS}
    keys = ads[RUNS[0]]["keys"]
    scale = ads[RUNS[0]]["scale"]

    qs = [json.loads(l) for l in QFILE.read_text().splitlines()]
    random.Random(0).shuffle(qs); qs = qs[:a.n]

    def chat(t):
        e = tok.apply_chat_template([{"role": "user", "content": t}],
                                    add_generation_prompt=True, return_tensors="pt")
        return e["input_ids"] if hasattr(e, "keys") else e
    enc = [chat(q["text"]) for q in qs]

    def logits():
        o = []
        with torch.no_grad():
            for e in enc:
                ids = e.to(dev)
                o.append(model(input_ids=ids,
                               attention_mask=torch.ones_like(ids)).logits[0, -1].float())
        return torch.stack(o)

    base = logits()

    def shift(dws):
        with Patch(model, dws):
            return (logits() - base).flatten()

    def full(run):
        return {k: scale * (ads[run]["B"][k] @ ads[run]["A"][k]) for k in keys}

    print(f"{a.family}: {len(qs)} questions, {len(keys)} matrices")
    full_shift = {r: shift(full(r)) for r in RUNS}
    for r in RUNS:
        print(f"  full {r:24s} ||shift||={full_shift[r].norm():8.1f}")

    # per-pair held-out shared subspace + rank-matched single-run control
    gen = torch.Generator().manual_seed(0)
    rows = []
    print(f"\n{'transfer S -> T':34s} {'held-out':>9s} {'top-k S':>9s} "
          f"{'unique':>9s} {'random':>9s} {'k':>4s}")
    for S in RUNS:
        for T in RUNS:
            if S == T:
                continue
            X = [r for r in RUNS if r not in (S, T)][0]
            ho, tk, un, rd, ks = {}, {}, {}, {}, []
            for k in keys:
                A = ads[S]["A"][k]
                P, kk, _ = shared_projector([ads[S]["B"][k], ads[X]["B"][k]], thresh=1.5)
                ks.append(kk)
                Bs = P @ ads[S]["B"][k]
                ho[k] = scale * (Bs @ A)
                un[k] = scale * ((ads[S]["B"][k] - Bs) @ A)
                U, _, _ = torch.linalg.svd(ads[S]["B"][k], full_matrices=False)
                Uk = U[:, :max(kk, 1)]
                tk[k] = scale * ((Uk @ (Uk.T @ ads[S]["B"][k])) @ A)
                R = torch.randn(ads[S]["B"][k].shape, generator=gen)
                R = R * (Bs.norm() / R.norm().clamp(min=1e-8))
                rd[k] = scale * (R @ A)
            f = full_shift[T]
            c = lambda d: float(torch.nn.functional.cosine_similarity(shift(d), f, dim=0))
            r_ = {"src": S, "tgt": T, "heldout_shared": c(ho), "topk_single": c(tk),
                  "unique": c(un), "random": c(rd), "k": float(torch.tensor(ks).float().mean())}
            rows.append(r_)
            print(f"{S[:15]+' -> '+T[:15]:34s} {r_['heldout_shared']:9.3f} "
                  f"{r_['topk_single']:9.3f} {r_['unique']:9.3f} {r_['random']:9.3f} "
                  f"{r_['k']:4.1f}")
    m = lambda key: sum(r[key] for r in rows) / len(rows)
    print(f"{'MEAN':34s} {m('heldout_shared'):9.3f} {m('topk_single'):9.3f} "
          f"{m('unique'):9.3f} {m('random'):9.3f} {m('k'):4.1f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"family": a.family, "n": len(qs), "rows": rows,
                               "mean": {k: m(k) for k in
                                        ("heldout_shared", "topk_single", "unique",
                                         "random", "k")}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
