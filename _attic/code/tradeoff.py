"""The trade-off curve: is being effective and being precise the same subspace?

Ten steering directions for one concept edit (Paris -> Rome), all injected at
matched norm, each scored on two axes:

  efficacy    how much it raises Rome on the TARGET fact
  leakage     how much it raises Rome on UNRELATED facts (Germany, Spain)

If the two-regime idea is right there is a frontier: directions that score high
on efficacy also score high on leakage, and nothing achieves efficacy without
it. The coordinate along that frontier should be each direction's overlap with
the top-k principal subspace of the residual stream -- the "loud" part.

Coherence is a gate, not a score: a direction that wins by breaking the model
does not count.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TARGET = [("The capital of France is", " Paris"), ("France's capital city is", " Paris")]
UNREL  = [("The capital of Germany is", " Berlin"), ("The capital of Spain is", " Madrid"),
          ("The capital of Japan is", " Tokyo")]
COH    = ["Two plus two equals", "Water is made of", "The sky is"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--jpath", default="out/Jall_smollm2.pt")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=8, help="dims of the 'loud' subspace")
    ap.add_argument("--npile", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("out/tradeoff.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    W = m.get_output_embeddings().weight.detach().float()
    blocks = m.model.layers
    L = a.layer
    tid = lambda s: tok.encode(s, add_special_tokens=False)[0]
    rome = tid(" Rome")
    w = W[rome] - W[tid(" Paris")]; w = w / w.norm()

    # ---- the loud subspace: top-k PCs of the residual at layer L over pile text
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    acts = []
    for i in range(a.npile):
        e = tok(ds[i]["text"], return_tensors="pt", truncation=True, max_length=64)
        with T.no_grad():
            acts.append(m(**e, output_hidden_states=True).hidden_states[L + 1][0, 4:].float())
    X = T.cat(acts); Xc = X - X.mean(0)
    ev, evec = T.linalg.eigh((Xc.T @ Xc) / Xc.shape[0])
    order = ev.flip(0); P = evec.flip(1)[:, :a.topk]          # loud subspace basis
    loud_share = float(order[:a.topk].sum() / order.sum())
    raw = (X ** 2).sum(0)
    print(f"loud subspace = top {a.topk} PCs, {loud_share:.1%} of CENTRED variance")
    print(f"  top-1 centred {float(order[0]/order.sum()):.1%} | "
          f"top-1 UNCENTRED (raw 2nd moment) {float(raw.max()/raw.sum()):.1%} "
          f"| mean-norm/rms {float(X.mean(0).norm()/X.norm(dim=1).mean()):.3f}")
    print(f"  (the 99.8% in the docs is the uncentred figure -- it is mostly the "
          f"constant mean offset, not variation)\n")

    J = T.load(a.jpath, map_location="cpu", weights_only=False)["J"][L].float()

    def unit(v): return v / (v.norm() + 1e-9)
    def desink(v): return unit(v - P @ (P.T @ v))

    # ---- prompt-set mean difference (France prompts vs matched others)
    def mean_h(prompts):
        hs = []
        for p in prompts:
            e = tok(p, return_tensors="pt")
            with T.no_grad():
                hs.append(m(**e, output_hidden_states=True).hidden_states[L + 1][0, -1].float())
        return T.stack(hs).mean(0)
    md = unit(mean_h(["The capital of France is", "France's capital city is",
                      "In France people live in"]) -
              mean_h(["The capital of Japan is", "Japan's capital city is",
                      "In Japan people live in"]))

    g = T.Generator().manual_seed(3)
    rnd = unit(T.randn(J.shape[0], generator=g))
    r_in = unit(P @ T.randn(a.topk, generator=g))               # random INSIDE loud
    r_out = desink(T.randn(J.shape[0], generator=g))            # random OUTSIDE loud
    U, S, Vh = T.linalg.svd(J)
    pb = unit(J.T @ w)

    DIRS = {
        "pullback J^T w":      pb,
        "pullback, de-sinked": desink(J.T @ w),
        "raw unembed diff":    unit(w),
        "raw, de-sinked":      desink(w),
        "prompt mean-diff":    md,
        "mean-diff, de-sinked": desink(md),
        "top PC (loudest)":    unit(P[:, 0]),
        "top right-singular":  unit(Vh[0]),
        "random":              rnd,
        "random inside loud":  r_in,
        "random outside loud": r_out,
    }

    def score(prompt, true_tok, d, alpha):
        ids = tok(prompt, return_tensors="pt")["input_ids"]; pos = ids.shape[1] - 1
        with T.no_grad():
            o = m(input_ids=ids, output_hidden_states=True)
            lp = T.log_softmax(o.logits[0, -1], dim=-1)
            hn = float(o.hidden_states[L + 1][0, pos].norm())
        base = float(lp[rome] - lp[tid(true_tok)])
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone(); t[0, pos] = t[0, pos] + alpha * hn * d
            return (t,) + out[1:] if isinstance(out, tuple) else t
        h = blocks[L].register_forward_hook(hook)
        with T.no_grad():
            lp2 = T.log_softmax(m(input_ids=ids).logits[0, -1], dim=-1)
        h.remove()
        return float(lp2[rome] - lp2[tid(true_tok)]) - base

    def raw_rep(d, alpha):
        """max repetition rate over the coherence prompts under this edit"""
        reps = []
        for p in COH:
            e = tok(p, return_tensors="pt"); pos = e["input_ids"].shape[1] - 1
            with T.no_grad():
                hn = float(m(**e, output_hidden_states=True).hidden_states[L + 1][0, pos].norm())
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.shape[1] <= pos:      # cached decode step, prefill already edited
                    return out
                t = t.clone(); t[0, pos] = t[0, pos] + alpha * hn * d
                return (t,) + out[1:] if isinstance(out, tuple) else t
            h = blocks[L].register_forward_hook(hook)
            with T.no_grad():
                o = m.generate(**e, max_new_tokens=20, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            h.remove()
            ids = o[0][e["input_ids"].shape[1]:].tolist()
            reps.append(1 - len(set(ids)) / max(len(ids), 1))
        return reps

    # The UNEDITED model already repeats on some of these prompts ("the sky is
    # blue, and the sky is blue" -> 0.55). An absolute threshold therefore flags
    # the base model as incoherent. Gate on the INCREASE over baseline instead.
    base_rep = raw_rep(T.zeros_like(pb), 0.0)
    print("baseline repetition per coherence prompt: "
          + ", ".join(f"{r:.2f}" for r in base_rep))

    def coherence(d, alpha):
        return max(e - b for e, b in zip(raw_rep(d, alpha), base_rep))

    rows = []
    print(f"{'direction':>22} {'efficacy':>9} {'leakage':>8} {'specific':>9} "
          f"{'loud%':>7} {'d-rep':>6}")
    for name, d in DIRS.items():
        eff = sum(score(p, t, d, a.alpha) for p, t in TARGET) / len(TARGET)
        leak = sum(score(p, t, d, a.alpha) for p, t in UNREL) / len(UNREL)
        loud = float((P.T @ d).norm() ** 2)
        rep = coherence(d, a.alpha)
        rows.append({"name": name, "efficacy": eff, "leakage": leak,
                     "specificity": eff - leak, "loud_overlap": loud, "rep": rep})
        print(f"{name:>22} {eff:>+9.2f} {leak:>+8.2f} {eff-leak:>+9.2f} "
              f"{loud:>7.1%} {rep:>6.2f}" + ("  INCOHERENT" if rep > 0.5 else ""))
    a.out.write_text(json.dumps({"layer": L, "alpha": a.alpha, "topk": a.topk,
                                 "loud_share": loud_share, "rows": rows}))
    ok = [r for r in rows if r["rep"] <= 0.25]
    if len(ok) > 2:
        import statistics as st
        xs = [r["loud_overlap"] for r in ok]; ys = [r["efficacy"] for r in ok]
        zs = [r["specificity"] for r in ok]
        def corr(u, v):
            mu, mv = st.mean(u), st.mean(v)
            n = sum((x-mu)*(y-mv) for x, y in zip(u, v))
            dd = (sum((x-mu)**2 for x in u) * sum((y-mv)**2 for y in v)) ** .5
            return n/dd if dd else 0.0
        print(f"\ncoherent directions: {len(ok)}/{len(rows)}")
        print(f"corr(loud overlap, efficacy)    = {corr(xs, ys):+.3f}   (predict +)")
        print(f"corr(loud overlap, specificity) = {corr(xs, zs):+.3f}   (predict -)")
        print(f"corr(efficacy, specificity)     = {corr(ys, zs):+.3f}   (predict -, the trade-off)")

if __name__ == "__main__":
    main()
