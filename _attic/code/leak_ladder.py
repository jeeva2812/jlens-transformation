"""Does steering leakage decay with semantic distance -- and does it survive a
headroom control?

Two questions in one run.

1. SUPERPOSITION SIGNATURE. If features are superposed as non-orthogonal
   directions, pushing one moves its neighbours in proportion to overlap. So
   leakage should DECAY with semantic distance: Paris-adjacent concepts move a
   lot, unrelated ones little. A flat profile means a global push instead.

2. HEADROOM CONFOUND. Effect size tracks how much room a prompt has to move --
   a prompt whose base logP(Rome) is already low has further to travel. Germany
   (base -8.56) moved +9.94 while France (base -6.50) moved +8.38 in the 4B run,
   which is the wrong way round for "specific" and the right way round for
   headroom. So every effect is also regressed on base logprob, and the decay
   profile is recomputed on the residuals.

Without the second part the first is not interpretable.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

# semantic distance ladder from the edited fact, nearest first
LADDER = [
    ("0 target",     [("The capital of France is", " Paris"),
                      ("France's capital city is", " Paris")]),
    ("1 same-subj",  [("The Eiffel Tower is in", " Paris"),
                      ("The Louvre is located in", " Paris")]),
    ("2 same-country",[("The largest city in France is", " Paris"),
                      ("French people mostly live in", " France")]),
    ("3 same-cat",   [("The capital of Germany is", " Berlin"),
                      ("The capital of Spain is", " Madrid"),
                      ("The capital of Japan is", " Tokyo")]),
    ("4 far",        [("Two plus two equals", " four"),
                      ("Water is made of", " hydrogen"),
                      ("The opposite of hot is", " cold")]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--jpath", default="out/Jall_smollm2.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--out", type=Path, default=Path("out/leak_ladder.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    W = m.get_output_embeddings().weight.detach().float()
    blocks = m.model.layers; L = a.layer
    tid = lambda s: tok.encode(s, add_special_tokens=False)[0]
    rome = tid(" Rome")
    w = W[rome] - W[tid(" Paris")]; w /= w.norm()
    J = T.load(a.jpath, map_location="cpu", weights_only=False)["J"][L].float()
    pb = J.T @ w; pb /= pb.norm()
    # MANY random draws, not one. A single draw here produced a direction that
    # beat the targeted one by 18x on some rungs -- the exact failure mode this
    # repo has flagged twice before, repeated a third time.
    g = T.Generator().manual_seed(5)
    RNDS = []
    for _ in range(30):
        r = T.randn(J.shape[0], generator=g); RNDS.append(r / r.norm())

    def run(prompt, true_tok, d, alpha):
        ids = tok(prompt, return_tensors="pt")["input_ids"]; pos = ids.shape[1] - 1
        with T.no_grad():
            o = m(input_ids=ids, output_hidden_states=True)
            lp = T.log_softmax(o.logits[0, -1], dim=-1)
            hn = float(o.hidden_states[L + 1][0, pos].norm())
        base = float(lp[rome] - lp[tid(true_tok)])
        if d is None:
            return base, 0.0
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.shape[1] <= pos: return out
            t = t.clone(); t[0, pos] = t[0, pos] + alpha * hn * d
            return (t,) + out[1:] if isinstance(out, tuple) else t
        h = blocks[L].register_forward_hook(hook)
        with T.no_grad():
            lp2 = T.log_softmax(m(input_ids=ids).logits[0, -1], dim=-1)
        h.remove()
        return base, float(lp2[rome] - lp2[tid(true_tok)]) - base

    out = {}
    for alpha in a.alphas:
        pts = []
        print(f"\n=== alpha = {alpha} ===")
        print(f"{'rung':>14} {'n':>2} {'base logP':>10} {'effect':>8} "
              f"{'rand mu':>8} {'rand sd':>8} {'z':>7}")
        for rung, items in LADDER:
            es, bs, rs = [], [], []
            for p, t in items:
                b, e = run(p, t, pb, alpha)
                draws = [run(p, t, rv, alpha)[1] for rv in RNDS]
                rmu, rsd = st.mean(draws), st.pstdev(draws) or 1e-9
                es.append(e); bs.append(b); rs.append(rmu)
                pts.append({"rung": rung, "prompt": p, "base": b, "effect": e,
                            "rand_mean": rmu, "rand_sd": rsd, "z": (e - rmu) / rsd})
            zs = [p["z"] for p in pts if p["rung"] == rung]
            sds = [p["rand_sd"] for p in pts if p["rung"] == rung]
            print(f"{rung:>14} {len(items):>2} {st.mean(bs):>10.2f} "
                  f"{st.mean(es):>+8.2f} {st.mean(rs):>+8.2f} {st.mean(sds):>8.2f} "
                  f"{st.mean(zs):>+7.1f}")
        # headroom control: regress effect on base logprob, redo the profile on residuals
        xs = [p["base"] for p in pts]; ys = [p["effect"] for p in pts]
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        r = beta * (sxx ** .5) / ((sum((y - my) ** 2 for y in ys) or 1e-9) ** .5)
        print(f"\n  headroom check: corr(base logP, effect) = {r:+.3f}, slope {beta:+.3f}")
        print(f"  (strongly negative => effect is mostly headroom, not leakage)")
        for p in pts:
            p["resid"] = p["effect"] - (my + beta * (p["base"] - mx))
        print(f"\n{'rung':>14} {'effect':>8} {'resid':>8}   after removing headroom")
        prof = {}
        for rung, _ in LADDER:
            sel = [p for p in pts if p["rung"] == rung]
            prof[rung] = {"effect": st.mean(p["effect"] for p in sel),
                          "resid": st.mean(p["resid"] for p in sel)}
            print(f"{rung:>14} {prof[rung]['effect']:>+8.2f} {prof[rung]['resid']:>+8.2f}")
        out[str(alpha)] = {"points": pts, "profile": prof, "headroom_corr": r}
    a.out.write_text(json.dumps(out))
    print("\nDECAY = superposition signature; FLAT = global push.")

if __name__ == "__main__":
    main()
