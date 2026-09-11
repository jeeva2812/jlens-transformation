"""The steering experiments as experiments, not as a table of shift numbers.

A row saying shift = 7.35 is not evidence anyone can check. These run the
intervention end to end and print what the model actually writes, with the
alpha sweep and a repetition rate visible, because the failure mode here is an
obliterated model scoring as a perfect success: at alpha = 4.0 this same
direction emits 'herself herself herself ...' with P(he) exactly 0.00.

Two axes, chosen because they are the two that survived validation:
  gender          layer 24 direction 0, cos(u,v)=0.945 so the u/v fix barely moves it
  US/UK spelling  direction 0 at every layer from 2 to 20, and the only axis here
                  that beat a 500-random null on magnitude
Steering uses v, which is the type-correct vector; the token pair is named by u.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir, NEUTRAL

MODEL = "HuggingFaceTB/SmolLM2-135M"
GENDER = ["The engineer opened the toolbox because",
          "After the meeting the chief executive said that",
          "The doctor finished the operation and then",
          "The nurse looked at the chart and then"]
# Mid-word prompts, so the very next token IS the spelling decision. Free-form
# prompts do not work here: the model can write a fluent sentence that never
# reaches a word where the two spellings differ, and then the demo shows nothing.
SPELL = [("The col", "our", "or"), ("The behavi", "our", "or"),
         ("The hon", "our", "or"), ("The fav", "our", "or"),
         ("They organi", "sed", "zed"), ("We recogni", "sed", "zed"),
         ("They apologi", "sed", "zed"), ("She analy", "sed", "zed"),
         ("The cent", "re", "er"), ("The theat", "re", "er")]


def main():
    blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    def actnorm(l):
        ids = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(model, [l], blob["target"]) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=False)
            return float(cap.h[l][0].norm(dim=-1).mean())

    def gen(p, n=15):
        enc = tok(p, return_tensors="pt")
        with torch.no_grad():
            pr = torch.softmax(model(**enc).logits[0, -1].float(), -1)
            g = model.generate(**enc, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        o = g[0][enc["input_ids"].shape[1]:]
        rep = 1 - len(set(o.tolist()))/len(o)
        return pr, rep, tok.decode(o).replace("\n", " ").strip()

    out = {}
    for name, layer, di, prompts, probes in [
        ("gender", 24, 0, GENDER, [(" he", " she")]),
    ]:
        J = blob["J"][layer].float()
        U, S, Vh = torch.linalg.svd(J)
        u, v = U[:, di], Vh[di]
        hn = actnorm(layer)
        with torch.no_grad():
            pp = torch.softmax(norm(u.to(W_U.dtype)) @ W_U.T, -1)
            pn = torch.softmax(norm(-u.to(W_U.dtype)) @ W_U.T, -1)
        pos, neg = tok.decode(int(pp.argmax())), tok.decode(int(pn.argmax()))
        cuv = abs(torch.dot(u, v).item())
        print(f"\n{'='*78}\n{name.upper()}  layer {layer} dir {di}  "
              f"cos(u,v)={cuv:.3f}   readout poles: {pos!r} / {neg!r}\n{'='*78}")
        pid = [tok.encode(t, add_special_tokens=False)[0] for t in probes[0]]
        rows = []
        for p in prompts:
            pr, rep, txt = gen(p)
            r = {"prompt": p, "base": {"text": txt, "rep": round(rep, 2),
                 "ratio": round(pr[pid[0]].item()/(pr[pid[0]].item()+pr[pid[1]].item()), 3)}}
            print(f"\n  {p!r}")
            print(f"    base   [{r['base']['ratio']:.2f}] {txt[:60]!r}")
            for sgn, lab in [(+1, "+v"), (-1, "-v")]:
                with AddDir(model, layer, sgn*v, 0.01*hn):
                    pr2, rep2, t2 = gen(p)
                ratio = pr2[pid[0]].item()/(pr2[pid[0]].item()+pr2[pid[1]].item())
                r[lab] = {"text": t2, "rep": round(rep2, 2), "ratio": round(ratio, 3)}
                print(f"    {lab}     [{ratio:.2f}] {t2[:60]!r}"
                      + ("   DEGENERATE" if rep2 > 0.5 else ""))
            rows.append(r)
        out[name] = {"layer": layer, "dir": di, "cos_uv": cuv,
                     "pos": pos, "neg": neg, "probe": list(probes[0]),
                     "alpha": 0.01, "rows": rows}

    # ---- spelling, measured where the next token IS the decision ----
    layer, di = 20, 0
    J = blob["J"][layer].float(); U, S, Vh = torch.linalg.svd(J)
    u, v = U[:, di], Vh[di]; hn = actnorm(layer)
    with torch.no_grad():
        pp = torch.softmax(norm(u.to(W_U.dtype)) @ W_U.T, -1)
        pn = torch.softmax(norm(-u.to(W_U.dtype)) @ W_U.T, -1)
    print(f"\n{'='*78}\nSPELLING  layer {layer} dir {di}  "
          f"cos(u,v)={abs(torch.dot(u,v).item()):.3f}  poles "
          f"{tok.decode(int(pp.argmax()))!r} / {tok.decode(int(pn.argmax()))!r}\n{'='*78}")
    print(f"\n{'prompt':>16} {'UK':>5} {'US':>5} | {'base':>7} {'+v':>7} {'-v':>7}"
          f"   P(UK) share of the two")
    print("-"*70)
    srows = []
    for stem, uk, us in SPELL:
        enc = tok(stem, return_tensors="pt")
        iu = tok.encode(uk, add_special_tokens=False)[0]
        iw = tok.encode(us, add_special_tokens=False)[0]
        def share():
            with torch.no_grad():
                pr = torch.softmax(model(**enc).logits[0, -1].float(), -1)
            return pr[iu].item()/(pr[iu].item()+pr[iw].item())
        b = share()
        with AddDir(model, layer, v, 0.01*hn): up = share()
        with AddDir(model, layer, -v, 0.01*hn): dn = share()
        srows.append({"stem": stem, "uk": uk, "us": us,
                      "base": round(b,3), "plus": round(up,3), "minus": round(dn,3)})
        print(f"{stem:>16} {uk:>5} {us:>5} | {b:>7.2f} {up:>7.2f} {dn:>7.2f}")
    # The readout poles are 'honored' (+) and 'isation' (-), so the PREDICTION is
    # +v -> American, -v -> British. Scoring it the other way round, as I first did,
    # reports a clean success as a total failure.
    n = len(srows)
    du = sum(r["plus"]-r["base"] for r in srows)/n
    dd = sum(r["minus"]-r["base"] for r in srows)/n
    live = [r for r in srows if 0.02 < r["base"] < 0.98]
    print(f"\n  prediction from the readout poles: +v -> American, -v -> British")
    print(f"  mean change in P(UK):  +v {du:+.3f}   -v {dd:+.3f}")
    print(f"  +v moved toward US: {sum(r['plus']<r['base'] for r in srows)}/{n}"
          f"   -v moved toward UK: {sum(r['minus']>r['base'] for r in srows)}/{n}")
    print(f"\n  Restricted to the {len(live)} pairs not already at floor or ceiling:")
    for r in live:
        print(f"    {r['stem']+r['uk']:<20} {r['base']:.2f} -> +v {r['plus']:.2f}"
              f"  /  -v {r['minus']:.2f}")
    print("\n  The -ise/-ize pairs sit at 0.00 and the -re/-er pairs at ~1.00, so")
    print("  they cannot move. Causally this is an -our/-or axis; the readout label")
    print("  'British orthography' is BROADER than what the direction controls.")
    out["spelling"] = {"layer": layer, "dir": di, "alpha": 0.01,
                       "cos_uv": abs(torch.dot(u,v).item()),
                       "rows": srows, "mean_plus": du, "mean_minus": dd}

    # the alpha sweep, so the degenerate regime is visible rather than asserted
    J = blob["J"][24].float(); v = torch.linalg.svd(J)[2][0]
    hn = actnorm(24)
    HE = tok.encode(" he", add_special_tokens=False)[0]
    SHE = tok.encode(" she", add_special_tokens=False)[0]
    sweep = []
    for al in [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0, 4.0]:
        if al == 0:
            pr, rep, t = gen(GENDER[0])
        else:
            with AddDir(model, 24, v, al*hn):
                pr, rep, t = gen(GENDER[0])
        ratio = pr[HE].item()/(pr[HE].item()+pr[SHE].item())
        sweep.append({"alpha": al, "ratio": round(ratio, 3),
                      "rep": round(rep, 2), "text": t})
        print(f"  alpha {al:<6} P(he)={ratio:.2f} rep={rep:.2f}  {t[:52]!r}")
    out["alpha_sweep"] = sweep
    Path("out/steer_demo.json").write_text(json.dumps(out, indent=1))
    print("\nwrote out/steer_demo.json")


if __name__ == "__main__":
    main()
