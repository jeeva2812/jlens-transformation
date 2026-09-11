"""Redo the two headline demonstrations on the type-correct path.

Both were produced by injecting u at layer l, which is the wrong space. Redone
here with v, like for like: same directions, same prompts, same alpha, u and v
side by side.

  A. the gender direction at layer 24. cos(u,v) = 0.945 there, so u was nearly
     right and this should barely move. That is the control: if it DOES move a
     lot, the correction is doing something other than what I claim.

  B. the US/UK orthography axis, direction 0 at layers 2-20. cos(u,v) is ~0.3
     in early layers, so its published depth curve -- 0.36 at layer 2 rising to
     6.47 at layer 20 -- should be understated at the shallow end. If the curve
     flattens, then "one axis that becomes causally live with depth" was partly
     the bug too.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir, NEUTRAL

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = ["The engineer opened the toolbox because",
           "After the meeting the chief executive said that",
           "The doctor finished the operation and then",
           "The nurse looked at the chart and then"]


def main():
    blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    HE = tok.encode(" he", add_special_tokens=False)[0]
    SHE = tok.encode(" she", add_special_tokens=False)[0]

    def actnorm(l):
        ids = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(model, [l], blob["target"]) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=False)
            return float(cap.h[l][0].norm(dim=-1).mean())

    def probs_and_text(p):
        enc = tok(p, return_tensors="pt")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            pr = torch.softmax(lg.float(), -1)
            g = model.generate(**enc, max_new_tokens=14, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        he, she = pr[HE].item(), pr[SHE].item()
        return he / (he + she), tok.decode(g[0][enc["input_ids"].shape[1]:]).strip()

    # ---------------- A. gender ----------------
    L = 24
    J = blob["J"][L].float()
    U, S, Vh = torch.linalg.svd(J)
    u, v = U[:, 0], Vh[0]
    hn = actnorm(L)
    print("="*78)
    print(f"A. GENDER DIRECTION, layer {L}   cos(u,v) = "
          f"{abs(torch.dot(u,v).item()):.3f}  (u was nearly right here)")
    print("="*78)
    for p in PROMPTS:
        b, bt = probs_and_text(p)
        print(f"\n  {p!r}")
        print(f"    base        P(he|he,she) = {b:.2f}   {bt[:62]!r}")
        for name, d in [("u", u), ("v", v)]:
            for sgn, lab in [(+1, "+"), (-1, "-")]:
                with AddDir(model, L, sgn * d, 4.0 * hn):
                    q, t = probs_and_text(p)
                print(f"    steer {name}{lab}    P(he|he,she) = {q:.2f}   {t[:62]!r}")

    # ---------------- B. orthography ----------------
    print("\n" + "="*78)
    print("B. US/UK ORTHOGRAPHY AXIS, direction 0, by depth")
    print("="*78)
    def logratio(pos, neg):
        tot = 0.0
        for p in NEUTRAL:
            enc = tok(p, return_tensors="pt")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
            tot += float(lp[pos] - lp[neg])
        return tot / len(NEUTRAL)

    W_U = model.get_output_embeddings().weight.detach()
    _, norm = _find_blocks_and_norm(model)
    print(f"\n{'layer':>6} {'cos(u,v)':>9} {'+pole':<13}{'-pole':<13}"
          f"{'shift[u]':>9}{'shift[v]':>9}{'ratio':>7}")
    print("-"*68)
    rows = []
    for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        Jl = blob["J"][l].float()
        Ul, Sl, Vl = torch.linalg.svd(Jl)
        u_, v_ = Ul[:, 0], Vl[0]
        with torch.no_grad():
            pp = torch.softmax(norm(u_.to(W_U.dtype)) @ W_U.T, -1)
            pn = torch.softmax(norm(-u_.to(W_U.dtype)) @ W_U.T, -1)
        pos, neg = int(pp.argmax()), int(pn.argmax())
        base = logratio(pos, neg)
        hn_ = actnorm(l)
        with AddDir(model, l, u_, 0.01 * hn_):
            su = logratio(pos, neg) - base
        with AddDir(model, l, v_, 0.01 * hn_):
            sv = logratio(pos, neg) - base
        c = abs(torch.dot(u_, v_).item())
        print(f"{l:>6} {c:>9.3f} {tok.decode(pos)[:12]:<13}{tok.decode(neg)[:12]:<13}"
              f"{su:>9.2f}{sv:>9.2f}{sv/max(abs(su),1e-9):>7.2f}")
        rows.append({"layer": l, "cos_uv": c, "shift_u": su, "shift_v": sv})
    Path("out/redo_corrected.json").write_text(json.dumps(rows, indent=1))
    print("\nwrote out/redo_corrected.json")


if __name__ == "__main__":
    main()
