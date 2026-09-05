"""Jeeva's idea: use the eigenvector to decide WHAT, the SVD to decide HOW.

Two established results pull in opposite directions. Eigenvectors read better
(39.6% clear an axis probe vs 20.8%) because they are the type-correct object for
a map from the residual stream to itself. Singular vectors steer better (75% vs
44%) because Weyl guarantees sigma_1 >= |lambda_1|, so at fixed injection norm
they deliver the larger perturbation.

The proposal is to stop choosing. An eigenvector v_e names the semantics you
want to produce. To produce it by injecting at layer l you need J x proportional
to v_e, and expanding v_e in the singular basis, v_e = sum_i c_i u_i, gives

    x = sum_i (c_i / sigma_i) v_i

Injecting x delivers v_e's semantics through the SVD's machinery. Four ways to
approximate it, all judged on the SAME token pair -- the one the EIGENVECTOR's
readout names, since that is the semantics we are trying to deliver:

  eigen        inject v_e directly                        (the 44% baseline)
  closest-u    inject the v_i whose u_i best aligns with v_e   (Jeeva's version)
  gain-wtd     inject the v_i maximising sigma_i * <u_i, v_e>  (weighted by gain)
  pseudo-inv   inject sum over top-k of (c_i / sigma_i) v_i    (the full thing)

If any of the last three beats the eigenvector on ITS OWN token pair, the
dissociation is not a trade-off but a division of labour.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir, NEUTRAL

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=32, help="terms in the pseudo-inverse")
    ap.add_argument("--out", type=Path, default=Path("out/hybrid_steer.json"))
    a = ap.parse_args()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, norm = _find_blocks_and_norm(m)
    W_U = m.get_output_embeddings().weight.detach()
    target = blob["target"]

    def logratio(pos, neg):
        t = 0.0
        for p in NEUTRAL:
            e = tok(p, return_tensors="pt")
            with torch.no_grad():
                lp = torch.log_softmax(m(**e).logits[0, -1].float(), -1)
            t += float(lp[pos] - lp[neg])
        return t / len(NEUTRAL)

    g = torch.Generator().manual_seed(0)
    rows = []
    print(f"{'layer':>5}{'eig':>4}{'|lam|':>7}  {'+pole':<12}{'-pole':<12}"
          f"{'eigen':>8}{'closest':>9}{'gain-wtd':>10}{'pinv':>8}{'rand':>7}"
          f"   {'cos(u*,ve)':>10}")
    print("-" * 96)
    for l in a.layers:
        if l not in blob["J"] or l >= target:
            continue
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
        ids0 = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(m, [l], target) as cap:
            with torch.no_grad():
                m(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
            hn = float(cap.h[l][0].norm(dim=-1).mean())

        for rank, j in enumerate(eo.tolist()):
            ve = V[:, j].real
            ve = ve / ve.norm()
            # the token pair is named by the EIGENVECTOR's readout -- this is the
            # semantics every method is being asked to deliver
            tv = (J @ ve).to(W_U.dtype)
            with torch.no_grad():
                pos = int(torch.softmax(norm(tv) @ W_U.T, -1).argmax())
                neg = int(torch.softmax(norm(-tv) @ W_U.T, -1).argmax())
            if pos == neg:
                continue
            c = U.T @ ve                       # coefficients of v_e in the u basis
            i_close = int(c.abs().argmax())
            i_gain = int((S * c.abs()).argmax())
            k = min(a.topk, S.numel())
            pinv = (Vh[:k].T @ (c[:k] / S[:k]))
            # SIGN MATTERS. Injecting v_i produces sigma_i * u_i; if the
            # coefficient c_i is negative then u_i is ANTI-aligned with the
            # eigenvector, and injecting v_i unsigned delivers the opposite
            # semantics. Selecting on |c_i| and then ignoring its sign was a bug
            # that made both rank-1 methods look worse than they are.
            cands = {"eigen": ve,
                     "closest": torch.sign(c[i_close]) * Vh[i_close],
                     "gainwtd": torch.sign(c[i_gain]) * Vh[i_gain],
                     "pinv": pinv / pinv.norm()}
            base = logratio(pos, neg)
            sh = {}
            for nm, d in cands.items():
                with AddDir(m, l, d, a.alpha * hn):
                    sh[nm] = logratio(pos, neg) - base
            r = torch.randn(J.shape[0], generator=g)
            with AddDir(m, l, r, a.alpha * hn):
                sh["rand"] = logratio(pos, neg) - base
            rows.append({"layer": l, "eig": rank, "abs_lam": float(w[j].abs()),
                         "pos": tok.decode(pos), "neg": tok.decode(neg),
                         "cos_u_ve": float(c.abs().max()),
                         "sigma_closest": float(S[i_close]),
                         "sigma_gain": float(S[i_gain]), **sh})
            print(f"{l:>5}{rank:>4}{float(w[j].abs()):>7.2f}  "
                  f"{tok.decode(pos)[:11]:<12}{tok.decode(neg)[:11]:<12}"
                  f"{sh['eigen']:>8.2f}{sh['closest']:>9.2f}{sh['gainwtd']:>10.2f}"
                  f"{sh['pinv']:>8.2f}{sh['rand']:>7.2f}   {c.abs().max():>10.3f}",
                  flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    import statistics as st
    def passes(r, k):
        return abs(r[k]) > 0.5 and abs(r[k]) > 4 * abs(r["rand"])
    print(f"\n{'method':<12}{'passes':>10}{'rate':>8}{'median |shift|':>16}")
    print("-" * 48)
    for nm in ["eigen", "closest", "gainwtd", "pinv"]:
        p = sum(passes(r, nm) for r in rows)
        print(f"{nm:<12}{p:>7}/{len(rows):<3}{p/len(rows)*100:>7.1f}%"
              f"{st.median(abs(r[nm]) for r in rows):>16.2f}")
    print(f"\n  median cos(u*, v_e) = "
          f"{st.median(r['cos_u_ve'] for r in rows):.3f}  "
          f"(how well the best singular output direction matches the eigenvector)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
