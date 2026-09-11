"""Compose Rome-Paris subspace steering: keep SVD components with good gap.

Per layer l, with J = U S Vh and raw d = norm(W_U[false]-W_U[true]):
  gap(v_i) = logP(false | J v_i) - logP(true | J v_i)   (full-vocab readout)
  S = top-k components by gap
  d_star = sum_{i in S} (v_i . d) v_i, renormalized      (denoised d)
  pullback p = J^T w / ||J^T w||, w = norm(W_U[f]-W_U[t]) (optimal linear steerer)
Test: inject d_star vs raw d vs pullback vs random at last-subject-token,
clean run. Metrics: efficacy dlogP(false) on fact prompt, specificity
dlogP(false) on neighborhood prompt (should be ~0), coherence NLL shift.

Usage: PYTHONPATH=. .venv/bin/python -m jlens.compose_steer --layers 10 12 14 --topk 8 --alpha 1.0 --out out/compose_steer.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
FACT = ("France", "Paris", "Rome")
NEIGHBOR = "The Louvre is in"  # should stay Paris


def readout_gaps(J, Vh, norm, W_U, tid, fid):
    """gap(v_i) for every singular component. Returns tensor (d,)."""
    with torch.no_grad():
        T = (Vh @ J.T).T  # col i = J v_i = s_i u_i  (d x d)
        Tn = norm(T.T).to(W_U.dtype) @ W_U.T  # (d x vocab) logits
        lp = torch.log_softmax(Tn.float(), -1)
        return (lp[:, fid] - lp[:, tid]).float()  # (d,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 12, 14])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/compose_steer.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(a.seed)

    subj, true, false = FACT
    tid = tok.encode(" " + true, add_special_tokens=False)[0]
    fid = tok.encode(" " + false, add_special_tokens=False)[0]
    prompt = f"The capital of {subj} is"
    enc = tok(prompt, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    subj_ids = tok.encode(" " + subj, add_special_tokens=False)
    s0 = next(i for i in range(len(ids)) if ids[i:i + len(subj_ids)] == subj_ids)
    spos = s0 + len(subj_ids) - 1
    nenc = tok(NEIGHBOR, return_tensors="pt")
    ntid = tid  # neighbor answer is also Paris

    d_raw = (W_U[fid] - W_U[tid])
    d_raw = d_raw / d_raw.norm()
    w_out = d_raw.clone()
    rnd = torch.randn(d_raw.shape[0], generator=g)
    rnd = rnd / rnd.norm()

    def clean_logp(enc_, tid_, fid_):
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc_, use_cache=False).logits[0, -1].float(), -1)
        return float(lp[tid_]), float(lp[fid_])

    def inject_logp(enc_, layer, dvec, scale, pos=None):
        pos = spos if enc_ is enc else enc_["input_ids"].shape[1] - 1
        def hook(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            t2 = t.clone()
            t2[0, pos] += (scale * dvec).to(t2.dtype)
            return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
        h = blocks[layer].register_forward_hook(hook)
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc_, use_cache=False).logits[0, -1].float(), -1)
        h.remove()
        return float(lp[tid]), float(lp[fid])

    # clean hidden norms at subj pos per layer (position-matched scale)
    clean_h = {}
    with torch.no_grad():
        caps = {}
        def mk(l):
            def cap(m, i, o):
                caps[l] = (o if torch.is_tensor(o) else o[0])[0, spos].detach().float()
                return None
            return cap
        hs = [blocks[l].register_forward_hook(mk(l)) for l in a.layers]
        model(**enc, use_cache=False)
        for h in hs:
            h.remove()
        clean_h = dict(caps)

    base_t, base_f = clean_logp(enc, tid, fid)
    nb_t, nb_f = clean_logp(nenc, ntid, fid)
    print(f"fact: {prompt} P(Paris)={math.exp(base_t):.4f} P(Rome)={math.exp(base_f):.6f}")
    print(f"neighbor: {NEIGHBOR} P(Paris)={math.exp(nb_t):.4f}")

    rows = []
    for l in a.layers:
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        V = Vh.T  # col i = v_i
        gaps = readout_gaps(J, Vh, norm, W_U, tid, fid)
        order = torch.argsort(gaps, descending=True)
        Sset = order[: a.topk].tolist()
        npos = int((gaps > 0).sum())

        # compose: keep projection of d_raw onto good-gap components
        coeff = V.T @ d_raw  # (d,) c_i = v_i . d
        d_star = (V[:, Sset] @ coeff[Sset])
        d_star = d_star / d_star.norm()
        # pullback: optimal linear steerer
        pb = (J.T @ w_out)
        pb = pb / pb.norm()
        # how much of d_raw survives in top-k gap subspace?
        kept_frac = float(coeff[Sset].pow(2).sum() / coeff.pow(2).sum())

        hn = float(clean_h[l].norm())
        cands = {"raw": d_raw, "composed": d_star, "pullback": pb, "random": rnd}
        res = {"layer": l, "kept_frac": round(kept_frac, 3), "npos": npos,
               "top_gaps": [round(float(gaps[i]), 2) for i in Sset],
               "top_idx": Sset}
        print(f"\nL{l}: kept_frac={kept_frac:.3f} npos={npos}/{J.shape[0]} top_idx={Sset[:5]} gaps={[round(float(gaps[i]),1) for i in Sset[:5]]}")
        for name, dvec in cands.items():
            pt, pf = inject_logp(enc, l, dvec, a.alpha * hn)
            nt, nf = inject_logp(nenc, l, dvec, a.alpha * hn)
            eff = pf - base_f  # dlogP(Rome) on fact
            spec = nf - nb_f   # dlogP(Rome) on neighbor (want ~0)
            res[name] = {"eff": round(eff, 3), "spec": round(spec, 3),
                         "p_rome": round(math.exp(pf), 5)}
            print(f"  {name:<9} eff={eff:+.3f} spec={spec:+.3f} P(Rome)={math.exp(pf):.5f}", flush=True)
        rows.append(res)

    a.out.write_text(json.dumps({"fact": FACT, "neighbor": NEIGHBOR,
                                 "alpha": a.alpha, "topk": a.topk, "rows": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
