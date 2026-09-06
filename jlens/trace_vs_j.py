"""Trace vs J vs edit: does J-amplification predict edit site where tracing fails?

Hase et al 2023: tracing effect vs edit success rho=-0.13 on GPT-J layer 6.
Here per fact, per layer (last subject token, same position throughout):
  1. tracing effect (corrupt subject emb + restore clean h_l)
  2. J-amplification ||J_l d|| for fixed d = norm(W_U[false]-W_U[true])
  3. edit success from injecting that SAME d at that SAME position/norm
Controls (named in advance): random-direction null (100 draws, same norm),
outlier control (d orthogonalized against top activation PC per layer).
Fair comparison: same d, same position, same alpha scale across layers.

Usage: PYTHONPATH=. .venv/bin/python -m jlens.trace_vs_j --facts 8 --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 --out out/trace_vs_j.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"

# (subject, true, false) — all single-token objects on SmolLM2 tokenizer
FACTS = [
    ("France", "Paris", "Rome"),
    ("Italy", "Rome", "Paris"),
    ("Germany", "Berlin", "Paris"),
    ("Spain", "Madrid", "Rome"),
    ("Portugal", "Lisbon", "Madrid"),
    ("Netherlands", "Amsterdam", "Berlin"),
    ("Greece", "Athens", "Rome"),
    ("Poland", "Warsaw", "Berlin"),
    ("Canada", "Ottawa", "Toronto"),
    ("Australia", "Canberra", "Sydney"),
    ("Brazil", "Brasilia", "Lima"),
    ("Argentina", "Buenos Aires", "Lima"),
]


def prompt_of(subj):
    return f"The capital of {subj} is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--facts", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="push = alpha * hn (hn = mean residual norm at layer)")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--nrand", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("out/trace_vs_j.json"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    facts = FACTS[: a.facts]
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    J = {l: blob["J"][l].float() for l in blob["layers"]}
    layers = sorted(J.keys()) if a.layers is None else [l for l in a.layers if l in J]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    g = torch.Generator().manual_seed(a.seed)

    # filter to single-token true/false
    kept = []
    for subj, true, false in facts:
        ti = tok.encode(" " + true, add_special_tokens=False)
        fi = tok.encode(" " + false, add_special_tokens=False)
        if len(ti) == 1 and len(fi) == 1:
            kept.append((subj, true, false, ti[0], fi[0]))
        else:
            print(f"SKIP {subj}/{true}/{false}: multi-token")
    facts = kept
    print(f"{len(facts)} single-token facts, layers {layers}")

    # top-PC per layer for outlier control (20 pile prompts)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    pile = [ds[i]["text"] for i in range(20)]
    top_pc = {}
    Hmean = {}
    with torch.no_grad():
        for l in layers:
            Hs = []
            for t in pile:
                e = tok(t, return_tensors="pt", truncation=True, max_length=64)
                acts = {}

                def _cap(m, i, o, d=acts):
                    d["h"] = (o if torch.is_tensor(o) else o[0]).detach()
                    return None

                h = blocks[l].register_forward_hook(_cap)
                model(**e, use_cache=False)
                h.remove()
                Hs.append(acts["h"][0].float())
            H = torch.cat(Hs, 0)
            Hc = H - H.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            top_pc[l] = Vh[0] / Vh[0].norm()
            Hmean[l] = float(H.norm(dim=-1).mean())

    rows = []
    for subj, true, false, tid, fid in facts:
        p = prompt_of(subj)
        enc = tok(p, return_tensors="pt")
        ids = enc["input_ids"]
        T = ids.shape[1]
        # last subject token: find tokens of " France" inside prompt
        subj_ids = tok.encode(" " + subj, add_special_tokens=False)
        # locate subj span at end (prompt ends with " is", so subject is before last token)
        # simplest robust: last occurrence of subj first-token
        s0 = None
        flat = ids[0].tolist()
        for i in range(len(flat) - len(subj_ids), -1, -1):
            if flat[i:i + len(subj_ids)] == subj_ids:
                s0 = i
                break
        assert s0 is not None, f"subject span not found in {p}"
        subj_pos = s0 + len(subj_ids) - 1  # last subject token

        with torch.no_grad():
            clean_logits = model(**enc, use_cache=False).logits[0, -1].float()
            clean_lp = torch.log_softmax(clean_logits, -1)
        p_clean_true = float(clean_lp[tid].exp())
        base_lr = float(clean_lp[fid] - clean_lp[tid])

        # corrupted: noise on subject-token embeddings
        emb = model.get_input_embeddings().weight.detach()
        noise_vec = torch.randn(len(subj_ids), emb.shape[1], generator=g) * a.noise

        def run_with(restore=None, inject=None, corrupt=True):
            """restore: (layer, clean_h_at_pos) to patch in during corrupted run.
            inject: (layer, d_vec, scale) added at subj_pos.
            corrupt=False for edit runs (clean baseline); True for tracing."""
            handles = []
            if corrupt:
                def emb_hook(m, i, o):
                    e = o.clone()
                    e[0, s0:s0 + len(subj_ids)] += noise_vec.to(e.dtype)
                    return e
                handles.append(model.get_input_embeddings().register_forward_hook(emb_hook))
            cache = {}
            if restore is not None:
                rl, clean_h = restore
                def res_hook(m, i, o, clean_h=clean_h):
                    t = o if torch.is_tensor(o) else o[0]
                    t2 = t.clone()
                    t2[0, subj_pos] = clean_h.to(t2.dtype)
                    return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
                handles.append(blocks[rl].register_forward_hook(res_hook))
            if inject is not None:
                il, dvec, scale = inject
                def inj_hook(m, i, o, dvec=dvec, scale=scale):
                    t = o if torch.is_tensor(o) else o[0]
                    t2 = t.clone()
                    t2[0, subj_pos] += (scale * dvec).to(t2.dtype)
                    return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
                handles.append(blocks[il].register_forward_hook(inj_hook))
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[0, -1].float()
            for h in handles:
                h.remove()
            lp = torch.log_softmax(logits, -1)
            probs = lp.exp()
            return float(probs[tid]), float(probs[fid]), float(lp[fid] - lp[tid])

        p_corr_true, _, _ = run_with()
        denom = (p_clean_true - p_corr_true)
        if abs(denom) < 1e-6:
            print(f"SKIP {subj}: no corruption effect (clean={p_clean_true:.3f} corr={p_corr_true:.3f})")
            continue

        # clean hidden per layer at subj_pos (for restore)
        clean_h = {}
        with torch.no_grad():
            caps = {}

            def _mk(l):
                def _cap(m, i, o):
                    caps[l] = (o if torch.is_tensor(o) else o[0])[0, subj_pos].detach().float()
                    return None
                return _cap

            hs = [blocks[l].register_forward_hook(_mk(l)) for l in layers]
            model(**enc, use_cache=False)
            for h in hs:
                h.remove()
            clean_h = dict(caps)

        d_raw = (W_U[fid] - W_U[tid])
        d_raw = d_raw / d_raw.norm()
        R = torch.nn.functional.normalize(torch.randn(a.nrand, d_raw.shape[0], generator=g), dim=-1)

        print(f"\n{subj}: {true}->{false} clean P(true)={p_clean_true:.3f} corr={p_corr_true:.3f} base logratio={base_lr:+.2f}")
        print(f"{'layer':>5} {'trace':>6} {'J*d':>7} {'J*rand':>7} {'e_logP':>7} {'er_logP':>7} {'eoc':>7}")
        for l in layers:
            p_res, _, _ = run_with(restore=(l, clean_h[l]), corrupt=True)
            trace = (p_res - p_corr_true) / denom

            Jl = J[l]
            jamp = float((Jl @ d_raw).norm())
            jrand = float((Jl @ R.T).norm(dim=0).mean())

            # outlier control direction
            u0 = top_pc[l]
            d_oc = d_raw - (d_raw @ u0) * u0
            d_oc = d_oc / d_oc.norm()
            jamp_oc = float((Jl @ d_oc).norm())

            # position-matched scale: norm of THIS prompt's residual at subj_pos
            hn = float(clean_h[l].norm())
            # edit on CLEAN (no corruption); metric = gain in logP(false)
            with torch.no_grad():
                base_logf = float(torch.log_softmax(
                    model(**enc, use_cache=False).logits[0, -1].float(), -1)[fid])
            _, pf_e, _ = run_with(inject=(l, d_raw, a.alpha * hn), corrupt=False)
            _, pf_r, _ = run_with(inject=(l, R[0], a.alpha * hn), corrupt=False)
            _, pf_o, _ = run_with(inject=(l, d_oc, a.alpha * hn), corrupt=False)
            import math
            edit = math.log(max(pf_e, 1e-9)) - base_logf
            edit_r = math.log(max(pf_r, 1e-9)) - base_logf
            edit_oc = math.log(max(pf_o, 1e-9)) - base_logf

            rows.append({"subj": subj, "true": true, "false": false, "layer": l,
                         "p_clean": p_clean_true, "p_corr": p_corr_true, "p_restored": p_res,
                         "trace": trace, "jamp": jamp, "jrand": jrand, "jamp_oc": jamp_oc,
                         "edit": edit, "edit_rand": edit_r, "edit_oc": edit_oc,
                         "pf_edit": pf_e, "pf_rand": pf_r,
                         "alpha": a.alpha, "hn": hn})
            print(f"{l:>5} {trace:>6.3f} {jamp:>7.2f} {jrand:>7.2f} {edit:>+7.2f} {edit_r:>+7.2f} {edit_oc:>+7.2f}", flush=True)

    # correlations pooled over facts x layers
    import numpy as np
    def corr(x, y):
        x = np.array(x, float); y = np.array(y, float)
        if x.std() < 1e-12 or y.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    tr = [r["trace"] for r in rows]; ed = [r["edit"] for r in rows]
    ja = [r["jamp"] for r in rows]; eo = [r["edit_oc"] for r in rows]; jo = [r["jamp_oc"] for r in rows]
    print("\n==== pooled over %d points ====" % len(rows))
    print(f"corr(trace, edit)   = {corr(tr, ed):+.3f}   <- Hase reported -0.13 (replication target)")
    print(f"corr(J*d, edit)     = {corr(ja, ed):+.3f}   <- the claim being tested")
    print(f"corr(J*d_oc, edit_oc) = {corr(jo, eo):+.3f} <- outlier control; if this dies, the claim dies")
    print(f"mean edit_rand      = {np.mean([r['edit_rand'] for r in rows]):+.3f} (should be ~0)")

    a.out.write_text(json.dumps({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                 "rows": rows,
                                 "corr_trace_edit": corr(tr, ed), "corr_j_edit": corr(ja, ed),
                                 "corr_oc": corr(jo, eo)}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
