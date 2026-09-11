"""Experiment C: do the model's own directions ACT better, or only READ better?

Experiment B asked whether raw directions have more coherent logit-lens
readouts than rotations of themselves. That is a claim about what a direction
*says*. This asks what it *does*: inject each direction into the residual
stream and score the tokens the model actually ends up promoting.

Identical arms, identical bases, identical coherence metric as jlens.privilege
(same seed -> same groups), so the two experiments are directly comparable.
The only change is where the top-10 tokens come from:

    Experiment B   top-10 of  W_U (gamma * d)          -- the readout
    Experiment C   top-10 of  logits(h + a*d) - logits(h)  -- the intervention

The attention-head arm is again the calibration: rotating a head's basis is
exactly function-preserving, so raw and orbit are two names for the same set of
weights and must score the same.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.privilege import Weights, make_bases, arm_matrices
from jlens.corpus_npmi import NPMI, build as build_npmi
from jlens.tokspace import EmbedCoherence

PROMPTS = ["The doctor told the patient that the",
           "In the morning she walked down to the",
           "def compute(values):\n    return",
           "The government announced a new plan to"][:3]


def _blocks(model):
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "layers"):
        return m.layers, m.norm
    return model.transformer.h, model.transformer.ln_f


def run(model_id, n=64, n_draws=8, n_groups=8, topk=10, seed=0, only=None,
        alphas=(0.25, 0.5, 1.0), batch=48, dev=None,
        scorer="unsloth/Llama-3.2-1B", out=Path("out/priv/causal.json")):
    dev = dev or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _blocks(model)

    w = Weights(model_id)
    npmi = NPMI(build_npmi(model_id, n_windows=24576), device="cpu")
    kept = torch.nonzero(npmi.remap >= 0).squeeze(1)
    emb = EmbedCoherence(model_id, kept, npmi.df, scorer_id=scorer)
    keptd = kept.to(dev)
    print(f"{model_id}: d={w.cfg['hidden_size']} vocab_scored={len(kept)} "
          f"dev={dev} n={n} draws={n_draws} alphas={list(alphas)}")

    enc = [tok(p, return_tensors="pt") for p in PROMPTS]
    base_lp, hnorm = [], {}
    with torch.no_grad():
        for e in enc:
            ids = e["input_ids"].to(dev)
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=False, output_hidden_states=True)
            base_lp.append(torch.log_softmax(o.logits[0, -1].float(), -1))
            for li in range(len(blocks)):
                hnorm.setdefault(li, []).append(
                    float(o.hidden_states[li + 1][0, -1].norm()))
    hn = {li: sum(v) / len(v) for li, v in hnorm.items()}

    state = {}                                    # set by the hook each batch

    def hook(mod, inp, o):
        t = o if torch.is_tensor(o) else o[0]
        v = state.get("v")
        if v is None:
            return o
        t = t.clone()
        t[:, -1, :] = t[:, -1, :] + v.to(t.dtype)
        return t if torch.is_tensor(o) else (t,) + tuple(o[1:])

    def delta_logits(D, li, alpha):
        """D (d, m) unit columns -> (m, V) mean log-prob change at the last token."""
        m = D.shape[1]
        acc = torch.zeros(m, model.config.vocab_size, device=dev)
        h = blocks[li].register_forward_hook(hook)
        try:
            for pi, e in enumerate(enc):
                ids = e["input_ids"].to(dev)
                for s in range(0, m, batch):
                    v = (alpha * hn[li]) * D[:, s:s + batch].T.to(dev)   # (b, d)
                    b = v.shape[0]
                    state["v"] = v
                    with torch.no_grad():
                        lg = model(input_ids=ids.expand(b, -1),
                                   attention_mask=torch.ones(b, ids.shape[1],
                                                             dtype=torch.long, device=dev),
                                   use_cache=False).logits[:, -1].float()
                    acc[s:s + b] += torch.log_softmax(lg, -1) - base_lp[pi]
                    state["v"] = None
        finally:
            h.remove()
            state["v"] = None
        return acc / len(enc)

    gen = torch.Generator().manual_seed(seed)
    rows = []
    for arm, lab, A in arm_matrices(w, n, gen, n_groups):
        li = int(lab.split("L")[1].split("h")[0].split("e")[0]) if "L" in lab else 0
        A = A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)
        bases = make_bases(A, n_draws, gen)
        if only:
            bases = {k: v for k, v in bases.items() if k in only}
        for alpha in alphas:
            rec = {"arm": arm, "group": lab, "layer": li, "alpha": alpha,
                   "model": model_id, "scorer": scorer}
            for name, M in bases.items():
                dl = delta_logits(M, li, alpha)                  # (m, V)
                ids = keptd[dl[:, keptd].topk(topk, 1).indices].cpu()
                _, dc = emb.score(ids)
                per = dc.reshape(-1, n)
                # how concentrated is the induced change, and how much of it is
                # predicted by the logit lens?
                pos = dl.clamp(min=0)
                conc = float((pos.topk(topk, 1).values.sum(1) /
                              pos.sum(1).clamp(min=1e-6)).mean())
                rec[name] = {"cos": float(dc.mean()),
                             "q90": float(torch.quantile(per, 0.9, dim=1).mean()),
                             "q90_sd": float(torch.quantile(per, 0.9, dim=1).std()
                                             if per.shape[0] > 1 else 0.0),
                             "conc": conc}
            z = ((rec["raw"]["q90"] - rec["reparam_null"]["q90"]) /
                 max(rec["reparam_null"]["q90_sd"], 1e-9))
            for k in ("svd", "orth_null", "gram_null"):
                rec.setdefault(k, {"q90": float("nan"), "cos": float("nan"),
                                   "conc": float("nan")})
            rec["zq90_vs_reparam_null"] = z
            rows.append(rec)
            print(f"  a={alpha:<4g} {arm:14s} {lab:9s} raw={rec['raw']['q90']:+.4f} "
                  f"orbit={rec['reparam_null']['q90']:+.4f} orth={rec['orth_null']['q90']:+.4f} "
                  f"svd={rec['svd']['q90']:+.4f} zq={z:+6.1f} "
                  f"conc={rec['raw']['conc']:.3f}/{rec['reparam_null']['conc']:.3f}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": model_id, "n": n, "n_draws": n_draws,
                                   "seed": seed, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--out", type=Path, default=Path("out/priv/causal.json"))
    ap.add_argument("--scorer", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these bases, e.g. raw reparam_null")
    a = ap.parse_args()
    run(a.model, n=a.n, n_draws=a.draws, n_groups=a.groups, seed=a.seed,
        alphas=tuple(a.alphas), batch=a.batch, out=a.out, scorer=a.scorer,
        only=set(a.only) if a.only else None)
