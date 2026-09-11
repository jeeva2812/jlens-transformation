"""Can we steer the directions that turned out to be interpretable?

This is the second half of the claim. The first half -- that most subspaces are
not interpretable -- is measured in jlens/leftover.py. Here we take only the
directions that scored WELL there and ask whether being readable buys causal
control, in two places:

  TOP        the leading directions of J, where everyone already looks
  LEFTOVER   leading directions of J after removing both the top of J and the
             activation subspace -- structure nobody is looking at

and we ask whether the leftover ones steer as well as the top ones. If they do,
the usual practice of reading only the top of the spectrum is leaving usable
directions on the table.

The test is not "did its own top token go up". Pushing a direction always raises
its own top tokens; that is arithmetic. We build a HELD-OUT set: tokens near the
centroid of the direction's top tokens but deliberately excluded from the
readout's own top 200. If the direction means one thing, those are more of the
same thing and should rise together.

  lift = (held-out tokens moved) - (rank-matched control tokens moved)

Controls: a random direction of identical norm through the identical pipeline,
control tokens matched on how likely they already were, and the model's loss on
unrelated text -- a direction that "works" by breaking the model is not steering.

Type-correctness: J = U S V^T maps layer-l space to target space. u unembeds
(what the direction WRITES); v is what you inject (what it READS). We inject v.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.interp_score import Scorer
from jlens.leftover import ENGLISH, CODE, blocks_of

COH_TEXT = "The committee met on Tuesday to discuss the budget revisions."
TOPK, NEAR, EXCL = 12, 12, 200


class Add:
    def __init__(self, model, layer, d, alpha):
        self.d = F.normalize(d.float(), dim=0) * alpha
        self.h = blocks_of(model)[layer].register_forward_hook(self._f)
    def _f(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t2 = t + self.d.to(t.device, t.dtype)
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--scored", type=Path, default=Path("out/rare/lo_smollm2_english.json"),
                    help="output of jlens.leftover -- tells us which directions were readable")
    ap.add_argument("--other", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--n", type=int, default=25, help="per arm, most interpretable first")
    ap.add_argument("--content", action="store_true",
                    help="keep only directions pointing at words, not at punctuation")
    ap.add_argument("--bands", action="store_true",
                    help="instead of the top n, take n from each third of the "
                         "readability range -- this is what answers 'does being "
                         "readable buy steering', because it needs the contrast")
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="steering size as a fraction of the median residual norm")
    ap.add_argument("--nnull", type=int, default=100)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", type=Path, default=Path("out/rare/steer_smollm2.json"))
    a = ap.parse_args()

    d = json.load(open(a.scored))
    k, m = d["k"], d["m"]
    PROMPTS = {"english": ENGLISH, "code": CODE}[d.get("prompts", "english")]

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    sc = Scorer(W_U, tok, other_model=a.other, nnull=a.nnull)

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js = blob["J"]

    # rebuild the same activation subspace the scoring run used
    layers = sorted({r["layer"] for r in d["rows"]})
    store = {l: [] for l in layers}; hooks = []
    def mk(l):
        def f(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            store[l].append(t[0].detach().float().cpu())
        return f
    for l in layers: hooks.append(blocks_of(model)[l].register_forward_hook(mk(l)))
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks: h.remove()

    def final_lp():
        out = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.log_softmax(o.logits[0, -1].float(), -1).cpu())
        return torch.stack(out).mean(0)

    coh_ids = tok(COH_TEXT, return_tensors="pt")["input_ids"].to(dev)
    def text_loss():
        with torch.no_grad():
            o = model(input_ids=coh_ids, attention_mask=torch.ones_like(coh_ids), use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), coh_ids[0, 1:]))

    base_lp, base_loss = final_lp(), text_loss()
    lp_order = torch.argsort(base_lp, descending=True)
    lp_rank = torch.empty_like(lp_order); lp_rank[lp_order] = torch.arange(len(lp_order))

    # cache the decompositions once per layer
    cache = {}
    def decomp(l):
        if l in cache: return cache[l]
        J = Js[l].float()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        H = torch.cat([x[1:] for x in store[l]], 0)          # drop the sink at pos 0
        A = H - H.mean(0, keepdim=True)
        Va = torch.linalg.svd(A, full_matrices=False)[2][:m]
        Q, _ = torch.linalg.qr(torch.cat([Vh[:k], Va], 0).T)
        Uc, Sc_, Vc = torch.linalg.svd(J @ (torch.eye(J.shape[1]) - Q @ Q.T),
                                       full_matrices=False)
        scale = float(torch.stack([x[1:].norm(dim=-1).median() for x in store[l]]).mean())
        cache[l] = (U, S, Vh, Uc, Sc_, Vc, scale)
        return cache[l]

    g = torch.Generator().manual_seed(0)

    def probe(l, inject, ids_top, alpha):
        cen = F.normalize(sc.Wn[ids_top].mean(0), dim=0)
        sims = sc.Wn @ cen
        r = sc.Wc @ F.normalize(sc.Wc.T @ torch.zeros(1), dim=0) if False else None
        return sims

    rows = []
    for arm in ("top", "leftover"):
        cand = [r for r in d["rows"] if r["arm"] == arm and r["xm_z"] == r["xm_z"]]
        if a.content:
            n0 = len(cand)
            cand = [r for r in cand if Scorer.is_content(r["top"])]
            print(f"  {arm}: {len(cand)}/{n0} directions point at words rather than "
                  f"punctuation")
        cand.sort(key=lambda r: -r["xm_z"])
        if a.bands:
            t = len(cand) // 3
            pool = []
            for nm, chunk in (("high", cand[:t]), ("mid", cand[t:2*t]), ("low", cand[2*t:])):
                step = max(1, len(chunk) // a.n)
                take = chunk[::step][:a.n]
                for r in take: r["band"] = nm
                pool += take
                print(f"    {nm:5s} readability {take[-1]['xm_z']:5.1f}..{take[0]['xm_z']:5.1f}"
                      f"  n={len(take)}")
        else:
            pool = cand[:a.n]
            for r in pool: r["band"] = "top"
        print(f"\n### {arm}: {len(pool)} most interpretable directions "
              f"(cross-model z from {pool[-1]['xm_z']:.0f} to {pool[0]['xm_z']:.0f})")
        for rec in pool:
            l, i = rec["layer"], rec["dir"]
            U, S, Vh, Uc, Sc_, Vc, scale = decomp(l)
            u = (U[:, i] if arm == "top" else Uc[:, i])
            v = (Vh[i] if arm == "top" else Vc[i])
            alpha = a.alpha * scale

            rr = sc.Wc @ u
            top = torch.topk(rr, TOPK).indices
            cen = F.normalize(sc.Wn[top].mean(0), dim=0)
            sims = sc.Wn @ cen
            sims[torch.topk(rr, EXCL).indices] = -2.0
            near = torch.topk(sims, NEAR).indices
            used = set(int(t) for t in near) | set(int(t) for t in top)
            ctrl = []
            for w in lp_rank[near].tolist():
                for off in range(500):
                    for c in (int(lp_order[min(w + off, len(lp_order) - 1)]),
                              int(lp_order[max(w - off, 0)])):
                        if c not in used and float(sims[c]) < 0.15:
                            ctrl.append(c); used.add(c); break
                    else: continue
                    break
            ctrl = torch.tensor(ctrl[:NEAR])

            def run(direction):
                with Add(model, l, direction, alpha):
                    lp, loss = final_lp(), text_loss()
                dd = lp - base_lp
                return (float(dd[top[0]]), float(dd[near].mean() - dd[ctrl].mean()),
                        loss - base_loss)

            own, lift, dloss = run(v)
            rv = F.normalize(torch.randn(W_U.shape[1], generator=g), dim=0)
            r_own, r_lift, r_dloss = run(rv)
            rows.append(dict(arm=arm, band=rec.get("band", "top"),
                             layer=l, dir=i, xm_z=rec["xm_z"], coh_z=rec["coh_z"],
                             sigma=rec["sigma"], own=own, lift=lift, dloss=dloss,
                             r_own=r_own, r_lift=r_lift, r_dloss=r_dloss,
                             top=rec["top"], near=[tok.decode([int(t)]) for t in near]))
            print(f"  L{l:<3d} d{i:<3d} xm_z={rec['xm_z']:+6.0f} own={own:+6.2f} "
                  f"lift={lift:+6.2f} (rand {r_lift:+5.2f}) dloss={dloss:+5.2f} "
                  f"| {' '.join(repr(x) for x in rec['top'][:5])}", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(dict(model=a.model, scored=str(a.scored),
                                             alpha=a.alpha, rows=rows), indent=1))

    print(f"\n{'arm':10s} {'band':6s} {'n':>3s} {'readability':>12s} {'mean lift':>10s} "
          f"{'rand lift':>10s} {'steered':>10s}")
    for arm in ("top", "leftover"):
        for band in ("high", "mid", "low", "top"):
            gg = [r for r in rows if r["arm"] == arm and r["band"] == band]
            if not gg: continue
            w = sum(1 for r in gg if r["lift"] > max(r["r_lift"], 0.2))
            print(f"{arm:10s} {band:6s} {len(gg):3d} "
                  f"{st.mean(r['xm_z'] for r in gg):12.1f} "
                  f"{st.mean(r['lift'] for r in gg):10.2f} "
                  f"{st.mean(r['r_lift'] for r in gg):10.2f} "
                  f"{w:4d}/{len(gg):<4d} {w/len(gg):3.0%}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
