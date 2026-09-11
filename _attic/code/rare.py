"""Are J-Lens subspaces interpretable, and does interpretability buy steering?

The claim under test: "subspaces of J are rarely interpretable, but when they
are, we can steer them."

Three things were missing from every earlier run.

1. "Interpretable" was scored against a hand-written list of ~35 concepts, so a
   direction meaning something not on the list counted as meaningless. Here the
   score is label-free and comes in two flavours:
     coh_z  are its top tokens closer together than tokens of the same rarity?
     xm_z   are they still close together in a DIFFERENT model's embedding?
   The second is the strict one: it cannot be satisfied by a quirk of this
   model's tokenizer or by a frequency artefact, because the other model was
   trained separately on a different vocabulary.

2. We only ever steered the directions we had already labelled. If the
   unlabelled ones steer just as well, interpretability buys nothing. Here every
   direction gets the identical test.

3. Everything was pooled across layers. cos(u,v) rises with depth because J
   approaches the identity near the target, and so does steering strength, so a
   pooled correlation between them is mostly depth. Every number here is also
   reported WITHIN layer.

The steering test is deliberately not "did its own top token go up". Pushing a
direction always raises its own top tokens; that is arithmetic, not meaning.
We build a HELD-OUT set: tokens near the centroid of the top tokens but absent
from the readout's top 200. For a direction that means one thing those are more
of the same thing and should rise together; for a blend the centroid is mush.

  own    its own top token moved            (mechanical, expected everywhere)
  lift   held-out neighbours moved MORE than rank-matched control tokens

Type-correctness: J = U S V^T maps layer-l space to target space, so u lives in
target space (it unembeds, it is what the direction WRITES) and v lives in
layer-l space (it is what you inject, what the direction READS). We inject v.

Controls per direction: a random direction of identical norm through the same
pipeline; rank-matched control tokens; and the model's loss on unrelated text,
because a direction that "works" by breaking the model is not steering.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The report was finished on Tuesday and",
    "After the long walk home,",
    "In the middle of the afternoon",
    "The old building on the corner",
    "When the meeting finally ended,",
    "She opened the letter and",
    "Nobody expected the answer to be",
    "The train pulled into the station and",
]
COH_TEXT = "The committee met on Tuesday to discuss the budget revisions."
TOPK, NEAR, EXCL = 12, 12, 200


def blocks_of(model):
    m = model.model if hasattr(model, "model") else model
    return m.layers


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
    ap.add_argument("--other", default="unsloth/Llama-3.2-1B",
                    help="second model, embeddings only, for the cross-model check")
    ap.add_argument("--ndirs", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="steering size as a fraction of the median residual norm")
    ap.add_argument("--nnull", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", type=Path, default=Path("out/rare/smollm2.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = dict(float32=torch.float32, float16=torch.float16,
              bfloat16=torch.bfloat16)[a.dtype] if a.dtype else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)

    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wc = W - W.mean(0, keepdim=True)     # the mean direction is the sink, not a concept
    Wn = F.normalize(Wc, dim=1)
    nrm = Wc.norm(dim=1)
    order_n = torch.argsort(nrm)
    rank_n = torch.empty_like(order_n); rank_n[order_n] = torch.arange(len(order_n))

    # ---- second model, embeddings only, for the strict interpretability check ----
    XW = None
    if a.other:
        try:
            from transformers import AutoModel
            otok = AutoTokenizer.from_pretrained(a.other)
            oemb = AutoModel.from_pretrained(a.other, dtype=torch.float32).get_input_embeddings()
            OE = oemb.weight.detach().float()
            OE = F.normalize(OE - OE.mean(0, keepdim=True), dim=1)
            # map this model's tokens onto the other model's, by string
            xm_id = torch.full((W.shape[0],), -1, dtype=torch.long)
            for i in range(W.shape[0]):
                s = tok.convert_ids_to_tokens(i)
                if s is None: continue
                j = otok.convert_tokens_to_ids(s.replace("Ġ", "Ġ"))
                if j is None or j == otok.unk_token_id: 
                    j = otok.convert_tokens_to_ids(s)
                if j is not None and j != otok.unk_token_id and j < OE.shape[0]:
                    xm_id[i] = j
            XW = (OE, xm_id)
            print(f"cross-model: {int((xm_id>=0).sum())}/{W.shape[0]} tokens shared "
                  f"with {a.other}")
            del oemb
        except Exception as e:
            print(f"cross-model check unavailable: {e}")

    g = torch.Generator().manual_seed(0)

    def pair_cos(V):
        C = V @ V.T; n = V.shape[0]
        return float((C.sum() - n) / (n * (n - 1)))

    def zscore(ids, E, idmap=None):
        """how much tighter is this token set than tokens of the same rarity?"""
        if idmap is not None:
            keep = idmap[ids]; ids2 = keep[keep >= 0]
            if len(ids2) < 8: return float("nan")
            real = pair_cos(E[ids2]); pool = torch.arange(E.shape[0])
            null = [pair_cos(E[pool[torch.randint(0, len(pool), (len(ids2),), generator=g)]])
                    for _ in range(a.nnull)]
        else:
            real = pair_cos(E[ids]); rs = rank_n[ids]
            null = []
            for _ in range(a.nnull):
                pick = [int(order_n[torch.randint(max(0, int(r) - 500),
                                                  min(len(order_n), int(r) + 500),
                                                  (1,), generator=g)]) for r in rs]
                null.append(pair_cos(E[torch.tensor(pick)]))
        return (real - st.mean(null)) / (st.pstdev(null) + 1e-9)

    # ---- baselines ----
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

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js = blob["J"]; layers = sorted(Js)
    if a.layers: layers = [l for l in layers if l in a.layers]

    # residual scale per layer so alpha means the same thing at every depth
    hs, hooks = {}, []
    def mk(l):
        def f(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            hs.setdefault(l, []).append(t.detach().float().norm(dim=-1).median().cpu())
        return f
    for l in layers: hooks.append(blocks_of(model)[l].register_forward_hook(mk(l)))
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks: h.remove()
    scale = {l: float(torch.stack(hs[l]).mean()) for l in layers}

    rows = []
    for l in layers:
        J = Js[l].float()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        alpha = a.alpha * scale[l]
        for i in range(min(a.ndirs, U.shape[1])):
            u, v = U[:, i], Vh[i]
            r = Wc @ u                                  # what it writes, read out
            top = torch.topk(r, TOPK).indices
            coh_z = zscore(top, Wn)
            xm_z = zscore(top, XW[0], XW[1]) if XW else float("nan")

            cen = F.normalize(Wn[top].mean(0), dim=0)
            sims = Wn @ cen
            sims[torch.topk(r, EXCL).indices] = -2.0    # held out from the readout itself
            near = torch.topk(sims, NEAR).indices

            used = set(int(t) for t in near) | set(int(t) for t in top)
            ctrl = []
            for w in lp_rank[near].tolist():            # matched on how likely they already were
                for off in range(500):
                    for c in (int(lp_order[min(w + off, len(lp_order) - 1)]),
                              int(lp_order[max(w - off, 0)])):
                        if c not in used and float(sims[c]) < 0.15:
                            ctrl.append(c); used.add(c); break
                    else: continue
                    break
            ctrl = torch.tensor(ctrl[:NEAR])

            def probe(direction):
                with Add(model, l, direction, alpha):
                    lp, loss = final_lp(), text_loss()
                d = lp - base_lp
                return (float(d[top[0]]), float(d[near].mean() - d[ctrl].mean()),
                        loss - base_loss)

            own, lift, dloss = probe(v)
            rv = F.normalize(torch.randn(W.shape[1], generator=g), dim=0)
            r_own, r_lift, r_dloss = probe(rv)

            rows.append(dict(layer=l, dir=i, sigma=float(S[i]),
                             cos_uv=float(F.cosine_similarity(u, v, dim=0)),
                             coh_z=coh_z, xm_z=xm_z,
                             own=own, lift=lift, dloss=dloss,
                             r_own=r_own, r_lift=r_lift, r_dloss=r_dloss,
                             top=[tok.decode([int(t)]) for t in top],
                             near=[tok.decode([int(t)]) for t in near]))
            if i < 4 or i % 8 == 0:
                print(f"  L{l:<2d} d{i:<3d} sig={float(S[i]):6.2f} cos_uv={rows[-1]['cos_uv']:.2f} "
                      f"coh_z={coh_z:+6.1f} xm_z={xm_z:+6.1f} lift={lift:+6.2f} "
                      f"(rand {r_lift:+5.2f}) | {' '.join(repr(x) for x in rows[-1]['top'][:5])}",
                      flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(model=a.model, other=a.other, alpha=a.alpha,
                                         rows=rows), indent=1))
    print(f"\nwrote {a.out}  ({len(rows)} directions)")


if __name__ == "__main__":
    main()
