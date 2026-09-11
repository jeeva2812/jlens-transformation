"""Choose the concept first, then ask J for the vector that produces it.

Everything so far took J's singular directions and asked what they mean. About
half of them meant something readable, and of those about half actually moved
the concept when pushed. That is the wrong way round: you are stuck with
whatever concepts the spectrum happens to contain.

The inverse: name the concept, build the target-space direction w that means it,
and solve for the layer-l vector x whose transport lands on w.

    steering at layer l by x produces J x at the target

so we want x with J x pointing along w. Three answers, and they differ a lot
because J is ill-conditioned (effective rank ~170-240 of 576):

  transpose  x = J^T w                      maximises <w, Jx> at fixed |x|.
                                            Robust. Also drags along everything
                                            else J happens to write.
  pinv       x = J^+ w                      produces w and as little else as
                                            possible -- but divides by the tiny
                                            singular values, so it can explode.
  ridge      x = (J^T J + lam I)^-1 J^T w   the tunable middle.

THE MEASUREMENT IS THE POINT. Today's lesson was that checking a direction on
the tokens you built it from cannot fail. So every concept's word list is split
in half: w is built from the TRAIN half only, and the effect is scored on the
TEST half, which never touched the construction. Control tokens are matched on
how likely they already were, and a random direction of identical norm goes
through the identical pipeline.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CONCEPTS = {
 "food": "bread cheese soup sauce meal dinner recipe kitchen cooking baking "
         "pasta rice meat vegetables salad butter flour oven chef restaurant",
 "medicine": "patient doctor hospital surgery disease treatment symptoms clinical "
             "diagnosis nurse therapy infection cancer blood tumour dose surgeon "
             "medical illness recovery",
 "programming": "function variable compiler debug syntax array loop server "
                "database code python module import class method parser thread "
                "memory library runtime",
 "law": "court judge lawyer trial evidence statute contract verdict appeal "
        "defendant plaintiff testimony jury legal ruling justice attorney "
        "prosecution liability clause",
 "sports": "match player team coach score goal tournament league stadium "
           "athlete championship referee striker season training race "
           "defence victory pitch olympic",
 "finance": "market investor revenue profit shares bond interest capital "
            "portfolio dividend inflation trading equity assets debt loan "
            "banking currency earnings valuation",
 "female": "she her herself woman women girl mother daughter sister aunt "
           "lady female queen actress niece madam hers girlfriend widow bride",
}
PROMPTS = [
    "The report was finished on Tuesday and", "After the long walk home,",
    "In the middle of the afternoon", "The old building on the corner",
    "When the meeting finally ended,", "Nobody expected the answer to be",
    "The train pulled into the station and", "On the table there was a",
    "They spent the whole weekend talking about", "What he really wanted was some",
    "The nurse put down the clipboard and then", "My sister called last night because",
]
COH = "The committee met on Tuesday to discuss the budget revisions."


class Add:
    def __init__(self, model, layer, d, alpha):
        self.d = F.normalize(d.float(), dim=0) * alpha
        self.h = model.model.layers[layer].register_forward_hook(self._f)
    def _f(self, m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        t2 = t + self.d.to(t.device, t.dtype)
        return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 26])
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--out", type=Path, default=Path("out/rare/pullback.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wc = W - W.mean(0, keepdim=True)
    Wn = F.normalize(Wc, dim=1)

    def single_ids(words):
        out = []
        for w in words:
            for cand in (" " + w, w, " " + w.capitalize()):
                e = tok.encode(cand, add_special_tokens=False)
                if len(e) == 1: out.append(e[0]); break
        return sorted(set(out))

    def lps():
        o = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                r = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            o.append(torch.log_softmax(r.logits[0, -1].float(), -1).cpu())
        return torch.stack(o).mean(0)
    coh_ids = tok(COH, return_tensors="pt")["input_ids"].to(dev)
    def loss():
        with torch.no_grad():
            o = model(input_ids=coh_ids, attention_mask=torch.ones_like(coh_ids), use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), coh_ids[0, 1:]))
    base, base_loss = lps(), loss()
    order = torch.argsort(base, descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order))

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(0)
    rows = []
    for l in a.layers:
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        # residual scale so alpha means the same at every depth
        bag = []
        h = model.model.layers[l].register_forward_hook(
            lambda m, i, o: bag.append(((o if torch.is_tensor(o) else o[0])[0, 1:])
                                       .detach().float().cpu().norm(dim=-1).median()) and None)
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
        h.remove()
        alpha = a.alpha * float(torch.stack(bag).mean())

        for cname, words in CONCEPTS.items():
            ids = single_ids(words.split())
            if len(ids) < 12: continue
            tr, te = ids[0::2], ids[1::2]          # build on one half, score the other
            w = F.normalize(Wn[torch.tensor(tr)].mean(0), dim=0)

            # control tokens: matched on how likely they already were, and not
            # close to the concept
            sims = Wn @ w
            used = set(ids)
            ctrl = []
            for r0 in rank[torch.tensor(te)].tolist():
                for off in range(800):
                    for c in (int(order[min(r0 + off, len(order) - 1)]),
                              int(order[max(r0 - off, 0)])):
                        if c not in used and float(sims[c]) < 0.10:
                            ctrl.append(c); used.add(c); break
                    else: continue
                    break
            ctrl = torch.tensor(ctrl[:len(te)])
            te_t = torch.tensor(te)

            def score(x):
                with Add(model, l, x, alpha):
                    lp, dl = lps(), loss() - base_loss
                d = lp - base
                return float(d[te_t].mean() - d[ctrl].mean()), dl

            # THE control. w already has the residual's dimension, so you can just
            # add it at layer l and never touch J. If that works as well, J^T is
            # decoration and this whole result is ordinary activation steering.
            cands = {"w itself (no J)   ": w.clone(),
                     "transpose  J^T w": J.T @ w,
                     "pinv       J^+ w": torch.linalg.pinv(J) @ w}
            for lam in a.lams:
                cands[f"ridge lam={lam:<6g}"] = torch.linalg.solve(
                    J.T @ J + lam * torch.eye(J.shape[1]), J.T @ w)
            # today's approach, for comparison: the single SVD direction whose
            # readout is closest to this concept
            k = int(torch.argmax(torch.stack([F.cosine_similarity(U[:, i], w, dim=0).abs()
                                              for i in range(64)])))
            sgn = 1.0 if float(F.cosine_similarity(U[:, k], w, dim=0)) > 0 else -1.0
            cands[f"best SVD dir (d{k})"] = Vh[k] * sgn
            cands["random"] = F.normalize(torch.randn(J.shape[1], generator=g), dim=0)

            for nm, x in cands.items():
                lift, dl = score(x)
                rows.append(dict(layer=l, concept=cname, method=nm.split()[0],
                                 label=nm, lift=lift, dloss=dl))
        print(f"layer {l} done", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))

    meths = list(dict.fromkeys(r["label"] for r in rows))
    print(f"\nHeld-out concept words moved, minus rank-matched controls\n")
    print(f"{'method':22s} " + "".join(f"{c[:9]:>10s}" for c in CONCEPTS) +
          f"{'mean':>8s}{'works':>8s}")
    for m in meths:
        g2 = [r for r in rows if r["label"] == m]
        cells = []
        for c in CONCEPTS:
            v = [r["lift"] for r in g2 if r["concept"] == c]
            cells.append(f"{st.mean(v):10.2f}" if v else f"{'-':>10s}")
        rnd = {(r["layer"], r["concept"]): r["lift"] for r in rows if r["label"] == "random"}
        w = sum(1 for r in g2 if r["lift"] > max(rnd.get((r["layer"], r["concept"]), 0), 0.2))
        print(f"{m:22s} " + "".join(cells) +
              f"{st.mean(r['lift'] for r in g2):8.2f}{w:5d}/{len(g2):<4d}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
