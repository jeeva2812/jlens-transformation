"""How much of J's spectrum does the pullback actually need?

This closes the subspace question rather than abandoning it. Write the pullback
in J's own basis:

    J^T w = sum_i  sigma_i <u_i, w>  v_i

so the pullback is not an alternative to steering with subspaces -- it IS a
combination of J's singular directions, each weighted by how hard J drives it
(sigma_i) times how much its output side agrees with the concept (<u_i, w>).
Every earlier attempt used ONE direction and hoped it happened to mean the right
thing. This uses all of them and lets the geometry set the weights.

So the real question is how many terms you need. Truncate the sum at k and sweep:

  k = 1     the single best-aligned direction, weighted -- close to what the
            SVD experiments were doing
  k small   a genuine subspace
  k = full  the plain pullback

If a few dozen terms recover the full effect, subspace steering was right and
was only ever let down by picking directions by hand. If you need the whole
spectrum, the small-sigma directions carry real weight and subspace thinking is
the wrong frame.

Scored, as before, on HELD-OUT concept words: w is built from half of each
concept's word list and the effect measured on the other half.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.pullback_steer import CONCEPTS, PROMPTS, COH, Add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 576])
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("out/rare/pullback_rank.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wn = F.normalize(W - W.mean(0, keepdim=True), dim=1)

    def single_ids(words):
        out = []
        for w in words:
            for c in (" " + w, w, " " + w.capitalize()):
                e = tok.encode(c, add_special_tokens=False)
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
    coh = tok(COH, return_tensors="pt")["input_ids"].to(dev)
    def loss():
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), coh[0, 1:]))
    base, base_loss = lps(), loss()
    order = torch.argsort(base, descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order))

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    rows = []
    for l in a.layers:
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
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
            tr, te = ids[0::2], ids[1::2]
            w = F.normalize(Wn[torch.tensor(tr)].mean(0), dim=0)
            coefs = S * (U.T @ w)                      # sigma_i <u_i, w>
            sims = Wn @ w
            used, ctrl = set(ids), []
            for r0 in rank[torch.tensor(te)].tolist():
                for off in range(800):
                    for c in (int(order[min(r0 + off, len(order) - 1)]),
                              int(order[max(r0 - off, 0)])):
                        if c not in used and float(sims[c]) < 0.10:
                            ctrl.append(c); used.add(c); break
                    else: continue
                    break
            ctrl = torch.tensor(ctrl[:len(te)]); te_t = torch.tensor(te)

            for k in a.ks:
                if k > len(S): continue
                x = Vh[:k].T @ coefs[:k]               # truncated pullback
                if x.norm() < 1e-8: continue
                with Add(model, l, x, alpha):
                    lp, dl = lps(), loss() - base_loss
                d = lp - base
                rows.append(dict(layer=l, concept=cname, k=k,
                                 lift=float(d[te_t].mean() - d[ctrl].mean()),
                                 dloss=dl,
                                 mass=float(coefs[:k].pow(2).sum() / coefs.pow(2).sum())))
        print(f"layer {l} done", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))

    print(f"\n{'directions kept':>16s} {'lift':>8s} {'% of full':>10s} "
          f"{'share of the pullback':>22s} {'damage':>8s}")
    full = st.mean(r["lift"] for r in rows if r["k"] == max(a.ks))
    for k in a.ks:
        g = [r for r in rows if r["k"] == k]
        if not g: continue
        m = st.mean(r["lift"] for r in g)
        print(f"{k:16d} {m:8.2f} {m/full:9.0%} {st.mean(r['mass'] for r in g):22.3f} "
              f"{st.mean(r['dloss'] for r in g):+8.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
