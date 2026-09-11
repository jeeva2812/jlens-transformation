"""What does the lens read out for THIS prompt, and which directions carried it?

Everything so far treated a direction as an object on its own: read its tokens,
push it, see what happens. That throws away the prompt. But J is a linear map, so
for an actual activation h the output decomposes exactly:

    J h = sum_i  sigma_i <v_i, h> u_i

one term per singular direction. <v_i, h> is how much this prompt excites
direction i; sigma_i <v_i, h> is how much output it contributes. So for any
prompt we can say which directions did the work and what each of them reads as.

Two versions side by side:

  J            directions ranked as the SVD of J ranks them -- every input
               direction weighted equally, including ones the model never visits
  J Sigma^1/2  the same thing after rescaling the input space by the covariance
               of real activations, so one unit means one standard deviation of
               what the model actually does

Sigma needs many more tokens than dimensions or it is rank-deficient and silently
throws most of the space away, so it is estimated from ~12k pile tokens, not from
the handful of prompts being read out.

Three readouts per prompt, all through the same unembedding:
  lens     W_U (J h)          what the transport says this prompt leads to
  model    the real logits    what the model actually predicts
  inside / outside            h split by the top-k subspace of J, each pushed
                              through J separately -- what the top of the lens
                              sees, and what it misses
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

PROMPTS = [
    "The report was finished on Tuesday and",
    "After the long walk home,",
    "The nurse put down the clipboard and then",
    "The doctor explained that the patient",
    "For dinner they decided to make",
    "The children ran outside to play with",
    "On the table there was a",
    "Water freezes at a temperature of",
    "The shop on the corner sells",
    "My sister called last night because",
    "import os\nimport sys\n\ndef main():\n    parser =",
    "for i in range(len(arr)):\n    if arr[i] >",
    "SELECT user_id, COUNT(*) FROM orders WHERE",
    "The error message said that the",
]


def top_tokens(vec, W_U, tok, k=10):
    return [tok.decode([int(t)]) for t in torch.topk(W_U @ vec, k).indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 20])
    ap.add_argument("--k", type=int, default=32, help="size of the 'top of J' subspace")
    ap.add_argument("--ndir", type=int, default=4, help="carrying directions to show")
    ap.add_argument("--ntok", type=int, default=12000, help="tokens for estimating Sigma")
    ap.add_argument("--npile", type=int, default=0,
                    help="add this many pile snippets as extra prompts. 14 hand-written "
                         "prompts is far too few to tell a 15%% hit rate from an 8%% one, "
                         "and pile snippets give topical variety without me choosing it")
    ap.add_argument("--out", type=Path, default=Path("out/rare/prompt_readout.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blk = model.model.layers
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    Wc = W_U - W_U.mean(0, keepdim=True)

    # ---- Sigma per layer, from pile text, dropping the attention sink ----
    store = {l: [] for l in a.layers}
    hooks = []
    def mk(l):
        def f(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            store[l].append(t[0].detach().float().cpu())
        return f
    for l in a.layers: hooks.append(blk[l].register_forward_hook(mk(l)))
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    n = 0
    for i in range(400):
        ids = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                  max_length=128)["input_ids"].to(dev)
        if ids.shape[1] < 8: continue
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
        n += ids.shape[1]
        if n > a.ntok: break
    for h in hooks: h.remove()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    dec = {}
    for l in a.layers:
        H = torch.cat([x[1:] for x in store[l]], 0)          # position 0 is the sink
        A = H - H.mean(0, keepdim=True)
        S = (A.T @ A) / (A.shape[0] - 1)
        ev, Wv = torch.linalg.eigh(S)
        ev = ev.clamp(min=0)
        Sh = Wv @ torch.diag(ev.sqrt()) @ Wv.T
        J = blob["J"][l].float()
        dec[l] = dict(J=J, Sh=Sh,
                      svd=torch.linalg.svd(J, full_matrices=False),
                      svdw=torch.linalg.svd(J @ Sh, full_matrices=False),
                      ntok=H.shape[0], erank=float(ev.sum()**2 / (ev**2).sum()))
        print(f"L{l}: Sigma from {H.shape[0]} tokens, rank "
              f"{int((ev > ev.max()*1e-8).sum())}/{len(ev)}, effective rank "
              f"{dec[l]['erank']:.1f}", flush=True)

    prompts = list(PROMPTS)
    if a.npile:
        seen = set()
        for i in range(2000):
            t = ds[i]["text"].strip().replace("\n", " ")
            t = " ".join(t.split()[:24])
            if len(t) < 80 or t[:40] in seen: continue
            seen.add(t[:40]); prompts.append(t)
            if len(prompts) >= len(PROMPTS) + a.npile: break
    print(f"{len(prompts)} prompts", flush=True)

    store.clear()
    rows = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        cap = {}
        hs = [blk[l].register_forward_hook(
                (lambda ll: (lambda m, i, o: cap.__setitem__(
                    ll, (o if torch.is_tensor(o) else o[0])[0, -1].detach().float().cpu())))(l))
              for l in a.layers]
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        for h in hs: h.remove()
        real = [tok.decode([int(t)]) for t in torch.topk(o.logits[0, -1].float(), 10).indices]

        for l in a.layers:
            d = dec[l]; h = cap[l]
            U, S, Vh = d["svd"]; Uw, Sw, Vw = d["svdw"]
            Jh = d["J"] @ h
            inside = Vh[:a.k].T @ (Vh[:a.k] @ h)
            outside = h - inside

            def carriers(Umat, Smat, Vmat, hh, weighted):
                # how much this prompt excites each direction, times how hard J
                # amplifies it -- the terms of  J h = sum_i sigma_i <v_i,h> u_i
                c = Vmat @ (torch.linalg.solve(d["Sh"], hh) if weighted else hh)
                contrib = Smat * c
                idx = torch.topk(contrib.abs(), a.ndir).indices
                return [dict(i=int(j), coef=float(contrib[j]),
                             share=float(contrib[j].abs() / contrib.abs().sum()),
                             tokens=top_tokens(Umat[:, j] * (1 if contrib[j] > 0 else -1),
                                               Wc, tok, 8))
                        for j in idx]

            rows.append(dict(
                prompt=p, layer=l,
                model_says=real,
                lens=top_tokens(Jh, Wc, tok, 10),
                inside=top_tokens(d["J"] @ inside, Wc, tok, 10),
                outside=top_tokens(d["J"] @ outside, Wc, tok, 10),
                inside_share=float(inside.norm() / h.norm()),
                carriers_J=carriers(U, S, Vh, h, False),
                carriers_W=carriers(Uw, Sw, Vw, h, True)))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))

    for r in rows[:len(PROMPTS) * len(a.layers)]:
        if r["layer"] != a.layers[0]: continue
        print(f"\n{'='*76}\n{r['prompt']!r}   (layer {r['layer']})")
        print(f"  model actually predicts : {' '.join(repr(x) for x in r['model_says'][:8])}")
        print(f"  lens reads out          : {' '.join(repr(x) for x in r['lens'][:8])}")
        print(f"  from inside the top-{a.k}  : {' '.join(repr(x) for x in r['inside'][:8])}"
              f"   ({r['inside_share']:.0%} of the residual)")
        print(f"  from outside it         : {' '.join(repr(x) for x in r['outside'][:8])}")
        print(f"  carried by, ranked by J:")
        for c in r["carriers_J"]:
            print(f"     d{c['i']:<4d} {c['share']:5.1%} {' '.join(repr(x) for x in c['tokens'][:6])}")
        print(f"  carried by, after weighting by real activations:")
        for c in r["carriers_W"]:
            print(f"     d{c['i']:<4d} {c['share']:5.1%} {' '.join(repr(x) for x in c['tokens'][:6])}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
