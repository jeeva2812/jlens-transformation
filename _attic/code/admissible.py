"""Does a readable direction fail, or does the prompt just not admit its concept?

Half the readable directions did not steer under a fixed set of neutral prompts.
But those prompts were the same for every direction, so for any given direction
they may simply be places where its concept could not come next. "she" only fits
where the grammar wants a pronoun; "noodles" fits almost anywhere a noun does.
That would make the failures a fact about the prompts, not about the direction.

The test avoids writing prompts by hand per concept, which would just move the
choice somewhere I could bias it. Instead, one bank of 40 varied prompts, and for
each direction the bank is split by the model's OWN baseline: prompts where its
top tokens were already relatively likely (the concept could plausibly come next)
versus where they were not. Same direction, same push, same measurement -- only
the context differs.

  admits    top third of prompts by baseline probability of the concept
  refuses   bottom third

If steering works on the admitting prompts and not the refusing ones, then
"can we steer it" was never a property of the direction on its own.
"""
from __future__ import annotations
import argparse, glob, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.interp_score import Scorer
from jlens.leftover import blocks_of

BANK = [
    # narrative
    "The report was finished on Tuesday and", "After the long walk home,",
    "In the middle of the afternoon", "The old building on the corner",
    "When the meeting finally ended,", "She opened the letter and",
    "The train pulled into the station and", "Nobody expected the answer to be",
    "He picked up the phone and", "In the summer of 1994 the",
    # people, so a pronoun has somewhere to go
    "The nurse put down the clipboard and then", "My sister called last night because",
    "The teacher looked at the class and", "When the doctor came back in,",
    "His brother had already left, so", "The girl finished her homework and",
    # description, so a noun has somewhere to go
    "On the table there was a", "The shop on the corner sells",
    "Her favourite thing in the world is", "What he really wanted was some",
    "The box was full of", "They spent the whole weekend talking about",
    # topical openers
    "The doctor explained that the patient", "Prices rose sharply after the",
    "The results of the experiment showed", "The judge listened carefully while",
    "Water freezes at a temperature of", "The company announced today that",
    "Scientists have recently discovered that", "The recipe calls for a cup of",
    "The diagnosis came back and it was", "For dinner they decided to make",
    "The children ran outside to play with", "At the airport she checked in her",
    # technical
    "import os\nimport sys\n\ndef main():\n    parser =",
    "for i in range(len(arr)):\n    if arr[i] >",
    "The function returns a list of", "You can configure this by editing the",
    "SELECT user_id, COUNT(*) FROM orders WHERE", "The error message said that the",
]
COH_TEXT = "The committee met on Tuesday to discuss the budget revisions."
NEAR, EXCL = 12, 200


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
    ap.add_argument("--other", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--readable", type=float, default=12.0)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--nnull", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("out/rare/admissible.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W = model.get_output_embeddings().weight.detach().float().cpu()
    sc = Scorer(W, tok, other_model=a.other, nnull=a.nnull)
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)

    # baseline log-probs, one row per prompt (not averaged -- the split needs them apart)
    def lps(hook=None):
        out = []
        for p in BANK:
            ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.log_softmax(o.logits[0, -1].float(), -1).cpu())
        return torch.stack(out)                       # (P, V)

    base = lps()
    coh_ids = tok(COH_TEXT, return_tensors="pt")["input_ids"].to(dev)
    def loss():
        with torch.no_grad():
            o = model(input_ids=coh_ids, attention_mask=torch.ones_like(coh_ids), use_cache=False)
        return float(F.cross_entropy(o.logits[0, :-1].float(), coh_ids[0, 1:]))
    base_loss = loss()
    mean_base = base.mean(0)
    lp_order = torch.argsort(mean_base, descending=True)
    lp_rank = torch.empty_like(lp_order); lp_rank[lp_order] = torch.arange(len(lp_order))

    cand = []
    for f in sorted(glob.glob("out/rare/steer_*.json")):
        for r in json.load(open(f))["rows"]:
            if r["arm"] == "top" and r["xm_z"] > a.readable and Scorer.is_content(r["top"]):
                cand.append(r)
    seen, uniq = set(), []
    for r in cand:
        k = (r["layer"], r["dir"])
        if k in seen: continue
        seen.add(k); uniq.append(r)
    print(f"{len(uniq)} readable word-directions to re-test on {len(BANK)} prompts\n")

    cache, g, rows = {}, torch.Generator().manual_seed(0), []
    for rec in uniq:
        l, i = rec["layer"], rec["dir"]
        if l not in cache:
            J = blob["J"][l].float()
            U, S, Vh = torch.linalg.svd(J, full_matrices=False)
            hs, st_ = [], []
            def grab(m, inp, out):          # must return None, or it REPLACES the
                t = out if torch.is_tensor(out) else out[0]   # block's output
                st_.append(t[0].detach().float().cpu())
            h = blocks_of(model)[l].register_forward_hook(grab)
            for p in BANK[:8]:
                ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
                with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
            h.remove()
            hs = [x[1:].norm(dim=-1).median() for x in st_]
            cache[l] = (U, S, Vh, float(torch.stack(hs).mean()))
        U, S, Vh, scale = cache[l]
        u, v = U[:, i], Vh[i]

        rr = sc.Wc @ u
        top = torch.topk(rr, 12).indices
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

        # split the bank by whether this concept was already plausible there
        fit = base[:, near].mean(1) - base[:, ctrl].mean(1)     # per prompt
        o = torch.argsort(fit, descending=True)
        t = len(BANK) // 3
        admits, refuses = o[:t].tolist(), o[-t:].tolist()

        def run(direction):
            with Add(model, l, direction, a.alpha * scale):
                lp, dl = lps(), loss() - base_loss
            d = lp - base
            per = d[:, near].mean(1) - d[:, ctrl].mean(1)       # lift per prompt
            return per, dl

        per, dl = run(v)
        rv = F.normalize(torch.randn(W.shape[1], generator=g), dim=0)
        rper, rdl = run(rv)
        row = dict(layer=l, dir=i, xm_z=rec["xm_z"], top=rec["top"],
                   admits=float(per[admits].mean()), refuses=float(per[refuses].mean()),
                   all=float(per.mean()),
                   r_admits=float(rper[admits].mean()), r_all=float(rper.mean()),
                   dloss=dl,
                   best_prompt=BANK[int(o[0])])
        rows.append(row)
        print(f"  L{l:<3d} d{i:<3d} admits={row['admits']:+6.2f} "
              f"refuses={row['refuses']:+6.2f} all={row['all']:+6.2f} "
              f"(rand {row['r_admits']:+5.2f}) | {' '.join(repr(x) for x in rec['top'][:5])}",
              flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))

    def rate(key, rkey="r_admits"):
        return sum(1 for r in rows if r[key] > max(r[rkey], 0.2)) / len(rows)
    print(f"\n{'condition':34s} {'mean lift':>10s} {'works':>10s}")
    print(f"{'prompts that admit the concept':34s} "
          f"{st.mean(r['admits'] for r in rows):10.2f} {rate('admits'):10.0%}")
    print(f"{'prompts that refuse it':34s} "
          f"{st.mean(r['refuses'] for r in rows):10.2f} {rate('refuses'):10.0%}")
    print(f"{'the whole bank, undifferentiated':34s} "
          f"{st.mean(r['all'] for r in rows):10.2f} {rate('all', 'r_all'):10.0%}")
    print(f"{'random direction, admitting prompts':34s} "
          f"{st.mean(r['r_admits'] for r in rows):10.2f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
