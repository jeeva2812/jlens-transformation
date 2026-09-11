"""Positive-result attempt: label experts by unembedding their ROUTER DIRECTION,
then test the label with the CONTINUOUS router score rather than top-8 selection.

Why this should work where routing failed: top-8 is a hard threshold, so a small
phrasing change flips membership. The score underneath is smooth. If an expert's
router row points at mathematics tokens, its score should be high on maths
prompts regardless of wording -- even when it does not crack the top 8.
"""
from __future__ import annotations
import math, random, statistics as st, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"
FEWSHOT = ("The LCM of 4 and 6 is 12.\nThe LCM of 10 and 15 is 30.\n"
           "The LCM of 8 and 12 is 24.\n")
CATS = {
 "math": [" mathematics"," algebra"," equation"," theorem"," arithmetic"," integer",
          " divisor"," multiple"," numeric"," calculation"," geometry"," fraction"],
 "people": [" people"," person"," society"," individuals"," community"," citizens"],
 "place": [" country"," national"," European"," international"," region"," city"],
 "quantity": [" hundred"," percent"," thousand"," million"," dozen"," amount"],
}

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.float32).eval()
    for p in m.parameters(): p.requires_grad_(False)
    W_U = m.get_output_embeddings().weight.detach().float()
    blk = m.model.layers; L = len(blk); E = m.config.num_experts
    ids_of = lambda ws: [tok.encode(w, add_special_tokens=False)[0] for w in ws]
    CIDS = {c: T.tensor(ids_of(ws)) for c, ws in CATS.items()}

    # ---- 1. score every (layer, expert) router direction against each category
    print("Finding the expert whose router direction most points at each category")
    print("(score = mean unembed logit on category tokens, z-scored over vocab)\n")
    best = {}
    for c, cid in CIDS.items():
        rows = []
        for l in range(L):
            R = blk[l].mlp.gate.weight.detach().float()
            Z = (R / R.norm(dim=1, keepdim=True)) @ W_U.T          # (E, V)
            z = (Z - Z.mean(1, keepdim=True)) / Z.std(1, keepdim=True)
            s = z[:, cid].mean(1)
            for e in range(E): rows.append((float(s[e]), l, e))
        rows.sort(reverse=True)
        best[c] = rows[:3]
        for sc, l, e in rows[:3]:
            R = blk[l].mlp.gate.weight.detach().float()[e]
            t = [repr(tok.decode([i]))[1:-1] for i in ((R/R.norm())@W_U.T).topk(6).indices.tolist()]
            print(f"  {c:>9}  L{l:>2} e{e:<3} z={sc:+.2f}   " + " ".join(f"{x:<11}" for x in t))
    print()

    # ---- 2. does the CONTINUOUS score separate the right prompts, held out?
    rng = random.Random(3); pairs = []
    while len(pairs) < 20:
        g = rng.randint(2, 11); a = rng.randint(2, 13); b = rng.randint(2, 13)
        if math.gcd(a, b) != 1 or a == b or g*a > 99 or g*b > 99: continue
        pairs.append((g*a, g*b))
    SETS = {
      "math (seen wording)":   [f"The LCM of {a} and {b} is" for a, b in pairs],
      "math (NEW wording)":    [f"The least common multiple of {a} and {b} is" for a, b in pairs],
      "math (sum, NEW)":       [f"The total of {a} and {b} is" for a, b in pairs],
      "people":                ["The people of this town are", "Most individuals in society are",
                                "The citizens of the country are", "A community of persons is"]*5,
      "facts":                 ["The capital of France is", "The author of Hamlet is",
                                "The colour of the sky is", "The currency of Japan is"]*5,
    }
    def scores(text):
        ids = tok(text, return_tensors="pt")["input_ids"]
        with T.no_grad(): o = m(input_ids=ids, output_router_logits=True)
        pos = ids.shape[1]-1
        return {l: T.softmax(o.router_logits[l].view(-1, E)[pos].float(), -1) for l in range(L)}
    cache = {k: [scores(FEWSHOT+t) for t in v] for k, v in SETS.items()}
    print("CONTINUOUS router probability for the labelled expert (not top-8 membership)\n")
    hdr = f"{'expert':>18} " + " ".join(f"{k[:17]:>18}" for k in SETS)
    print(hdr); print("-"*len(hdr))
    for c in CATS:
        sc, l, e = best[c][0]
        row = []
        for k in SETS:
            row.append(st.mean(float(s[l][e]) for s in cache[k]))
        print(f"{c+' L'+str(l)+' e'+str(e):>18} " + " ".join(f"{v:>18.4f}" for v in row))
    print(f"\n(uniform would be {1/E:.4f}. Test: does the MATH expert stay high on the")
    print(" NEW wording, where top-8 selection collapsed from 100% to 8%?)")

if __name__ == "__main__":
    main()
