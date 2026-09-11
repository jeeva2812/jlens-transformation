"""Step 2: record, for every prompt, (a) which 8 of 64 experts woke up at each
middle layer, and (b) the residual stream there. Free signal + expensive signal
side by side.

Four sets, built so the token-identity control is possible:
  G  needs gcd, never says "gcd"
  W  says "gcd", needs no arithmetic
  C  plain facts, neither
  H  open-ended / hedging
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"
FEWSHOT = ("The LCM of 4 and 6 is 12.\n"
           "The LCM of 10 and 15 is 30.\n"
           "The LCM of 8 and 12 is 24.\n")

def build(n=60, seed=0):
    rng = random.Random(seed)
    G, W, C, H, N = [], [], [], [], []
    seen = set()
    while len(G) < n:                      # a = g*m, b = g*n, gcd(m,n)=1
        g = rng.randint(2, 10); m = rng.randint(2, 12); k = rng.randint(2, 12)
        if math.gcd(m, k) != 1 or m == k: continue
        a, b = g * m, g * k
        if (a, b) in seen or a > 99 or b > 99: continue
        seen.add((a, b))
        G.append({"set": "G", "text": FEWSHOT + f"The LCM of {a} and {b} is",
                  "a": a, "b": b, "gcd": g, "lcm": a * b // g})
        # N: identical digits, identical shape, different operation. Separates
        # "gcd computation" from "two numbers are present".
        N.append({"set": "N", "text": FEWSHOT + f"The sum of {a} and {b} is",
                  "a": a, "b": b, "gcd": g, "sum": a + b})
    # EVERY prompt must end in the SAME token. Routing is computed at that
    # token and is strongly token-dependent, so unmatched endings make the whole
    # comparison trivial: the first version had G ending in " is" and W in " ",
    # and "the router can tell those apart" is not a finding.
    WORDS = ["In mathematics, the abbreviation gcd is",
             "The Python library function for gcd is",
             "Another name for the gcd is",
             "The person who first described the gcd algorithm is",
             "The topic taught just before gcd is",
             "The usual notation for gcd is",
             "The opposite operation to gcd is",
             "A synonym for gcd is"]
    for i in range(n):
        W.append({"set": "W", "text": FEWSHOT + WORDS[i % len(WORDS)]})
    FACTS = ["The capital of France is", "The largest planet is",
             "The colour of the sky is", "The currency of Japan is",
             "The tallest mountain is", "The longest river is",
             "The chemical symbol for gold is", "The author of Hamlet is"]
    for i in range(n):
        C.append({"set": "C", "text": FEWSHOT + FACTS[i % len(FACTS)]})
    OPEN = ["The most likely outcome is", "The best explanation is",
            "The meaning of this poem is", "The reason for all of this is",
            "Whether the plan succeeds is"]
    for i in range(n // 2):
        H.append({"set": "H", "text": FEWSHOT + OPEN[i % len(OPEN)]})
    return G + W + C + H + N

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(5, 14)))
    ap.add_argument("--out", type=Path, default=Path("out/moe_capture.pt"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    prompts = build(a.n)
    print(f"{len(prompts)} prompts: " +
          ", ".join(f"{s}={sum(1 for p in prompts if p['set']==s)}" for s in "GWCHN"), flush=True)
    K = m.config.num_experts_per_tok
    recs = []
    for i, p in enumerate(prompts):
        ids = tok(p["text"], return_tensors="pt")["input_ids"]
        with T.no_grad():
            o = m(input_ids=ids, output_hidden_states=True, output_router_logits=True)
        pos = ids.shape[1] - 1                       # final prefill token
        h = {l: o.hidden_states[l + 1][0, pos].float().clone() for l in a.layers}
        rl = o.router_logits                          # tuple per layer, (seq, E)
        experts = {}
        for l in a.layers:
            r = rl[l]
            r = r.view(-1, r.shape[-1])[pos].float()
            experts[l] = T.topk(r, K).indices.sort().values.tolist()
        recs.append({**{k: v for k, v in p.items() if k != "text"},
                     "text": p["text"][len(FEWSHOT):], "h": h, "experts": experts,
                     "ntok": ids.shape[1]})
        if (i + 1) % 30 == 0: print(f"  {i+1}/{len(prompts)}", flush=True)
    T.save({"layers": a.layers, "K": K, "E": m.config.num_experts, "recs": recs}, a.out)
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
