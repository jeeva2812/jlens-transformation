"""Do the weight-derived labels predict behaviour on held-out prompts?

Two controls built in:
  - prompts NEVER contain the concept's own defining tokens, so a hit cannot be
    the router matching the literal word (the trap that killed the LCM experts)
  - the router score is averaged over ALL content positions, not just the final
    token, because routing at the final token is strongly token-dependent (an
    earlier run of this project was entirely an artifact of that)
Null: permute the prompt-to-concept assignment, take the family-wise max.
"""
from __future__ import annotations
import json, statistics as st
import torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"
PROMPTS = {
 "medicine": ["The surgeon closed the wound after a long operation in the",
   "She was prescribed antibiotics because the infection had spread to her",
   "The nurse checked his temperature again and noted that the fever had",
   "Recovery from the injury took several months of careful",
   "The ward was full, so they moved the elderly man to a",
   "Blood pressure readings had been unusually high since the",
   "He complained of dizziness and nausea after taking the",
   "The scan revealed a small growth near the base of the"],
 "money": ["The company reported quarterly earnings well above what analysts had",
   "Interest rates rose sharply, which made borrowing far more",
   "She transferred the deposit into a savings account at the",
   "The contract stipulated a fee of five thousand pounds payable on",
   "Inflation eroded the value of their pension over the",
   "He filed for bankruptcy after the debts became impossible to",
   "The bank refused the mortgage application because his credit was",
   "Shareholders voted against the merger, citing concerns about the"],
 "politics": ["The chancellor faced a vote of no confidence after the",
   "Campaigners gathered outside the assembly to protest against the",
   "The bill passed its second reading despite opposition from the",
   "Turnout was low in the constituencies most affected by the",
   "The coalition collapsed when the junior partner withdrew its",
   "Diplomats met in Geneva to negotiate the terms of the",
   "The referendum result split the electorate almost exactly down the",
   "He resigned as leader following revelations about the"],
 "technology": ["The update broke backwards compatibility with older versions of the",
   "Engineers patched the vulnerability after it was disclosed by a",
   "The server crashed under load, so they scaled horizontally across several",
   "She wrote a script to automate the deployment pipeline for the",
   "Latency dropped considerably once they moved the cache closer to the",
   "The open-source project attracted contributors from across the",
   "Encryption keys are rotated automatically every ninety days on the",
   "The file was too large to fit in memory, so they streamed it from"],
 "sport": ["He struck twice in the second half to seal victory for the",
   "The manager substituted the striker after an hour of the",
   "Training was cancelled because the pitch had frozen overnight in the",
   "She broke the world record by nearly two seconds at the",
   "The referee awarded a penalty following a challenge inside the",
   "Injuries kept him out of the squad for most of the",
   "The final went to extra time and then to penalties at the",
   "Fans travelled across the country to watch the away leg of the"],
 "people": ["A great throng had gathered in the square long before the",
   "The village had grown from a few families into a thriving",
   "Immigration reshaped the demographics of the northern towns over the",
   "Neighbours rallied round to help after the fire destroyed the",
   "The census recorded a sharp decline in the rural",
   "Generations of the same clan had farmed that land near the",
   "Volunteers organised meals for those left homeless by the",
   "Public opinion shifted markedly in the years following the"],
}

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    for p in m.parameters(): p.requires_grad_(False)
    D = json.load(open("out/router_dict.json"))
    E = D["E"]
    hit = {h["concept"]: (h["layer"], h["expert"]) for h in D["hits"]}
    cons = [c for c in PROMPTS if c in hit]
    print("labelled experts under test: " +
          ", ".join(f"{c} L{hit[c][0]}e{hit[c][1]}" for c in cons) + "\n")
    # leakage guard: no prompt may contain any of its concept's defining tokens
    from jlens.router_dict import CONCEPTS
    for c in cons:
        bad = [w for w in CONCEPTS[c].split()
               for p in PROMPTS[c] if w.lower() in p.lower()]
        if bad: print(f"  WARNING leakage in {c}: {set(bad)}")
    sc = {c: {} for c in cons}
    for c in cons:
        for p in PROMPTS[c]:
            ids = tok(p, return_tensors="pt")["input_ids"]
            with T.no_grad():
                o = m(input_ids=ids, output_router_logits=True)
            for c2 in cons:
                l, e = hit[c2]
                r = o.router_logits[l].view(-1, E).float()
                pr = T.softmax(r, -1)[4:, e].mean() if r.shape[0] > 5 else T.softmax(r, -1)[:, e].mean()
                sc[c].setdefault(c2, []).append(float(pr))
    print(f"mean router probability (averaged over content positions). "
          f"uniform = {1/E:.4f}\n")
    print(f"{'expert for':>12} " + " ".join(f"{c[:9]:>10}" for c in cons))
    diag = []
    for c2 in cons:
        row = [st.mean(sc[c][c2]) for c in cons]
        own = row[cons.index(c2)]; oth = st.mean([v for i, v in enumerate(row) if cons[i] != c2])
        diag.append((c2, own, oth, own / oth if oth else 0))
        print(f"{c2:>12} " + " ".join(f"{v:>10.4f}" for v in row)
              + f"   own/other = {own/oth:.2f}x" if oth else "")
    print(f"\n{'concept':>12} {'own':>9} {'others':>9} {'ratio':>7}")
    for c2, own, oth, r in diag:
        print(f"{c2:>12} {own:>9.4f} {oth:>9.4f} {r:>6.2f}x"
              + ("   <-- label predicts behaviour" if r > 1.5 else ""))
    n_ok = sum(1 for _, _, _, r in diag if r > 1.5)
    print(f"\n{n_ok}/{len(diag)} labelled experts prefer their own concept by >1.5x")
    import itertools, random
    rng = random.Random(0)
    flat = {c2: [(c, v) for c in cons for v in sc[c][c2]] for c2 in cons}
    nulls = []
    for _ in range(2000):
        best = 0.0
        for c2 in cons:
            vals = [v for _, v in flat[c2]]
            rng.shuffle(vals)
            k = len(sc[cons[0]][c2])
            own = st.mean(vals[:k]); oth = st.mean(vals[k:])
            best = max(best, own / oth if oth else 0)
        nulls.append(best)
    nulls.sort()
    thr = nulls[int(.95 * len(nulls))]
    print(f"permutation null (2000 draws, family-wise max over {len(cons)} concepts):")
    print(f"  95th pct of the best own/other ratio = {thr:.2f}")
    beat = sum(1 for _, _, _, r in diag if r > thr)
    print(f"  {beat}/{len(diag)} experts exceed it")

if __name__ == "__main__":
    main()
