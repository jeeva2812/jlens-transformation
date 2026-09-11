"""Do J's subspaces represent anything abstract, and do different ones differ?

Forget next-token prediction. The question is whether a J direction responds to
a KIND of input -- a topic, a tense, a register -- and whether different
directions respond to different kinds.

The measurement uses the coefficients, not the readout tokens:

    c_i = <v_i, h>     how much this prompt excites direction i

One control decides whether any of this is about J. Any complete orthonormal
basis spans the same space, so decoding from ALL coordinates gives the identical
answer in every basis -- that number would be meaningless. What is not
basis-independent is whether the information is CONCENTRATED: can you decode the
category from the top 8, or 16, of J's directions, better than from 8 or 16
random orthogonal directions? That comparison is the experiment. Two more
baselines: the top principal directions of the activations themselves (the
obvious competitor), and a shuffled-label null.

Then, per direction: which single direction best separates cooking prompts from
medicine ones, and does that direction's own readout look like food? If the
direction that fires for cooking prompts also unembeds to food words, the two
halves of "meaning" agree, and that is about as direct as this gets.
"""
from __future__ import annotations
import argparse, itertools, json, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TOPICS = {
 "cooking": [
  "She chopped the onions finely and added them to the pan",
  "The recipe says to simmer the sauce for twenty minutes",
  "He kneaded the dough until it was smooth and elastic",
  "Preheat the oven and grease the baking tin thoroughly",
  "The soup needed more salt and a squeeze of lemon",
  "They roasted the vegetables with olive oil and rosemary",
  "Whisk the eggs together with a little cream and pepper",
  "The bread had risen overnight and was ready to bake",
  "She grated the cheese over the pasta before serving",
  "The stew had been cooking slowly since the morning",
  "Add the garlic to the hot oil and fry it briefly",
  "He tasted the sauce and decided it needed more herbs"],
 "medicine": [
  "The patient presented with a persistent cough and fever",
  "She was prescribed antibiotics for the bacterial infection",
  "The scan revealed a small lesion on the left lung",
  "His blood pressure had been elevated for several months",
  "The surgeon explained the risks of the procedure carefully",
  "Symptoms usually appear within two days of exposure",
  "The nurse checked her pulse and recorded the readings",
  "Treatment involves a course of chemotherapy and radiation",
  "The diagnosis was confirmed by a second blood test",
  "He had been experiencing chest pain for about a week",
  "The dosage should be reduced for patients with kidney damage",
  "Recovery from the operation took nearly three months"],
 "programming": [
  "The function returns a list of integers sorted ascending",
  "He refactored the loop into a separate helper method",
  "The compiler raised an error about an undefined variable",
  "She wrote unit tests covering the edge cases thoroughly",
  "The server responds with a JSON object containing the data",
  "Memory usage grew because the cache was never cleared",
  "The script parses the file and writes the output to disk",
  "They migrated the database schema in a single transaction",
  "The exception is caught and logged before being re-raised",
  "He added type annotations to make the interface clearer",
  "The algorithm runs in linear time with constant extra space",
  "She debugged the race condition by adding a mutex lock"],
 "law": [
  "The court ruled that the contract had been breached",
  "Counsel argued that the evidence was inadmissible at trial",
  "The statute requires notice to be given within thirty days",
  "She filed an appeal against the earlier judgment",
  "The defendant pleaded not guilty to all of the charges",
  "The tribunal found in favour of the claimant on liability",
  "Legislation passed last year amended the licensing rules",
  "The judge instructed the jury to disregard the remark",
  "Their lawyer advised them to settle out of court",
  "The clause limits liability to the value of the goods",
  "He was convicted of fraud and sentenced to four years",
  "The witness gave testimony about the events of that night"],
 "sports": [
  "He scored twice in the second half of the match",
  "The team trained hard through the whole pre-season",
  "She broke the national record by nearly two seconds",
  "The referee awarded a penalty after reviewing the replay",
  "They lost the final on penalties after extra time",
  "The striker was substituted in the seventieth minute",
  "He won the race despite a slow start off the blocks",
  "The tournament runs for two weeks every summer",
  "Her serve was too strong for her opponent to return",
  "The coach changed formation midway through the game",
  "He retired from professional play at the age of thirty five",
  "The crowd cheered as the runners entered the stadium"],
 "finance": [
  "The company reported a sharp fall in quarterly revenue",
  "Investors moved capital into bonds as yields rose",
  "The central bank raised interest rates by half a point",
  "She sold the shares before the market opened on Monday",
  "Inflation has eroded the real value of their savings",
  "The merger was valued at nearly four billion dollars",
  "Analysts downgraded the stock after weak guidance",
  "The fund charges an annual management fee of one percent",
  "Currency markets reacted badly to the policy announcement",
  "They refinanced the mortgage at a lower fixed rate",
  "The balance sheet shows a substantial increase in debt",
  "Profits were up despite rising costs across the sector"],
}
# form, not topic -- these cut across the topics above on purpose
AXES = {
 "question vs statement": (
  ["Why did the machine stop working last night",
   "What is the best way to reach the station from here",
   "How long does the treatment usually take to work",
   "Who signed the agreement on behalf of the company",
   "When will the results of the study be published",
   "Which of the two options is cheaper in the long run",
   "Where did they keep the records before the move",
   "Is there a reason the report was delayed again"],
  ["The machine stopped working last night without warning",
   "The station is a short walk from here down the hill",
   "The treatment usually takes about six weeks to work",
   "The agreement was signed on behalf of the company",
   "The results of the study will be published in March",
   "The cheaper of the two options costs less over time",
   "The records were kept in the basement before the move",
   "The report was delayed again for the usual reasons"]),
 "past vs present": (
  ["She walked to the office and unlocked the front door",
   "They finished the work early and went home",
   "He opened the letter and read it twice",
   "The meeting ended before anyone had asked a question",
   "We arrived late and missed the opening remarks",
   "The company hired forty people over the winter",
   "It rained heavily throughout the afternoon",
   "The train left the platform exactly on time"],
  ["She walks to the office and unlocks the front door",
   "They finish the work early and go home",
   "He opens the letter and reads it twice",
   "The meeting ends before anyone has asked a question",
   "We arrive late and miss the opening remarks",
   "The company hires forty people over the winter",
   "It rains heavily throughout the afternoon",
   "The train leaves the platform exactly on time"]),
}


def nearest_centroid_loo(X, y):
    """leave-one-out nearest centroid -- no fitting worth the name, so no room
    for the classifier to memorise"""
    X = F.normalize(X, dim=1)
    labs = sorted(set(y)); ok = 0
    for i in range(len(y)):
        cs = []
        for L in labs:
            idx = [j for j in range(len(y)) if y[j] == L and j != i]
            cs.append(F.normalize(X[idx].mean(0), dim=0))
        ok += labs[int(torch.stack(cs).mv(X[i]).argmax())] == y[i]
    return ok / len(y)


def auc(a, b):
    """probability a random item from a scores above one from b"""
    w = sum((x > yv) + 0.5 * (x == yv) for x in a for yv in b)
    return w / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 28])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--nrand", type=int, default=20, help="random bases to average over")
    ap.add_argument("--out", type=Path, default=Path("out/rare/subspace_meaning.json"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wc = W - W.mean(0, keepdim=True)
    blk = model.model.layers

    prompts, topic = [], []
    for t, ps in TOPICS.items():
        prompts += ps; topic += [t] * len(ps)
    axis_span = {}
    for nm, (A, B) in AXES.items():
        axis_span[nm] = (len(prompts), len(A), len(B))
        prompts += A + B

    store = {l: [] for l in a.layers}
    hooks = []
    def mk(l):
        def f(m, i, o):
            t = o if torch.is_tensor(o) else o[0]
            store[l].append(t[0, 1:].detach().float().cpu().mean(0))   # drop the sink
        return f
    for l in a.layers: hooks.append(blk[l].register_forward_hook(mk(l)))
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks: h.remove()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(0)
    nT = len(topic)
    out = {}
    for l in a.layers:
        H = torch.stack(store[l])
        Hc = H - H.mean(0, keepdim=True)
        J = blob["J"][l].float()
        Vj = torch.linalg.svd(J, full_matrices=False)[2]            # J's input directions
        Uj = torch.linalg.svd(J, full_matrices=False)[0]
        Vp = torch.linalg.svd(Hc, full_matrices=False)[2]           # activation PCs
        d = H.shape[1]

        print(f"\n{'='*70}\nlayer {l}: can you tell the topic from the coefficients?"
              f"  ({len(TOPICS)} topics, {nT} prompts, chance {1/len(TOPICS):.0%})")
        print(f"{'directions used':>16s} {'top of J':>10s} {'activation PCs':>15s} "
              f"{'random':>9s} {'shuffled':>9s}")
        res = {}
        for k in a.ks:
            if k > d: continue
            aj = nearest_centroid_loo(Hc[:nT] @ Vj[:k].T, topic)
            ap_ = nearest_centroid_loo(Hc[:nT] @ Vp[:k].T, topic)
            rs = []
            for _ in range(a.nrand):
                R = torch.linalg.qr(torch.randn(d, k, generator=g))[0]
                rs.append(nearest_centroid_loo(Hc[:nT] @ R, topic))
            sh = topic[:]; 
            perm = torch.randperm(nT, generator=g).tolist()
            shuf = [topic[i] for i in perm]
            sn = nearest_centroid_loo(Hc[:nT] @ Vj[:k].T, shuf)
            res[k] = dict(J=aj, pca=ap_, rand=st.mean(rs), rand_sd=st.pstdev(rs), shuf=sn)
            print(f"{k:16d} {aj:10.0%} {ap_:15.0%} {st.mean(rs):8.0%} "
                  f"{'+-'}{st.pstdev(rs):.0%} {sn:8.0%}")
        out[l] = dict(topic=res)

        # which single direction separates which pair of topics?
        C = Hc[:nT] @ Vj[:64].T
        print(f"\n  the single J direction that best tells one topic from another:")
        best = []
        for x, y in itertools.combinations(sorted(TOPICS), 2):
            ai = [i for i in range(nT) if topic[i] == x]
            bi = [i for i in range(nT) if topic[i] == y]
            sc = [(abs(auc(C[ai, j].tolist(), C[bi, j].tolist()) - .5) + .5, j)
                  for j in range(64)]
            s, j = max(sc)
            toks = [tok.decode([int(t)]) for t in torch.topk(Wc @ Uj[:, j], 6).indices]
            best.append(dict(a=x, b=y, dir=j, auc=s, tokens=toks))
            print(f"    {x:12s} vs {y:12s}  d{j:<3d} auc {s:.2f}  "
                  f"| {' '.join(repr(t) for t in toks)}")
        out[l]["pairs"] = best

        for nm, (off, na, nb) in axis_span.items():
            A = Hc[off:off+na] @ Vj[:64].T
            B = Hc[off+na:off+na+nb] @ Vj[:64].T
            sc = [(abs(auc(A[:, j].tolist(), B[:, j].tolist()) - .5) + .5, j) for j in range(64)]
            s, j = max(sc)
            n_sel = sum(1 for v, _ in sc if v > 0.85)
            toks = [tok.decode([int(t)]) for t in torch.topk(Wc @ Uj[:, j], 6).indices]
            print(f"  {nm:24s} best d{j:<3d} auc {s:.2f}, {n_sel} of 64 directions "
                  f"separate it at auc>0.85 | {' '.join(repr(t) for t in toks)}")
            out[l].setdefault("axes", {})[nm] = dict(dir=j, auc=s, n_sel=n_sel, tokens=toks)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
