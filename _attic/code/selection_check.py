"""Two corrections to the per-pair 'direction dNN means cooking' claims.

1. MAX-STATISTIC NULL. Those AUCs were the best of 64 directions. The right
   null is therefore not 0.5 -- it is the distribution of the BEST of 64 under
   shuffled labels. Same scan, same 64 directions, labels permuted.

2. HELD OUT. Pick the best direction on one half of the prompts, then score it
   on the other half. If the direction really responds to the category, it keeps
   working on prompts that were not used to select it.

The topic-classification table is not affected by either -- it was already
leave-one-out and had a matched random-direction baseline. Only the per-pair
claims were selected on.
"""
import itertools, statistics as st, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.subspace_meaning import TOPICS

M, dev, ND = "HuggingFaceTB/SmolLM2-135M", "mps", 64
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
W = model.get_output_embeddings().weight.detach().float().cpu()
Wc = W - W.mean(0, keepdim=True)

prompts, topic = [], []
for t, ps in TOPICS.items():
    prompts += ps; topic += [t] * len(ps)

def auc(a, b):
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b))

blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
g = torch.Generator().manual_seed(0)

for L in (4, 12, 20):
    store = []
    h = model.model.layers[L].register_forward_hook(
        lambda m, i, o: store.append(((o if torch.is_tensor(o) else o[0])[0, 1:])
                                     .detach().float().cpu().mean(0)) and None)
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    h.remove()
    H = torch.stack(store); Hc = H - H.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(blob["J"][L].float(), full_matrices=False)
    C = Hc @ Vh[:ND].T

    def best_over_dirs(ai, bi, dirs=range(ND)):
        sc = [(abs(auc(C[ai, j].tolist(), C[bi, j].tolist()) - .5) + .5, j) for j in dirs]
        return max(sc)

    real, nulls, held = [], [], []
    for x, y in itertools.combinations(sorted(TOPICS), 2):
        ai = [i for i in range(len(topic)) if topic[i] == x]
        bi = [i for i in range(len(topic)) if topic[i] == y]
        s, j = best_over_dirs(ai, bi); real.append(s)

        # null: same scan, shuffled labels
        pool = ai + bi
        for _ in range(60):
            pm = [pool[k] for k in torch.randperm(len(pool), generator=g).tolist()]
            nulls.append(best_over_dirs(pm[:len(ai)], pm[len(ai):])[0])

        # held out: choose the direction on half, score it on the other half
        ha, hb = ai[::2], bi[::2]      # selection half
        ta, tb = ai[1::2], bi[1::2]    # test half
        _, jh = best_over_dirs(ha, hb)
        held.append(abs(auc(C[ta, jh].tolist(), C[tb, jh].tolist()) - .5) + .5)

    nulls.sort()
    print(f"\n=== layer {L} ===")
    print(f"  best-of-{ND} AUC, real labels      : mean {st.mean(real):.3f}  "
          f"min {min(real):.3f}")
    print(f"  best-of-{ND} AUC, shuffled labels  : mean {st.mean(nulls):.3f}  "
          f"95th pct {nulls[int(.95*len(nulls))]:.3f}  max {nulls[-1]:.3f}")
    thr = nulls[int(.95 * len(nulls))]
    print(f"  pairs beating the shuffled 95th pct: "
          f"{sum(1 for r in real if r > thr)}/{len(real)}")
    print(f"  HELD OUT (direction picked on other prompts): mean {st.mean(held):.3f}  "
          f"{sum(1 for x in held if x > .75)}/{len(held)} still above 0.75")
