"""readout(subspace, activation): decomposing J h over the singular basis.

    J h = sum_i sigma_i <v_i, h> u_i

so each singular component contributes a FIXED board pattern u_i, scaled by a
POSITION-DEPENDENT coefficient c_i(h) = sigma_i <v_i, h>. That is the readout as
a function of both a subspace and an activation.

Two questions this answers:
  1. How many components does the readout need?  Truncate at k and compare the
     readout of J P_k h against the readout of the full J h, and against the
     model's own logits.
  2. Do different positions excite different components, or does one component
     dominate everywhere?  If the latter, the subspace is not capturing
     position-specific features at all.

Note on the norm: the decomposition of J h is exact, but ln_f is not linear, so
the READOUTS of the components do not sum to the readout of J h. Truncation is
therefore measured the honest way -- readout of the truncated input, not a sum
of component readouts.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "austindavis/chess-gpt2-uci-8x8x512"; SQ0 = 4
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NCOMP = 24

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
for p in m.parameters():
    p.requires_grad_(False)
_, ln = _find_blocks_and_norm(m)
W_U = m.get_output_embeddings().weight.detach().float()
blob = torch.load("out/chess/J_chess.pt", map_location="cpu", weights_only=False)
target = blob["target"]; layers = [l for l in blob["layers"] if l < target]
sample = json.load(open("out/chess/sample_game.json"))
ids, fens = sample["ids"], sample["fens"]

def rd(v):
    with torch.no_grad():
        return (ln(v.float().unsqueeze(0)).squeeze(0) @ W_U.T)[SQ0:SQ0 + 64]

SV = {}
for l in layers:
    U, S, Vh = torch.linalg.svd(blob["J"][l].float())
    SV[l] = (U, S, Vh)

cos = torch.nn.functional.cosine_similarity
recs = []
for k in range(1, min(len(ids), len(fens))):
    t = torch.tensor([ids[:k]])
    with _MultiCapture(m, layers, target) as cap:
        with torch.no_grad():
            m(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False)
        hs = {l: cap.h[l][0, -1].detach().clone() for l in layers}
        h_t = cap.target(target)[0, -1].detach().clone()
    actual = rd(h_t)
    rec = {"ply": k, "trunc": {}, "coef": {}, "compboard": {}}
    for l in layers:
        U, S, Vh = SV[l]
        h = hs[l]
        c = S * (Vh @ h)                       # coefficients sigma_i <v_i, h>
        full = rd(U @ c)                       # == rd(J h)
        tr = []
        for kk in KS:
            approx = U[:, :kk] @ c[:kk]
            tr.append({"k": kk,
                       "cos_full": round(float(cos(rd(approx), full, dim=0)), 3),
                       "cos_model": round(float(cos(rd(approx), actual, dim=0)), 3)})
        rec["trunc"][str(l)] = tr
        rec["coef"][str(l)] = [round(float(x), 3) for x in c[:NCOMP]]
        rec["compboard"][str(l)] = [[round(float(x), 3) for x in rd(c[i] * U[:, i])]
                                    for i in range(8)]
    recs.append(rec)
    if k % 20 == 0:
        print("ply", k, flush=True)

Path("out/chess/decomp.json").write_text(json.dumps(
    {"target": target, "layers": layers, "ks": KS, "ncomp": NCOMP, "plies": recs}))

print("\n== components needed: mean cosine of readout(J P_k h) vs the MODEL's logits ==")
hdr = "layer " + " ".join(f"k={k:<4}" for k in KS)
print(hdr)
for l in layers:
    row = []
    for i, kk in enumerate(KS):
        row.append(sum(r["trunc"][str(l)][i]["cos_model"] for r in recs) / len(recs))
    print(f"L{l:<5}" + " ".join(f"{v:>6.3f}" for v in row))

print("\n== which component dominates? argmax_i |c_i(h)| across plies ==")
for l in layers:
    cnt = Counter()
    for r in recs:
        c = r["coef"][str(l)]
        cnt[max(range(len(c)), key=lambda i: abs(c[i]))] += 1
    top = ", ".join(f"i={i} ({n}/{len(recs)})" for i, n in cnt.most_common(4))
    print(f"L{l}: {len(cnt)} distinct winners | {top}")
