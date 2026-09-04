"""Are the unflagged directions superpositions, or are they just not there?

"76% of directions clear no axis probe" has at least three explanations and they
are not the same claim:

  (a) they encode features my ten English axes do not cover
  (b) they are MIXTURES of features -- superposition
  (c) the model never visits them at all, so there is no feature to find

(c) is not a feature story, and it already has evidence: only 25-32% of real
transported activation lives in the top-64 raw singular directions. So before
reaching for superposition, split the unflagged directions by whether the model
actually occupies them.

For each direction compute how much real activation projects onto it, relative
to what a random direction gets. On-manifold AND unflagged is where a
superposition hypothesis would live. Off-manifold AND unflagged is a direction
the transport amplifies and the data never uses -- no feature required.
"""
import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _MultiCapture

M="HuggingFaceTB/SmolLM2-135M-Instruct"
blob=torch.load("out/ft/J_step0.pt",map_location="cpu",weights_only=False)
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
tok=AutoTokenizer.from_pretrained(M)
D=json.load(open("out/explorer/smollm2_ft.json"))
ds=load_dataset("NeelNanda/pile-10k",split="train")
texts=[ds[i]["text"] for i in range(30)]

print(f"{'layer':>5} {'group':<26} {'n':>4} {'on-manifold energy':>19} {'vs random':>10}")
print("-"*70)
rows=[]
for l in [8,12,16,20]:
    H=[]
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=128)
        with _MultiCapture(model,[l],28) as cap:
            with torch.no_grad(): model(**enc,use_cache=False)
            H.append(cap.h[l][0].detach().float())
    H=torch.cat(H,0); H=H-H.mean(0,keepdim=True)
    tot=H.pow(2).sum()
    J=blob["J"][l].float()
    U,S,Vh=torch.linalg.svd(J)
    # energy of real activations along each INPUT direction v_i (layer-l space)
    e=[(H@Vh[i]).pow(2).sum().item()/tot.item() for i in range(10)]
    g=torch.Generator().manual_seed(0)
    rnd=[]
    for _ in range(200):
        r=torch.randn(J.shape[0],generator=g); r=r/r.norm()
        rnd.append((H@r).pow(2).sum().item()/tot.item())
    base=float(np.mean(rnd))
    key=f"step0|{l}"
    dirs=D["single"].get(key,{}).get("dirs",[])
    fl=[d["i"] for d in dirs if d.get("axes")]
    for name,idx in [("flagged by an axis",fl),
                     ("unflagged",[i for i in range(10) if i not in fl])]:
        if not idx: continue
        v=[e[i] for i in idx]
        print(f"{l:>5} {name:<26} {len(idx):>4} {np.mean(v):>18.5f} "
              f"{np.mean(v)/base:>9.1f}x")
        rows.append((l,name,len(idx),float(np.mean(v)),float(np.mean(v)/base)))
    print(f"{'':>5} {'(random direction)':<26} {'':>4} {base:>18.5f} {1.0:>9.1f}x")
json.dump(rows,open("out/superposition_check.json","w"),indent=1)
