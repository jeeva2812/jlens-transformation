"""Minimal version: does the gain/occupancy anti-correlation exist at init?

The full version averaged J over 15 prompts at 3 layers and was still running
after 48 minutes under contention. The measurement is a CORRELATION over 576
directions, not a precise energy estimate, so it does not need that. One layer,
three prompts, and the trained J loaded from disk rather than recomputed.
"""
import torch, numpy as np, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from jlens.lens import _MultiCapture, jacobians_all_layers
M="HuggingFaceTB/SmolLM2-135M-Instruct"; L=16; TARGET=28
tok=AutoTokenizer.from_pretrained(M)
ds=load_dataset("NeelNanda/pile-10k",split="train")
texts=[ds[i]["text"] for i in range(3)]

def occ_stats(model, J):
    H=[]
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=96)
        with _MultiCapture(model,[L],TARGET) as cap:
            with torch.no_grad(): model(**enc,use_cache=False)
            H.append(cap.h[L][0].detach().float())
    H=torch.cat(H,0); H=H-H.mean(0,keepdim=True)
    d=H.shape[1]; tot=H.pow(2).sum()
    U,S,Vh=torch.linalg.svd(J.float())
    occ=((H@Vh.T).pow(2).sum(0)/tot).numpy()*d
    return {"corr": float(np.corrcoef(np.arange(d),occ)[0,1]),
            "top10": float(occ[:10].mean()), "bottom50": float(occ[-50:].mean()),
            "ratio": float(occ[-50:].mean()/max(occ[:10].mean(),1e-9)),
            "sigma_spread": float(S[0]/S.median())}

out={}
t0=time.time()
cfg=AutoConfig.from_pretrained(M); torch.manual_seed(0)
m=AutoModelForCausalLM.from_config(cfg).eval()
for p in m.parameters(): p.requires_grad_(False)
def b():
    for t in texts:
        e=tok(t,return_tensors="pt",truncation=True,max_length=96)
        yield e["input_ids"], e["attention_mask"]
Jr=jacobians_all_layers(m,b(),[L],TARGET,chunk=192)[L]
out["random_init"]=occ_stats(m,Jr)
print(f"random init  ({time.time()-t0:.0f}s): {out['random_init']}",flush=True)
del m

m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
for p in m.parameters(): p.requires_grad_(False)
Jt=torch.load("out/ft/J_step0.pt",map_location="cpu",weights_only=False)["J"][L]
out["trained"]=occ_stats(m,Jt)
print(f"trained: {out['trained']}",flush=True)
json.dump(out,open("out/occupancy_init_vs_trained.json","w"),indent=1)
print("\n" + "="*66)
print(f"{'':<14}{'corr(rank,occ)':>16}{'top10':>9}{'bottom50':>10}{'ratio':>8}{'sigma spread':>14}")
for k,v in out.items():
    print(f"{k:<14}{v['corr']:>+16.3f}{v['top10']:>9.2f}{v['bottom50']:>10.2f}"
          f"{v['ratio']:>8.1f}{v['sigma_spread']:>14.1f}")
