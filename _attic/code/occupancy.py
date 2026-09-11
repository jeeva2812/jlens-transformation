"""Occupancy vs gain, across every singular direction, not just the top 10.

The 10-direction check said the leading input directions of J are LESS occupied
by real activations than a random direction is. If that is real and not an
artefact of looking only at the top, occupancy should rise with rank -- the
directions the transport amplifies least should be the ones the data actually
lives in. That is a strong, easily-falsified prediction, so test it.

Sanity anchor: a random unit direction in d dims should capture 1/d of the
energy. If the random baseline does not come out at 1/d, the measurement is
wrong and nothing else here means anything.
"""
import torch, numpy as np, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _MultiCapture
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M="HuggingFaceTB/SmolLM2-135M-Instruct"
blob=torch.load("out/ft/J_step0.pt",map_location="cpu",weights_only=False)
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).eval()
tok=AutoTokenizer.from_pretrained(M)
ds=load_dataset("NeelNanda/pile-10k",split="train")
texts=[ds[i]["text"] for i in range(30)]

fig,axes=plt.subplots(1,3,figsize=(15.2,4.4))
out={}
for ax,l in zip(axes,[8,16,20]):
    H=[]
    for t in texts:
        enc=tok(t,return_tensors="pt",truncation=True,max_length=128)
        with _MultiCapture(model,[l],28) as cap:
            with torch.no_grad(): model(**enc,use_cache=False)
            H.append(cap.h[l][0].detach().float())
    H=torch.cat(H,0); H=H-H.mean(0,keepdim=True)
    tot=H.pow(2).sum().item(); d=H.shape[1]
    J=blob["J"][l].float()
    U,S,Vh=torch.linalg.svd(J)
    occ=((H@Vh.T).pow(2).sum(0)/tot).numpy()          # energy per input direction
    g=torch.Generator().manual_seed(0)
    R=torch.randn(d,200,generator=g); R=R/R.norm(dim=0,keepdim=True)
    rnd=((H@R).pow(2).sum(0)/tot).numpy()
    print(f"layer {l}: random baseline {rnd.mean():.6f}  vs 1/d = {1/d:.6f}  "
          f"(ratio {rnd.mean()*d:.2f})")
    print(f"  occupancy of rank 0-9   : {occ[:10].mean():.6f}  "
          f"({occ[:10].mean()/rnd.mean():.2f}x random)")
    print(f"  occupancy of rank 50-99 : {occ[50:100].mean():.6f}  "
          f"({occ[50:100].mean()/rnd.mean():.2f}x random)")
    print(f"  occupancy of last 50    : {occ[-50:].mean():.6f}  "
          f"({occ[-50:].mean()/rnd.mean():.2f}x random)")
    r=np.corrcoef(np.arange(d), occ)[0,1]
    print(f"  corr(rank, occupancy) = {r:+.3f}")
    out[l]={"occ":occ.tolist(),"sigma":S.tolist(),"rand":float(rnd.mean()),
            "d":d,"corr_rank_occ":float(r)}
    ax.semilogy(occ,lw=.8,c="#0B6E78",label="occupancy of $v_i$ by real activations")
    ax.axhline(rnd.mean(),ls="--",c="crimson",lw=1.2,
               label=f"random direction (1/d = {1/d:.4f})")
    ax2=ax.twinx()
    ax2.plot(S.numpy(),lw=1.4,c="#A83A63",alpha=.75)
    ax2.set_ylabel("$\\sigma_i$  (gain)",color="#A83A63",fontsize=9)
    ax2.tick_params(axis='y',labelcolor="#A83A63",labelsize=8)
    ax.set_xlabel("singular direction rank"); ax.set_ylabel("fraction of activation energy")
    ax.set_title(f"layer {l}   corr(rank, occupancy) = {r:+.2f}",fontsize=10.5)
    ax.legend(fontsize=8,frameon=False,loc="lower right"); ax.grid(alpha=.2)
fig.suptitle("The directions J amplifies most are the ones the model occupies least",
             fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F8_gain_vs_occupancy.png",dpi=150,
                                bbox_inches="tight")
json.dump(out,open("out/gain_occupancy.json","w"))
print("\nwrote out/report/F8_gain_vs_occupancy.png")
