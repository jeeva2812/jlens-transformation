"""Where a weight change sits matters ~3x more than how big it is."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
TEAL,ROSE,GREY="#0B6E78","#A83A63","#9AA8AD"
loc=json.load(open("out/localised_edit.json"))
att=json.load(open("out/layer_attribution.json"))
L=np.array([r["layer"] for r in loc]); dW=np.array([r["dW"] for r in loc])
dJ=np.array([r["dJ"] for r in loc]); pu=np.array([r["per_unit"] for r in loc])
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))

ax[0].plot(L,pu,"o-",c=TEAL,ms=8,lw=2.2)
for x,y in zip(L,pu):
    ax[0].annotate(f"{y:.3f}",(x,y),xytext=(0,8),textcoords="offset points",
                   ha="center",fontsize=8.5)
ax[0].set_xlabel("layer carrying the edit"); ax[0].set_ylabel("$\\|\\Delta J\\|$ per unit $\\|\\Delta W\\|$")
ax[0].grid(alpha=.25); ax[0].set_ylim(0,0.30)
ax[0].set_title(f"Position dominates\nr = {np.corrcoef(L,pu)[0,1]:+.3f}, "
                f"{pu[0]/pu[-1]:.1f}$\\times$ from layer 6 to 26",fontsize=10.5)

ax[1].scatter(dW,dJ,s=80,c=TEAL,zorder=3)
for x,y,l in zip(dW,dJ,L):
    ax[1].annotate(f"L{int(l)}",(x,y),xytext=(7,3),textcoords="offset points",fontsize=8.5)
z=np.polyfit(dW,dJ,1); xx=np.linspace(dW.min(),dW.max(),30)
ax[1].plot(xx,np.polyval(z,xx),c=ROSE,lw=1.8,ls="--")
ax[1].set_xlabel("$\\|\\Delta W\\|$  (how much the layer changed)")
ax[1].set_ylabel("$\\|\\Delta J\\|$  (effect on transport)")
ax[1].grid(alpha=.25)
ax[1].set_title(f"Magnitude alone is ANTI-predictive\nr = {np.corrcoef(dW,dJ)[0,1]:+.3f}",
                fontsize=10.5)

n=np.array([r["naive"] for r in att]); y=np.array([r["actual"] for r in att])
ax[2].scatter(n,y,s=34,c=GREY,alpha=.8,edgecolors="none")
z2=np.polyfit(n,y,1); x2=np.linspace(n.min(),n.max(),30)
ax[2].plot(x2,np.polyval(z2,x2),c="0.4",lw=1.6,ls="--")
ax[2].set_xlabel("$\\|\\Delta A_u\\|$ in the real fine-tune")
ax[2].set_ylabel("actual $\\|\\Delta J\\|$ from reverting layer $u$")
ax[2].grid(alpha=.25)
ax[2].set_title(f"Why the uncontrolled test missed it\n"
                f"r = {np.corrcoef(n,y)[0,1]:+.2f}: a real LoRA changes late layers\n"
                f"MORE, cancelling the position effect",fontsize=10)
fig.suptitle("An identical weight change matters ~3$\\times$ more early than late  ·  "
             "single fine-tuned layer grafted onto the base model",fontsize=12.5,y=1.04)
fig.tight_layout(); fig.savefig("out/report/F14_position_dominates.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F14_position_dominates.png")
