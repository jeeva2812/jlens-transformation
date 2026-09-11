"""Depth is a lever: the same weight change buys ~4x more early than late."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
TEAL,ROSE,GREY,AMB="#0B6E78","#A83A63","#9AA8AD","#8A6410"
R=json.load(open("out/position_sweep.json"))
L=np.array([r["layer"] for r in R],dtype=float)
pg=np.array([r["pu_graft"] for r in R]); pr=np.array([r["pu_rand"] for r in R])
kg=np.array([r["kl_graft"] for r in R]); kr=np.array([r["kl_rand"] for r in R])
dW=np.array([r["dW"] for r in R])
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))

ax[0].plot(L,pg,"o-",c=TEAL,ms=5,lw=2,label="fine-tuned layer grafted in")
ax[0].plot(L,pr,"o-",c=ROSE,ms=5,lw=2,label="random noise, matched $\\|\\Delta W\\|$")
for v,c,lab in [(pg,TEAL,"graft"),(pr,ROSE,"random")]:
    z=np.polyfit(L,np.log(v),1)
    ax[0].plot(L,np.exp(np.polyval(z,L)),c=c,lw=1,ls=":",alpha=.7)
ax[0].set_yscale("log"); ax[0].set_xlabel("layer carrying the change")
ax[0].set_ylabel("$\\|\\Delta J\\|$ per unit $\\|\\Delta W\\|$")
ax[0].grid(alpha=.25); ax[0].legend(fontsize=8.5,frameon=False)
ax[0].set_title(f"Same-size change, different depth\n"
                f"graft r={np.corrcoef(L,pg)[0,1]:+.2f} ({pg[0]/pg[-1]:.1f}$\\times$)   "
                f"random r={np.corrcoef(L,pr)[0,1]:+.2f} ({pr[0]/pr[-1]:.1f}$\\times$)",
                fontsize=10.5)

ax[1].plot(L,kg,"o-",c=TEAL,ms=5,lw=2,label="graft")
ax[1].plot(L,kr,"o-",c=ROSE,ms=5,lw=2,label="random")
ax[1].set_xlabel("layer carrying the change"); ax[1].set_ylabel("KL from base on real text")
ax[1].grid(alpha=.25); ax[1].legend(fontsize=8.5,frameon=False)
ax[1].set_title(f"It shows in behaviour too, more weakly\n"
                f"r = {np.corrcoef(L,kg)[0,1]:+.2f} (graft), "
                f"{np.corrcoef(L,kr)[0,1]:+.2f} (random)",fontsize=10.5)

ax[2].plot(L,dW,"o-",c=AMB,ms=5,lw=2)
ax[2].set_xlabel("layer"); ax[2].set_ylabel("$\\|\\Delta W\\|$ the LoRA actually put there")
ax[2].grid(alpha=.25)
ax2b=ax[2].twinx()
ax2b.plot(L,pg,"o--",c=TEAL,ms=4,lw=1.4,alpha=.75)
ax2b.set_ylabel("$\\|\\Delta J\\|$ per unit",color=TEAL,fontsize=9)
ax2b.tick_params(axis="y",labelcolor=TEAL,labelsize=8)
ax[2].set_title(f"The fine-tune spends where it buys least\n"
                f"$\\|\\Delta W\\|$ vs depth: r = {np.corrcoef(L,dW)[0,1]:+.2f}, "
                f"efficiency: {np.corrcoef(L,pg)[0,1]:+.2f}",fontsize=10.5)
fig.suptitle("Depth is a lever  ·  an identical weight change moves the transport "
             "~4$\\times$ more at layer 2 than at layer 27",fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F15_position_sweep.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F15_position_sweep.png")
