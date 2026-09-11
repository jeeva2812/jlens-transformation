import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
d=json.load(open("out/occupancy_control.json"))
TEAL,ROSE,GREY="#0B6E78","#A83A63","#9AA8AD"
ks=[0,1,3,10]; lab=["raw","−top 1","−top 3","−top 10"]
fig,ax=plt.subplots(1,3,figsize=(15.2,4.4))
for i,(L,v) in enumerate(sorted(d.items(),key=lambda x:int(x[0]))):
    if i>2: break
    r=[v[str(k)]["corr"] for k in ks if str(k) in v]
    rt=[v[str(k)]["ratio"] for k in ks if str(k) in v]
    vr=[v[str(k)]["var_top1"]*100 for k in ks if str(k) in v]
    x=np.arange(len(r))
    ax[i].axhline(0,c="0.4",lw=1)
    ax[i].bar(x,r,.6,color=[ROSE if q>0 else TEAL for q in r],edgecolor="white")
    for j,(q,vv) in enumerate(zip(r,vr)):
        ax[i].annotate(f"{q:+.2f}\n({vv:.0f}% in top-1)",(j,q),
                       textcoords="offset points",xytext=(0,8 if q>0 else -26),
                       ha="center",fontsize=8.5)
    ax[i].set_xticks(x); ax[i].set_xticklabels(lab,fontsize=9.5)
    ax[i].set_ylim(-1.15,0.75); ax[i].grid(alpha=.25,axis="y")
    ax[i].set_ylabel("corr(singular rank, occupancy)")
    ax[i].set_title(f"layer {L}",fontsize=11)
ax[0].annotate("positive = J amplifies\nwhat the model AVOIDS",(0,0.35),fontsize=8.5,
               color=ROSE,ha="center")
ax[0].annotate("negative = J amplifies\nwhat the model USES",(2,-0.55),fontsize=8.5,
               color=TEAL,ha="center")
fig.suptitle("The anti-correlation was the massive-activation direction, and it "
             "reverses without it",fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("out/report/F11_occupancy_retraction.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F11_occupancy_retraction.png")
