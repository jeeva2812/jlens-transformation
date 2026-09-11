import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
R=json.load(open("out/hybrid_steer.json"))
TEAL,ROSE,GREY,AMB="#0B6E78","#A83A63","#9AA8AD","#8A6410"
def ok(r,k): return abs(r[k])>0.5 and abs(r[k])>4*abs(r["rand"])
M=["eigen","closest","gainwtd","pinv"]
LAB=["eigenvector\n(inject directly)","closest u\n(rank 1)",
     "gain-weighted u\n(rank 1)","pseudo-inverse\n(top 32)"]
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))
rate=[sum(ok(r,k) for r in R)/len(R)*100 for k in M]
med=[st.median(abs(r[k]) for r in R) for k in M]
x=np.arange(4); c=[TEAL,GREY,GREY,ROSE]
ax[0].bar(x,rate,.6,color=c,edgecolor="white")
for i,v in enumerate(rate):
    ax[0].annotate(f"{v:.0f}%",(i,v),ha="center",fontsize=10,
                   xytext=(0,5),textcoords="offset points")
ax[0].set_xticks(x); ax[0].set_xticklabels(LAB,fontsize=8.5)
ax[0].set_ylabel("% steering as the eigenvector's readout predicts")
ax[0].grid(alpha=.25,axis="y")
ax[0].set_title("Delivering eigenvector semantics\nthrough the SVD",fontsize=10.5)
ax[1].bar(x,med,.6,color=c,edgecolor="white")
for i,v in enumerate(med):
    ax[1].annotate(f"{v:.2f}",(i,v),ha="center",fontsize=10,
                   xytext=(0,5),textcoords="offset points")
ax[1].set_xticks(x); ax[1].set_xticklabels(LAB,fontsize=8.5)
ax[1].set_ylabel("median |shift|"); ax[1].grid(alpha=.25,axis="y")
ax[1].set_title("Effect size",fontsize=10.5)
cs=np.array([r["cos_u_ve"] for r in R])
ax[2].hist(cs,bins=16,color=TEAL,edgecolor="white")
ax[2].axvline(np.median(cs),c=ROSE,lw=2)
ax[2].annotate(f"median {np.median(cs):.2f}",(np.median(cs),ax[2].get_ylim()[1]*.85),
               fontsize=9,color=ROSE,ha="left",xytext=(6,0),
               textcoords="offset points")
ax[2].set_xlabel("cos(best singular output direction, eigenvector)")
ax[2].set_ylabel("count"); ax[2].grid(alpha=.25,axis="y")
ax[2].set_title("Why rank 1 is not enough\nno single u matches the eigenvector",
                fontsize=10.5)
fig.suptitle("Use the eigenvector to decide WHAT, the SVD to decide HOW  ·  "
             "the full inversion beats both parents",fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F16_hybrid.png",dpi=150,bbox_inches="tight")
print("wrote out/report/F16_hybrid.png")
