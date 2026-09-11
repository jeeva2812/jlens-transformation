"""Read with eigenvectors, steer with singular vectors -- and why."""
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
TEAL,ROSE,GREY="#0B6E78","#A83A63","#9AA8AD"
rows=json.load(open("out/steer_eigen.json"))
sh=json.load(open("out/decomp_shootout.json"))["total"]
hen={4:3.13,8:2.78,12:2.62,16:2.02,20:1.64,24:1.13}
def passes(r): return abs(r["shift"])>0.5 and abs(r["shift"])>4*abs(r["rand"])

fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))

# 1 the dissociation
fams=["eigen","SVD u","SVD v","PCA(h)","balanced","random"]
read=[sh[f]["rate"]*100 for f in fams]
x=np.arange(len(fams))
c=[TEAL if f=="eigen" else GREY for f in fams]
ax[0].bar(x,read,.6,color=c,edgecolor="white")
ax[0].set_xticks(x); ax[0].set_xticklabels(fams,fontsize=8.5,rotation=20,ha="right")
ax[0].set_ylabel("% clearing an axis probe"); ax[0].grid(alpha=.25,axis="y")
ax[0].set_title("READING: eigenvectors win\n96 directions per family",fontsize=10.5)
for i,v in enumerate(read):
    ax[0].annotate(f"{v:.0f}%",(i,v),ha="center",fontsize=9,
                   xytext=(0,4),textcoords="offset points")

st_rate={f:sum(passes(r) for r in rows if r["family"]==f)/
           len([r for r in rows if r["family"]==f])*100 for f in ["eigen","SVD v"]}
ax[1].bar([0,1],[st_rate["eigen"],st_rate["SVD v"]],.5,
          color=[TEAL,ROSE],edgecolor="white")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["eigen","SVD v"])
ax[1].set_ylabel("% steering as predicted"); ax[1].grid(alpha=.25,axis="y")
ax[1].set_title("STEERING: singular vectors win\n36 directions per family",fontsize=10.5)
for i,v in enumerate([st_rate["eigen"],st_rate["SVD v"]]):
    ax[1].annotate(f"{v:.0f}%",(i,v),ha="center",fontsize=9,
                   xytext=(0,4),textcoords="offset points")

# 3 why: the advantage is the non-normality
xs,ys=[],[]
for l,h in hen.items():
    s=[r for r in rows if r["layer"]==l and r["family"]=="SVD v"]
    e=[r for r in rows if r["layer"]==l and r["family"]=="eigen"]
    if not s or not e: continue
    xs.append(h); ys.append(st.median(abs(r["shift"]) for r in s)/
                            max(st.median(abs(r["shift"]) for r in e),1e-6))
ax[2].scatter(xs,ys,s=70,c=TEAL,zorder=3)
for h,y,l in zip(xs,ys,[l for l in hen if any(r["layer"]==l for r in rows)]):
    ax[2].annotate(f"L{l}",(h,y),xytext=(6,4),textcoords="offset points",fontsize=8.5)
z=np.polyfit(xs,ys,1); xx=np.linspace(min(xs),max(xs),40)
ax[2].plot(xx,np.polyval(z,xx),c=ROSE,lw=1.8,ls="--")
ax[2].axhline(1,c="0.5",lw=1,ls=":")
ax[2].set_xlabel("departure from normality  $\\sigma_1/|\\lambda_1|$")
ax[2].set_ylabel("SVD / eigen  median |shift|")
ax[2].grid(alpha=.25)
ax[2].set_title(f"WHY: the advantage IS the non-normality\n"
                f"r = {np.corrcoef(xs,ys)[0,1]:+.2f}; at the target they coincide",
                fontsize=10.5)
fig.suptitle("Read with eigenvectors, steer with singular vectors  ·  "
             "Weyl gives $\\sigma_1 \\geq |\\lambda_1|$, and the gap is the whole story",
             fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F13_dissociation.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F13_dissociation.png")
