import json, numpy as np, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=json.load(open("out/steer_compare.json"))
TEAL,ROSE,GREY,AMB="#0B6E78","#A83A63","#9AA8AD","#8A6410"
COL={"SVD(J)":TEAL,"PCA(h)":ROSE,"whitened":AMB,"random":GREY}
by=collections.defaultdict(list)
for r in R: by[r["family"]].append(r)
fam=["SVD(J)","PCA(h)","whitened","random"]
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))

x=np.arange(4)
ax[0].bar(x,[np.median([abs(r["shift"]) for r in by[f]]) for f in fam],
          .6,color=[COL[f] for f in fam],edgecolor="white")
for i,f in enumerate(fam):
    p=sum(abs(r["shift"])>0.5 and abs(r["shift"])>4*abs(r["rand"]) for r in by[f])
    ax[0].annotate(f"{p}/{len(by[f])}\npass",(i,np.median([abs(r['shift']) for r in by[f]])),
                   textcoords="offset points",xytext=(0,6),ha="center",fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(fam,fontsize=9.5)
ax[0].set_ylabel("median |shift|"); ax[0].grid(alpha=.25,axis="y")
ax[0].set_title("SVD(J) steers best — despite the LOWEST occupancy",fontsize=10.5)

for f in fam:
    ax[1].scatter([r["gain"] for r in by[f]],[abs(r["shift"]) for r in by[f]],
                  s=26,c=COL[f],alpha=.7,edgecolors="none",label=f)
g=np.array([r["gain"] for r in R]); s=np.array([abs(r["shift"]) for r in R])
z=np.polyfit(np.log10(g),s,1); xx=np.linspace(g.min(),g.max(),60)
ax[1].plot(xx,np.polyval(z,np.log10(xx)),c="0.35",lw=1.6,ls="--")
ax[1].set_xlabel("gain  $\\|Jd\\|$"); ax[1].set_ylabel("|shift|")
ax[1].legend(fontsize=8.5,frameon=False); ax[1].grid(alpha=.25)
ax[1].set_title("gain predicts steerability   r = +0.66",fontsize=10.5)

for f in fam:
    ax[2].scatter([max(r["occ"],1e-3) for r in by[f]],[abs(r["shift"]) for r in by[f]],
                  s=26,c=COL[f],alpha=.7,edgecolors="none",label=f)
ax[2].set_xscale("log")
ax[2].set_xlabel("occupancy  ($\\times$ a random direction)"); ax[2].set_ylabel("|shift|")
ax[2].grid(alpha=.25); ax[2].legend(fontsize=8.5,frameon=False)
ax[2].set_title("occupancy does not   r = $-$0.22\n"
                "partial r given gain = +0.03",fontsize=10.5)
fig.suptitle("What makes a direction steerable: gain, not occupancy  ·  "
             "96 directions, SmolLM2-135M, 4 layers",fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F9_gain_not_occupancy.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F9_gain_not_occupancy.png")
