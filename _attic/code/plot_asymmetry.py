import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=json.load(open("out/reanalyse_occ.json"))
TEAL,ROSE,GREY="#0B6E78","#A83A63","#9AA8AD"
def sel(rows,drop=None):
    r=[x for x in rows if abs(x["gain"]-x["gain_c"])<0.05*max(x["gain"],1e-9)]
    return [x for x in r if not drop or x["family"] not in drop]
st=sel(d["steer"],{"whitened"}); ab=sel(d["ablate"])
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))
for a_,rows,key,lab in [(ax[0],st,"shift","steering  (inject)"),
                        (ax[1],ab,"kl","ablation  (remove)")]:
    g=np.log10([max(r["gain_c"],1e-6) for r in rows])
    y=np.array([abs(r[key]) for r in rows])
    a_.scatter(10**g,y,s=26,c=TEAL if key=="shift" else ROSE,alpha=.65,
               edgecolors="none")
    z=np.polyfit(g,y,1); xx=np.linspace(g.min(),g.max(),50)
    a_.plot(10**xx,np.polyval(z,xx),c="0.3",lw=1.8,ls="--")
    a_.set_xscale("log"); a_.set_xlabel("gain  $\\|Jd\\|$")
    a_.set_ylabel("|shift|" if key=="shift" else "KL from base")
    a_.grid(alpha=.25)
    a_.set_title(f"{lab}\ncorr with gain = {np.corrcoef(g,y)[0,1]:+.2f}",
                 fontsize=10.5)
labels=["steering","ablation"]
raw=[np.corrcoef(np.log10([max(r["occ_raw"],1e-6) for r in rows]),
                 [abs(r[k]) for r in rows])[0,1] for rows,k in [(st,"shift"),(ab,"kl")]]
cln=[np.corrcoef(np.log10([max(r["occ_clean"],1e-6) for r in rows]),
                 [abs(r[k]) for r in rows])[0,1] for rows,k in [(st,"shift"),(ab,"kl")]]
x=np.arange(2); w=0.35
ax[2].bar(x-w/2,raw,w,color=GREY,label="raw occupancy")
ax[2].bar(x+w/2,cln,w,color=TEAL,label="outlier dims removed")
ax[2].axhline(0,c="0.3",lw=1)
for i,(r_,c_) in enumerate(zip(raw,cln)):
    ax[2].annotate(f"{r_:+.2f}",(i-w/2,r_),ha="center",fontsize=9,
                   xytext=(0,6 if r_>0 else -14),textcoords="offset points")
    ax[2].annotate(f"{c_:+.2f}",(i+w/2,c_),ha="center",fontsize=9,
                   xytext=(0,6 if c_>0 else -14),textcoords="offset points")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
ax[2].set_ylabel("corr(occupancy, effect)"); ax[2].grid(alpha=.25,axis="y")
ax[2].legend(fontsize=8.5,frameon=False)
ax[2].set_title("Occupancy's apparent role was the outlier\n"
                "both correlations collapse once it is removed",fontsize=10.5)
fig.suptitle("What actually predicts an intervention: the SIGN OF GAIN flips "
             "between injecting and removing",fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F12_intervention_asymmetry.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F12_intervention_asymmetry.png")
