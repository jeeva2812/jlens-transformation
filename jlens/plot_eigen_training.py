"""Training drives the residual transport from amplifying to rotating."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
d=json.load(open("out/eigen_training.json"))
TEAL,ROSE,AMB="#0B6E78","#A83A63","#8A6410"
labs=[v["label"] for v in d.values()]
x=np.arange(len(labs))
fig,ax=plt.subplots(1,3,figsize=(15.4,4.4))
for l,c in [("8",TEAL),("16",AMB),("24",ROSE)]:
    g=[v[l]["gt1"] for v in d.values() if l in v]
    r=[v[l]["rot_angle_mean"]*57.3 for v in d.values() if l in v]
    n=[v[l]["near1"] for v in d.values() if l in v]
    ax[0].plot(x,g,"o-",c=c,ms=5,label=f"layer {l}")
    ax[1].plot(x,r,"o-",c=c,ms=5,label=f"layer {l}")
    ax[2].plot(x,n,"o-",c=c,ms=5,label=f"layer {l}")
for a_,t,yl in [(ax[0],"Self-reinforcing channels collapse\n"
                       "$|\\lambda|>1$, out of 4096","count with $|\\lambda|>1$"),
                (ax[1],"…and the transport turns to rotation\n"
                       "mean rotation of the complex spectrum","degrees per layer"),
                (ax[2],"Pass-through channels\n$|\\lambda|$ within 0.15 of 1",
                 "count near $|\\lambda|=1$")]:
    a_.set_xticks(x); a_.set_xticklabels(labs,rotation=55,ha="right",fontsize=8)
    a_.set_title(t,fontsize=10.5); a_.set_ylabel(yl); a_.grid(alpha=.25)
    a_.legend(frameon=False,fontsize=9); a_.axvline(6.5,c="0.75",ls="--",lw=1)
ax[0].set_yscale("symlog")
ax[0].annotate("2102 at init\n$\\rightarrow$ 32 at the end\n(layer 8)",
               xy=(0,2102),xytext=(2.0,300),fontsize=8.5,color=TEAL,
               arrowprops=dict(arrowstyle="->",color=TEAL,lw=1.1))
fig.suptitle("Training converts the residual transport from AMPLIFYING to ROTATING"
             "  ·  Olmo 3 7B, eigenvalues of $J_\\ell$",fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig("out/report/F10_eigen_training.png",dpi=150,
                                bbox_inches="tight")
print("wrote out/report/F10_eigen_training.png")
for l in ["8","16","24"]:
    v=list(d.values())
    print(f"  layer {l}: |lam|>1  {v[0][l]['gt1']:>5} -> {v[-1][l]['gt1']:>5}"
          f"   ({v[0][l]['gt1']/max(v[-1][l]['gt1'],1):.1f}x drop)"
          f"   rotation {v[0][l]['rot_angle_mean']*57.3:.1f} -> "
          f"{v[-1][l]['rot_angle_mean']*57.3:.1f} deg")
