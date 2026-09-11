"""Plot the two held-out evaluation domains from saved corpus experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    files={'Neutral prose evaluation':'out/rare/pullback_corpus.json',
           'Code-like evaluation':'out/rare/pullback_corpus_code_eval.json'}
    methods=['direct_w','code','pooled_prose','saved_global']
    labels=['Direct target w','Code-estimated','Pooled-prose-estimated','Saved 25-Pile J']
    colors=['#999999','#D55E00','#0072B2','#009E73']
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    summary={}
    for ax,(domain,path) in zip(axs,files.items()):
        d=json.loads(Path(path).read_text()); vals=[]
        summary[domain]={}
        for m in methods:
            x=np.array([r['lift'] for r in d['effect_rows'] if r['method']==m])
            vals.append(x.mean()); summary[domain][m]={'mean':float(x.mean()),'n_cells':len(x),'cell_sd':float(x.std(ddof=1))}
        ax.bar(range(4),vals,color=colors)
        ax.set_xticks(range(4),labels,rotation=24,ha='right')
        ax.set_title(domain); ax.set_ylabel('Mean held-out concept lift (nats vs controls)')
        ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals): ax.text(i,v+.035,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Changing the Jacobian-estimation corpus changes causal transfer\nSame seven concepts × four layers; equal intervention norm')
    fig.tight_layout(); fig.savefig('out/figs_core/fig9_bidirectional_corpus.png',dpi=180)
    Path('out/rare/pullback_corpus_bidirectional_summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
