"""Cross-model rank and topic-subspace figures from saved experiment JSON."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT=Path('out/rare'); FIG=Path('out/figs_core'); FIG.mkdir(parents=True,exist_ok=True)
MODELS=[
 ('SmolLM2-135M',576,'pullback_rank.json','subspace_meaning_full.json',[4,12,20,28],28),
 ('Qwen3.5-4B',2560,'qwen35_4b_pullback_rank.json','qwen35_4b_subspace_meaning.json',[4,13,21,28],30),
 ('OLMo-3-7B',4096,'olmo3_7b_pullback_rank.json','olmo3_7b_subspace_meaning.json',[4,12,22,28],30),
]


def rank_rows(path):
    d=json.loads((ROOT/path).read_text()); return d if isinstance(d,list) else d['rows']


def main():
    fig,ax=plt.subplots(figsize=(9,5.5))
    summary={}
    for name,width,rfile,_,_,_ in MODELS:
        groups=defaultdict(list)
        for r in rank_rows(rfile): groups[str(r['k'])].append(r['lift'])
        xs=[]; ys=[]
        for k,v in groups.items():
            xs.append(width if k=='full' or int(k)==width else int(k)); ys.append(float(np.mean(v)))
        order=np.argsort(xs); xs=np.array(xs)[order]; ys=np.array(ys)[order]
        ax.plot(xs,ys,'-o',label=name); ax.scatter([width],[ys[xs==width][0]],s=130,facecolors='none',edgecolors=ax.lines[-1].get_color(),linewidths=2)
        summary[name]={str(int(x)):float(y) for x,y in zip(xs,ys)}
    ax.set_xscale('log',base=2); ax.set(xlabel='Singular components retained (ring = exact full pullback)',ylabel='Mean held-out concept lift',title='How much of the Jacobian spectrum does steering need?')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'fig10_cross_model_rank.png',dpi=190)

    fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    topic_summary={}
    for ax,(name,width,_,sfile,layers,target) in zip(axs,MODELS):
        d=json.loads((ROOT/sfile).read_text()); vals={'J':[],'PCA':[],'Random':[]}
        for l in layers:
            if name.startswith('Smol'):
                r=d[str(l)]['topic']['8']; row={'J':r['J'],'PCA':r['pca'],'Random':r['rand']}
            else:
                r=next(x for x in d['layers'][str(l)]['rows'] if x['k']==8); row={'J':r['J'],'PCA':r['pca'],'Random':r['random_mean']}
            for k in vals: vals[k].append(row[k])
        x=np.array(layers)/target
        for method,style in [('J','-o'),('PCA','-s'),('Random','--^')]: ax.plot(x,vals[method],style,label=method)
        ax.axhline(1/6,color='gray',ls=':',label='chance' if name==MODELS[0][0] else None)
        ax.set(title=name,xlabel='Source depth / target depth',ylim=(.1,1.05)); ax.grid(alpha=.2)
        topic_summary[name]={m:v for m,v in vals.items()}
    axs[0].set_ylabel('Six-topic accuracy using eight dimensions'); axs[-1].legend(loc='lower right')
    fig.suptitle('The SmolLM early J-subspace advantage does not replicate at 4B or 7B\nLarge-model J curves use seeded randomized rank-80 SVD; PCA is fitted to the same 72 prompts')
    fig.tight_layout(); fig.savefig(FIG/'fig11_cross_model_topic_subspace.png',dpi=190)
    (ROOT/'cross_model_extension_summary.json').write_text(json.dumps({'rank':summary,'topic_top8':topic_summary},indent=2))

    prompt_files=[('SmolLM2-135M','smollm_prompt_example_matched.json'),('Qwen3.5-4B','qwen35_4b_prompt_example.json'),('OLMo-3-7B','olmo3_7b_prompt_example.json')]
    methods=['clean','direct_w','pullback']; labels=['Clean','Direct w','Jᵀw']; x=np.arange(3); width=.24
    fig,ax=plt.subplots(figsize=(10,5.5)); prompt_summary={}
    for j,m in enumerate(methods):
        vals=[]
        for model_name,file in prompt_files:
            d=json.loads((ROOT/file).read_text()); vals.append(100*d['methods'][m]['mean_heldout_food_probability'])
            prompt_summary.setdefault(model_name,{})[m]=d['methods'][m]
        ax.bar(x+(j-1)*width,vals,width,label=labels[j])
    for i,(model_name,file) in enumerate(prompt_files):
        d=json.loads((ROOT/file).read_text()); clean=d['methods']['clean']['top10'][0]['token'].strip() or repr(d['methods']['clean']['top10'][0]['token']); pb=d['methods']['pullback']['top10'][0]['token'].strip() or repr(d['methods']['pullback']['top10'][0]['token'])
        ax.text(i,max(100*d['methods'][m]['mean_heldout_food_probability'] for m in methods)*1.12,f'top: {clean} → {pb}',ha='center',fontsize=9)
    ax.set_yscale('log'); ax.set_ylim(.006,.34); ax.set_xticks(x,[p[0] for p in prompt_files]); ax.set(ylabel='Mean probability of held-out food tokens (%) · log scale',title='Actual prompt on all models: “On the table there was a”')
    ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'fig12_cross_model_prompt_example.png',dpi=190)
    summary_path=ROOT/'cross_model_extension_summary.json'; all_summary=json.loads(summary_path.read_text()); all_summary['prompt_example']=prompt_summary; summary_path.write_text(json.dumps(all_summary,indent=2))


if __name__=='__main__': main()
