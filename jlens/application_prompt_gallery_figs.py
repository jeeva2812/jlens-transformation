"""Render the complete 12-prompt food-steering gallery and corpus-shift test."""
import json,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


ROOT=Path('out/rare'); FIG=Path('out/figs_core'); FIG.mkdir(parents=True,exist_ok=True)
FILES=[('SmolLM2-135M','smollm_prompt_gallery.json'),('Qwen3.5-4B','qwen35_4b_prompt_gallery.json'),('OLMo-3-7B','olmo3_7b_prompt_gallery.json')]


def tokens(method):
    return '  '.join(f"{x['token'].replace(chr(10),'↵')!r} {100*x['probability']:.1f}%" for x in method['top10'][:3])


def main():
    for model,file in FILES:
        d=json.loads((ROOT/file).read_text()); table=[]; mult=[]
        for r in d['prompt_rows']:
            m=r['methods']; ratio=m['pullback']['mean_heldout_food_probability']/max(m['clean']['mean_heldout_food_probability'],1e-12); mult.append(ratio)
            table.append([textwrap.fill(r['prompt'],32),tokens(m['clean']),tokens(m['direct_w']),tokens(m['pullback']),f'{ratio:.1f}×'])
        fig,ax=plt.subplots(figsize=(19,10));ax.axis('off')
        tab=ax.table(cellText=table,colLabels=['Prompt','Clean top 3','Direct w top 3','Jᵀw top 3','Food-prob. multiplier'],cellLoc='left',colWidths=[.23,.22,.22,.22,.11],loc='center')
        tab.auto_set_font_size(False);tab.set_fontsize(8.5);tab.scale(1,2.7)
        for j in range(5):tab[(0,j)].set_facecolor('#d9eaf7');tab[(0,j)].set_text_props(weight='bold')
        norm=Normalize(vmin=1,vmax=max(mult));cmap=plt.get_cmap('YlGn')
        for i,v in enumerate(mult,1):tab[(i,4)].set_facecolor(cmap(norm(v)))
        ax.set_title(f'{model}: all 12 food-steering prompts at a matched 0.15 dose\nTop-3 next tokens; multiplier is mean probability over held-out food tokens',fontsize=17,pad=20)
        fig.tight_layout();fig.savefig(FIG/f"fig13_{model.split('-')[0].lower()}_prompt_gallery.png",dpi=170,bbox_inches='tight');plt.close(fig)

    d=json.loads((ROOT/'corpus_conditioning_permutation.json').read_text()); names=[]; shifts=[]; cos=[]; ps=[]
    for key,v in d['comparisons'].items():names.append(key.replace('_vs_',' vs\n').replace('_',' '));shifts.append(v['normalized_mean_shift']);cos.append(v['mean_cosine']);ps.append(v['permutation_p_global'])
    fig,axs=plt.subplots(1,2,figsize=(12,4.8));colors=['#999999','#D55E00','#D55E00']
    axs[0].bar(names,shifts,color=colors);axs[0].set(ylabel='Normalized squared mean shift',title='Prompt-level mean-vector shift')
    for i,(x,p) in enumerate(zip(shifts,ps)):axs[0].text(i,x+.02,'p = 1.000' if p==1 else 'p < .001',ha='center')
    axs[1].bar(names,cos,color=colors);axs[1].set(ylim=(0,1),ylabel='Cosine between bank-mean pullbacks',title='Direction agreement')
    for i,x in enumerate(cos):axs[1].text(i,x+.02,f'{x:.3f}',ha='center')
    fig.suptitle('These particular prose and code banks differ beyond prompt-resampling noise\n5,000 label permutations; 7 concepts × 4 layers')
    fig.tight_layout();fig.savefig(FIG/'fig14_corpus_permutation.png',dpi=190);plt.close(fig)


if __name__=='__main__':main()
