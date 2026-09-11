"""Prompt-level test of whether corpus pullback differences exceed sampling noise.

Saves each prompt's J(x).T@w for 7 concepts x 4 layers, then performs an exact-
exchangeability Monte Carlo label-permutation test for each pair of 18-prompt
banks. This tests these empirical prompt distributions, not a universal
code/prose causal claim.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from jlens.lens import LensSpec,lens_vectors
from jlens.pullback_corpus import PROSE_A,PROSE_B,CODE
from jlens.pullback_steer import CONCEPTS
from jlens.pullback_robustness import single_token_ids


def bh_count(p,q=.05):
    p=np.sort(np.asarray(p)); ok=np.where(p <= q*np.arange(1,len(p)+1)/len(p))[0]
    return int(ok[-1]+1) if len(ok) else 0


def stats(X,Y):
    pooled=torch.cat([X,Y]); center=pooled.mean(0)
    denom=((pooled-center).pow(2).sum(-1).mean(0)).clamp_min(1e-12)
    delta=(X.mean(0)-Y.mean(0)).pow(2).sum(-1)/denom
    cosine=F.cosine_similarity(X.mean(0),Y.mean(0),dim=-1)
    return delta,cosine


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--nperm',type=int,default=5000); ap.add_argument('--nboot',type=int,default=2000)
    ap.add_argument('--cache',type=Path,default=Path('out/rare/corpus_prompt_pullbacks.pt')); ap.add_argument('--out',type=Path,default=Path('out/rare/corpus_conditioning_permutation.json'))
    a=ap.parse_args(); model_id='HuggingFaceTB/SmolLM2-135M'; layers=[4,12,20,26]
    if a.cache.exists(): saved=torch.load(a.cache,map_location='cpu',weights_only=True); banks=saved['banks']; names=saved['concepts']
    else:
        tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True); tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.float32,local_files_only=True).to(a.device).eval()
        for p in model.parameters():p.requires_grad_(False)
        W=model.get_output_embeddings().weight.detach().float().cpu(); Wn=F.normalize(W-W.mean(0,keepdim=True),dim=1)
        names=list(CONCEPTS); seeds=[]
        for name in names:
            ids=single_token_ids(tok,CONCEPTS[name].split()); seeds.append(F.normalize(Wn[ids[::2]].mean(0),dim=0))
        seeds=torch.stack(seeds).to(a.device); banks={}
        for bank_name,texts in {'prose_a':PROSE_A,'prose_b':PROSE_B,'code':CODE}.items():
            by_prompt=[]
            for i,text in enumerate(texts):
                e=tok(text,return_tensors='pt',truncation=True,max_length=64); batch=[(e['input_ids'].to(a.device),e['attention_mask'].to(a.device))]
                by_layer=[]
                for l in layers:
                    by_layer.append(lens_vectors(model,batch,LensSpec(layer=l,target_layer=28,n_prompts=1,max_len=64,skip_first=4,weighting='uniform'),seeds))
                by_prompt.append(torch.stack(by_layer)); print(bank_name,i+1,'/',len(texts),flush=True)
            banks[bank_name]=torch.stack(by_prompt) # prompt, layer, concept, d
        a.cache.parent.mkdir(parents=True,exist_ok=True); torch.save({'banks':banks,'concepts':names,'layers':layers},a.cache)
    rng=np.random.default_rng(20260911); result={'config':{'nperm':a.nperm,'nboot':a.nboot,'layers':layers,'concepts':names},'comparisons':{}}
    for left,right in [('prose_a','prose_b'),('prose_a','code'),('prose_b','code')]:
        X,Y=banks[left].float(),banks[right].float(); observed,cos=stats(X,Y); pooled=torch.cat([X,Y]); null_global=[]; null_cells=[]
        for _ in range(a.nperm):
            ix=torch.tensor(rng.permutation(len(pooled))); d,_=stats(pooled[ix[:len(X)]],pooled[ix[len(X):]])
            null_global.append(float(d.mean())); null_cells.append(d.numpy())
        null_cells=np.stack(null_cells); pcell=(1+(null_cells>=observed.numpy()).sum(0))/(a.nperm+1)
        boots=[]
        for _ in range(a.nboot):
            xb=X[torch.tensor(rng.integers(0,len(X),len(X)))]; yb=Y[torch.tensor(rng.integers(0,len(Y),len(Y)))]; _,c=stats(xb,yb); boots.append(float(c.mean()))
        global_stat=float(observed.mean()); pglobal=(1+sum(v>=global_stat for v in null_global))/(a.nperm+1)
        result['comparisons'][f'{left}_vs_{right}']={'mean_cosine':float(cos.mean()),'bootstrap_cosine_95':np.quantile(boots,[.025,.975]).tolist(),'normalized_mean_shift':global_stat,'permutation_p_global':pglobal,'fdr_significant_cells_28':bh_count(pcell.ravel()),'cell_p':pcell.tolist()}
        print(left,right,result['comparisons'][f'{left}_vs_{right}'],flush=True)
    a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
