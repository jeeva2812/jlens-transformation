"""Concrete next-token example plus per-prompt corpus effects (not generation)."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jlens.pullback_corpus import PROSE_A, PROSE_B, CODE
from jlens.pullback_steer import CONCEPTS, PROMPTS
from jlens.pullback_robustness import single_token_ids, AddEverywhere
from jlens.lens import LensSpec, lens_vectors


def main():
    torch.set_num_threads(4)
    data=json.loads(Path('out/rare/pullback_corpus.json').read_text())
    layer=20; prompt='On the table there was a'; model_id='HuggingFaceTB/SmolLM2-135M'
    tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.float32,local_files_only=True).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W=model.get_output_embeddings().weight.detach(); Wn=F.normalize(W-W.mean(0,keepdim=True),dim=1)
    ids=single_token_ids(tok,CONCEPTS['food'].split()); train,test=ids[::2],ids[1::2]
    w=F.normalize(Wn[train].mean(0),dim=0)
    dirs={'direct_w':w}
    for name,texts in [('prose_a',PROSE_A),('prose_b',PROSE_B),('code',CODE)]:
        batches=[]
        for start in range(0,len(texts),6):
            e=tok(texts[start:start+6],return_tensors='pt',padding=True,truncation=True,max_length=64)
            batches.append((e['input_ids'],e['attention_mask']))
        dirs[name]=lens_vectors(model,batches,LensSpec(layer=layer,target_layer=28,n_prompts=18,max_len=64,skip_first=4,weighting='uniform'),w[None])[0]
        print('estimated',name,flush=True)
    dirs['pooled_prose']=(dirs['prose_a']+dirs['prose_b'])/2
    e=tok(prompt,return_tensors='pt')
    with torch.no_grad():
        logits={'clean':model(**e).logits[0,-1].float()}
        for name,d in dirs.items():
            with AddEverywhere(model,layer,d,.15*data['scales'][str(layer)]):
                logits[name]=model(**e).logits[0,-1].float()
    result=dict(prompt=prompt,layer=layer,concept='food',dose=.15,train_tokens=[tok.decode(i) for i in train],test_tokens=[tok.decode(i) for i in test],methods={})
    for name,z in logits.items():
        p=z.softmax(-1); values,indices=p.topk(10)
        result['methods'][name]=dict(top10=[dict(token=tok.decode(i),probability=float(v)) for i,v in zip(indices,values)],
            heldout_food_probabilities={tok.decode(i):float(p[i]) for i in test})
    Path('out/rare/prompt_example.json').write_text(json.dumps(result,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    names=['direct_w','prose_a','prose_b','pooled_prose','code']
    matrix=np.array([next(r['prompt_lifts'] for r in data['effect_rows'] if r['layer']==layer and r['concept']=='food' and r['method']==name) for name in names])
    im=axs[0].imshow(matrix,aspect='auto',cmap='viridis')
    axs[0].set_yticks(range(len(names)),names); axs[0].set_xticks(range(12),range(1,13)); axs[0].set(xlabel='Evaluation prompt index (table example = 8)',title='Food steering at layer 20: each evaluation prompt')
    fig.colorbar(im,ax=axs[0],label='Held-out log-probability lift minus controls')
    x=np.arange(len(test)); width=.23
    for j,name in enumerate(['clean','pooled_prose','code']):
        vals=list(result['methods'][name]['heldout_food_probabilities'].values())
        axs[1].bar(x+(j-1)*width,np.array(vals)*100,width,label=name)
    axs[1].set_xticks(x,[tok.decode(i).strip() for i in test],rotation=60,ha='right')
    axs[1].set(ylabel='Next-token probability (%)',title='Actual prompt: “On the table there was a”'); axs[1].legend()
    fig.suptitle('SmolLM2-135M · fixed food target · equal intervention norm · no sampled completions')
    fig.tight_layout(); fig.savefig('out/figs_core/fig7_prompt_and_corpus.png',dpi=170)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
