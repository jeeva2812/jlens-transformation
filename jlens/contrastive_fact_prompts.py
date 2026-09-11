"""Prompt-level factual tests for the +Rome/-Paris pullback."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere
from jlens.contrastive_logit_steering import one_token


CASES = [
    {"prompt": "The Eiffel Tower is located in", "paris_answer": "Paris", "rome_answer": "Rome"},
    {"prompt": "The Louvre Museum is located in", "paris_answer": "Paris", "rome_answer": "Rome"},
    {"prompt": "The capital of France is", "paris_answer": "Paris", "rome_answer": "Rome"},
    {"prompt": "The most widely spoken language in Paris is", "paris_answer": "French", "rome_answer": "Italian"},
    {"prompt": "A person from Paris is called", "paris_answer": "French", "rome_answer": "Italian"},
    {"prompt": "The Colosseum is located in", "paris_answer": "Paris", "rome_answer": "Rome"},
    {"prompt": "The capital of Italy is", "paris_answer": "Paris", "rome_answer": "Rome"},
    {"prompt": "The most widely spoken language in Rome is", "paris_answer": "French", "rome_answer": "Italian"},
]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--lens',type=Path,required=True)
    ap.add_argument('--layer',type=int,required=True); ap.add_argument('--scale-json',type=Path,required=True)
    ap.add_argument('--dose',type=float,default=.15); ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16'); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); dtype=getattr(torch,a.dtype)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    batch=tok([c['prompt'] for c in CASES],return_tensors='pt',padding=True); ids=batch['input_ids'].to(a.device); mask=batch['attention_mask'].to(a.device)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).to(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks,_=_find_blocks_and_norm(model); emb=model.get_output_embeddings().weight
    rome,paris=one_token(tok,'Rome'),one_token(tok,'Paris'); w=F.normalize(emb[rome].detach().float().cpu()-emb[paris].detach().float().cpu(),dim=0)
    J=torch.load(a.lens,map_location='cpu',weights_only=False)['J'][a.layer].float(); pb=J.T@w
    scales_blob=json.loads(a.scale_json.read_text()); scales=scales_blob.get('scales',scales_blob.get('config',{}).get('scales',{})); scale=float(scales.get(str(a.layer),scales.get(a.layer)))
    @torch.inference_mode()
    def logits(): return model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
    zs={'clean':logits()}
    for name,d in [('direct_w',w),('pullback',pb),('reverse_pullback',-pb)]:
        with AddEverywhere(blocks[a.layer],d,a.dose*scale,device=a.device,dtype=dtype): zs[name]=logits()
    out={'config':vars(a),'model':a.model,'layer':a.layer,'dose':a.dose,'rows':[]}
    for i,case in enumerate(CASES):
        row=case|{'methods':{}}; rid=one_token(tok,case['rome_answer']); pid=one_token(tok,case['paris_answer'])
        for name,z in zs.items():
            probs=z[i].softmax(-1); vals,ix=probs.topk(10)
            row['methods'][name]={'rome_side_answer':tok.decode(rid),'paris_side_answer':tok.decode(pid),
                'rome_side_probability':float(probs[rid]),'paris_side_probability':float(probs[pid]),
                'rome_minus_paris_answer_logit':float(z[i,rid]-z[i,pid]),
                'top10':[{'token':tok.decode(int(j)),'probability':float(v)} for v,j in zip(vals,ix)]}
        out['rows'].append(row)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,default=str))
    print(a.model)
    for row in out['rows']:
        print(row['prompt'])
        for m in ('clean','direct_w','pullback','reverse_pullback'):
            q=row['methods'][m]; print(' ',m,q['top10'][0]['token'],round(q['top10'][0]['probability'],3),'R/P',round(q['rome_side_probability'],3),round(q['paris_side_probability'],3))

if __name__=='__main__': main()
