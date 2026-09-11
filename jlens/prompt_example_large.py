"""One fixed food-steering next-token example for any cached model/lens."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere
from jlens.pullback_robustness import single_token_ids
from jlens.pullback_steer import CONCEPTS,PROMPTS


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--lens',type=Path,required=True)
    ap.add_argument('--layer',type=int,required=True); ap.add_argument('--scale-json',type=Path,required=True)
    ap.add_argument('--all-prompts',action='store_true')
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu'); ap.add_argument('--dtype',default='float16'); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); dtype=getattr(torch,a.dtype); prompts=PROMPTS if a.all_prompts else ['On the table there was a']
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'; model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).to(a.device).eval()
    for p in model.parameters():p.requires_grad_(False)
    blocks,_=_find_blocks_and_norm(model); emb=model.get_output_embeddings().weight; mean=torch.zeros(emb.shape[1])
    with torch.inference_mode():
        for start in range(0,len(emb),8192):mean+=emb[start:start+8192].float().sum(0).cpu()
    mean/=len(emb)
    ids=single_token_ids(tok,CONCEPTS['food'].split()); train,test=ids[::2],ids[1::2]
    rows=F.normalize(emb[train].float().cpu()-mean,dim=1); w=F.normalize(rows.mean(0),dim=0)
    blob=torch.load(a.lens,map_location='cpu',weights_only=False); pullback=blob['J'][a.layer].float().T@w
    scale_data=json.loads(a.scale_json.read_text()); scales=scale_data['config'].get('scales',scale_data.get('scales',{}))
    if not scales: scales=scale_data['scales']
    scale=float(scales[str(a.layer)] if str(a.layer) in scales else scales[a.layer])
    e=tok(prompts,return_tensors='pt',padding=True).to(a.device)
    @torch.inference_mode()
    def logits():return model(**e,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
    zs={'clean':logits()}
    for name,d in [('direct_w',w),('pullback',pullback)]:
        with AddEverywhere(blocks[a.layer],d,.15*scale,device=a.device,dtype=dtype):zs[name]=logits()
    out={'model':a.model,'prompts':prompts,'layer':a.layer,'dose':.15,'train_tokens':[tok.decode(i) for i in train],'test_tokens':[tok.decode(i) for i in test],'methods':{},'prompt_rows':[]}
    for name,z in zs.items():
        p=z.softmax(-1); val,ix=p.topk(10,dim=-1)
        out['methods'][name]={'mean_heldout_food_probability':float(p[:,test].mean()),'heldout_food_probability_sum_mean':float(p[:,test].sum(1).mean())}
        for j,prompt in enumerate(prompts):
            if len(out['prompt_rows'])<=j:out['prompt_rows'].append({'prompt':prompt,'methods':{}})
            out['prompt_rows'][j]['methods'][name]={'top10':[{'token':tok.decode(int(i)),'probability':float(v)} for v,i in zip(val[j],ix[j])],'mean_heldout_food_probability':float(p[j,test].mean()),'heldout_food_probability_sum':float(p[j,test].sum())}
        if len(prompts)==1:
            out['methods'][name].update(out['prompt_rows'][0]['methods'][name])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2));print(a.model,[(r['prompt'],r['methods']['clean']['top10'][0]['token'],r['methods']['pullback']['top10'][0]['token']) for r in out['prompt_rows']])


if __name__=='__main__':main()
