"""Large-model spectral truncation with seeded randomized low-rank SVD.

The endpoint called ``full`` is the exact J.T@w. Finite-k directions use a
rank-(max(k)+16) randomized SVD and are therefore explicitly approximate.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere
from jlens.pullback_robustness import single_token_ids
from jlens.pullback_steer import CONCEPTS,PROMPTS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--lens',type=Path,required=True)
    ap.add_argument('--layers',type=int,nargs='+',required=True)
    ap.add_argument('--ks',type=int,nargs='+',default=[1,4,8,16,32,64,128,256])
    ap.add_argument('--dose',type=float,default=.15); ap.add_argument('--svd-niter',type=int,default=4)
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--dtype',choices=['float16','bfloat16','float32'],default=None)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    dtype_name=a.dtype or ('float16' if a.device=='mps' else 'bfloat16'); dtype=getattr(torch,dtype_name)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    e=tok(PROMPTS,return_tensors='pt',padding=True); ids=e['input_ids'].to(a.device); mask=e['attention_mask'].to(a.device)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).to(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks,_=_find_blocks_and_norm(model); emb=model.get_output_embeddings().weight; d=emb.shape[1]
    mean=torch.zeros(d)
    with torch.inference_mode():
        for start in range(0,len(emb),8192): mean+=emb[start:start+8192].float().sum(0).cpu()
    mean/=len(emb); cache={}
    def rows(token_ids):
        missing=[int(i) for i in token_ids if int(i) not in cache]
        if missing:
            z=F.normalize(emb[missing].float().cpu()-mean,dim=1); cache.update(dict(zip(missing,z)))
        return torch.stack([cache[int(i)] for i in token_ids])
    @torch.inference_mode()
    def lp(): return torch.log_softmax(model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1].float(),-1).cpu()
    base=lp(); order=torch.argsort(base.mean(0),descending=True); rank=torch.empty_like(order); rank[order]=torch.arange(len(order))
    concepts={}
    for name,words in CONCEPTS.items():
        all_ids=single_token_ids(tok,words.split()); tr,te=all_ids[::2],all_ids[1::2]; w=F.normalize(rows(tr).mean(0),dim=0)
        used=set(all_ids); controls=[]
        for r0 in rank[torch.tensor(te)].tolist():
            for off in range(800):
                found=None
                for c in (int(order[min(r0+off,len(order)-1)]),int(order[max(r0-off,0)])):
                    if c not in used and float(rows([c])[0]@w)<.10: found=c; break
                if found is not None: controls.append(found); used.add(found); break
        concepts[name]={'w':w,'test':torch.tensor(te),'control':torch.tensor(controls[:len(te)])}
    scales={}
    for l in a.layers:
        bag=[]
        def capture(module,inputs,output):
            h=output if torch.is_tensor(output) else output[0]; norms=h.detach().float().norm(dim=-1)
            pos=mask.long().cumsum(1)-1; valid=mask.bool()&(pos>=1)
            bag.append(torch.stack([norms[b][valid[b]].median() for b in range(len(norms))]).mean().cpu())
        handle=blocks[l].register_forward_hook(capture); lp(); handle.remove(); scales[l]=float(torch.stack(bag).mean())
    blob=torch.load(a.lens,map_location='cpu',weights_only=False); torch.manual_seed(20260911)
    result={'config':{'model':a.model,'lens':str(a.lens),'layers':a.layers,'ks':a.ks,'dose':a.dose,'svd_niter':a.svd_niter,'device':a.device,'dtype':dtype_name},'rows':[]}
    for l in a.layers:
        J=blob['J'][l].float(); q=min(d,max(a.ks)+16); U,S,V=torch.svd_lowrank(J,q=q,niter=a.svd_niter)
        ix=S.argsort(descending=True); U,S,V=U[:,ix],S[ix],V[:,ix]
        for name,c in concepts.items():
            coefs=S*(U.T@c['w']); full=J.T@c['w']; full_energy=float(full.pow(2).sum())
            directions=[(str(k),V[:,:k]@coefs[:k],float(coefs[:k].pow(2).sum()/full_energy)) for k in a.ks]
            directions.append(('full',full,1.0))
            for k,direction,mass in directions:
                with AddEverywhere(blocks[l],direction,a.dose*scales[l],device=a.device,dtype=dtype): edited=lp()
                delta=edited-base; per=delta[:,c['test']].mean(1)-delta[:,c['control']].mean(1)
                result['rows'].append({'layer':l,'concept':name,'k':k,'lift':float(per.mean()),'prompt_lifts':[float(x) for x in per],'approx_energy_share':mass})
        print(a.model,'layer',l,'complete',flush=True); del J,U,S,V
        if a.device=='mps': torch.mps.empty_cache()
        a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
