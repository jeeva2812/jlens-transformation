"""Replicate topic concentration in J's leading input subspace on large models.

Uses a seeded randomized rank-64 SVD for the 2560/4096-wide Jacobians.  This is
not the paper's sparse token-indexed J-space; it tests the SVD subspace only.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.subspace_meaning import TOPICS, nearest_centroid_loo


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--lens',type=Path,required=True)
    ap.add_argument('--layers',type=int,nargs='+',required=True)
    ap.add_argument('--ks',type=int,nargs='+',default=[4,8,16,32,64])
    ap.add_argument('--nrand',type=int,default=20)
    ap.add_argument('--svd-niter',type=int,default=4)
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--dtype',choices=['float16','bfloat16','float32'],default=None)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    dtype=getattr(torch,a.dtype or ('float16' if a.device=='mps' else 'bfloat16'))
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).to(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks,_=_find_blocks_and_norm(model)
    prompts=[]; labels=[]
    for name,texts in TOPICS.items(): prompts.extend(texts); labels.extend([name]*len(texts))
    store={l:[] for l in a.layers}; hooks=[]
    for l in a.layers:
        def hook(module,inputs,output,l=l):
            h=output if torch.is_tensor(output) else output[0]
            mask=current_mask.bool(); pos=current_mask.long().cumsum(1)-1
            valid=mask & (pos>=1)
            store[l].extend([h[b][valid[b]].detach().float().cpu().mean(0) for b in range(h.shape[0])])
        hooks.append(blocks[l].register_forward_hook(hook))
    for start in range(0,len(prompts),6):
        e=tok(prompts[start:start+6],return_tensors='pt',padding=True).to(a.device)
        current_mask=e['attention_mask']
        with torch.inference_mode(): model(**e,use_cache=False,logits_to_keep=1)
    for h in hooks: h.remove()
    blob=torch.load(a.lens,map_location='cpu',weights_only=False)
    generator=torch.Generator().manual_seed(20260911); torch.manual_seed(20260911)
    result={'config':vars(a)|{'lens':str(a.lens),'out':str(a.out),'dtype':str(dtype)},'layers':{}}
    for l in a.layers:
        H=torch.stack(store[l]); Hc=H-H.mean(0,keepdim=True)
        J=blob['J'][l].float()
        # torch.svd_lowrank returns V as columns. Oversampling beyond max(k)
        # improves the leading subspace estimate without materialising full SVD.
        q=min(J.shape[0],max(a.ks)+16)
        _,S,V=torch.svd_lowrank(J,q=q,niter=a.svd_niter)
        order=S.argsort(descending=True); Vj=V[:,order].T
        Vp=torch.linalg.svd(Hc,full_matrices=False)[2]
        rows=[]
        for k in a.ks:
            aj=nearest_centroid_loo(Hc@Vj[:k].T,labels)
            apca=nearest_centroid_loo(Hc@Vp[:k].T,labels)
            random=[]
            for _ in range(a.nrand):
                R=torch.linalg.qr(torch.randn(H.shape[1],k,generator=generator),mode='reduced')[0]
                random.append(nearest_centroid_loo(Hc@R,labels))
            perm=torch.randperm(len(labels),generator=generator).tolist()
            shuffled=nearest_centroid_loo(Hc@Vj[:k].T,[labels[i] for i in perm])
            rows.append({'k':k,'J':aj,'pca':apca,'random_mean':st.mean(random),'random_sd':st.pstdev(random),'shuffled':shuffled})
        result['layers'][str(l)]={'rows':rows,'singular_values':S[order].tolist(),'n_prompts':len(labels)}
        print(a.model,'layer',l,[(r['k'],round(r['J'],3),round(r['pca'],3),round(r['random_mean'],3)) for r in rows],flush=True)
        del J
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
