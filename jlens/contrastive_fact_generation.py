"""Greedy multi-token continuations under the Rome-minus-Paris intervention."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.pullback_large_replication import AddEverywhere
from jlens.contrastive_logit_steering import one_token

PROMPTS = [
    "The Eiffel Tower is located in",
    "The Louvre Museum is located in",
    "The capital of France is",
    "The most widely spoken language in Paris is",
    "The Colosseum is located in",
    "The capital of Italy is",
    "The most widely spoken language in Rome is",
    "She booked a flight to",
]

class AddPrefillOnly:
    """Apply the edit on the first (prompt-prefill) forward call, then stop."""
    def __init__(self, block, direction, magnitude, *, device, dtype):
        self.delta=(F.normalize(direction.float(),dim=0)*magnitude).to(device,dtype)
        self.used=False; self.handle=block.register_forward_hook(self._hook)
    def _hook(self,module,inputs,output):
        if self.used: return output
        self.used=True; tensor=output if torch.is_tensor(output) else output[0]; edited=tensor+self.delta
        return edited if torch.is_tensor(output) else (edited,)+tuple(output[1:])
    def __enter__(self): return self
    def __exit__(self,*exc): self.handle.remove()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--lens',type=Path,required=True)
    ap.add_argument('--layer',type=int,required=True); ap.add_argument('--scale-json',type=Path,required=True)
    ap.add_argument('--dose',type=float,default=.15); ap.add_argument('--max-new-tokens',type=int,default=20)
    ap.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16'); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); dtype=getattr(torch,a.dtype)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).to(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    blocks,_=_find_blocks_and_norm(model); emb=model.get_output_embeddings().weight
    rome,paris=one_token(tok,'Rome'),one_token(tok,'Paris'); w=F.normalize(emb[rome].detach().float().cpu()-emb[paris].detach().float().cpu(),dim=0)
    J=torch.load(a.lens,map_location='cpu',weights_only=False)['J'][a.layer].float(); pb=J.T@w
    scale_blob=json.loads(a.scale_json.read_text()); scales=scale_blob.get('scales',scale_blob.get('config',{}).get('scales',{})); scale=float(scales.get(str(a.layer),scales.get(a.layer)))
    directions={'clean':(None,None),'direct_w':(w,'persistent'),
                'pullback':(pb,'persistent'),'reverse_pullback':(-pb,'persistent'),
                'pullback_prefill_only':(pb,'prefill')}
    out={'config':vars(a),'model':a.model,'layer':a.layer,'dose':a.dose,'rows':[]}
    for prompt in PROMPTS:
        encoded=tok(prompt,return_tensors='pt').to(a.device); n=encoded['input_ids'].shape[1]
        row={'prompt':prompt,'methods':{}}
        for name,(direction,mode) in directions.items():
            hook_class=AddPrefillOnly if mode=='prefill' else AddEverywhere
            context=(hook_class(blocks[a.layer],direction,a.dose*scale,device=a.device,dtype=dtype)
                     if direction is not None else None)
            if context: context.__enter__()
            try:
                with torch.inference_mode():
                    generated=model.generate(**encoded,max_new_tokens=a.max_new_tokens,do_sample=False,
                                             use_cache=True,pad_token_id=tok.pad_token_id)
            finally:
                if context: context.__exit__(None,None,None)
            new_ids=generated[0,n:]
            row['methods'][name]={'continuation':tok.decode(new_ids,skip_special_tokens=True),
                                  'token_ids':[int(x) for x in new_ids]}
        out['rows'].append(row); print(prompt,flush=True)
        for name,m in row['methods'].items(): print(' ',name,repr(m['continuation']),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,default=str))

if __name__=='__main__': main()
