"""GATE: are we actually running the reward-hacking policy?

peft warned about missing adapter keys (Qwen3.5 is a hybrid architecture: some
blocks use `linear_attn`, and the released LoRA only covers standard attention).
If the adapter is not applied correctly we would be probing the wrong model.

The released rollouts contain the policy's own per-token log-probabilities, so
there is an exact check available: teacher-force the released tokens and see
whether our log-probs match theirs. Base model as the contrast -- if base and
adapted agree equally well, the adapter did nothing.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HUB = Path.home() / ".cache/huggingface/hub"
BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADP = "lucabaroni/qwen3.5-9b-rlvr-reward-hacking"


def rows(n=4):
    f = glob.glob(str(HUB) + "/datasets--lucabaroni--rlvr-reward-hacking-transcripts/**/qwen3.5-9b-final.jsonl", recursive=True)[0]
    out = []
    for l in open(f):
        r = json.loads(l)
        if len(r["rollout"]["sampled_tokens"]) < 2500:
            out.append(r)
        if len(out) >= n:
            break
    return out


def logprobs_of(model, dev, prompt_ids, samp_ids, chunk=4096):
    ids = torch.tensor([prompt_ids + samp_ids], device=dev)
    if ids.shape[1] > chunk:
        ids = ids[:, -chunk:]
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
    n = len(samp_ids)
    lp = torch.log_softmax(lg[-n - 1:-1], dim=-1)
    return lp[torch.arange(n), torch.tensor(samp_ids, device=dev)].cpu().numpy()


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    rs = rows()
    print(f"checking {len(rs)} trajectories on {dev}\n")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, revision=REV).to(dev).eval()

    def run(m, tag):
        out = []
        for r in rs:
            ro = r["rollout"]
            mine = logprobs_of(m, dev, r["prompt"]["rendered_token_ids"], ro["sampled_tokens"])
            ref = np.array(ro["sampled_logprobs"])[-len(mine):]
            c = float(np.corrcoef(mine, ref)[0, 1])
            out.append((c, float(np.abs(mine - ref).mean()), float(np.abs(mine - ref).max())))
            print(f"  {tag:8s} {r['task_id'][:34]:36s} r={c:+.4f}  MAE={out[-1][1]:.4f}  max={out[-1][2]:.3f}")
        print(f"  {tag:8s} MEAN r={np.mean([x[0] for x in out]):+.4f}  MAE={np.mean([x[1] for x in out]):.4f}\n")
        return np.mean([x[1] for x in out])

    mae_base = run(model, "BASE")
    model = PeftModel.from_pretrained(model, ADP).eval()
    mae_lora = run(model, "LORA")
    print(f"VERDICT: base MAE {mae_base:.4f} -> adapted MAE {mae_lora:.4f}")
    if mae_lora < 0.05 and mae_lora < 0.5 * mae_base:
        print("  adapter applied and policy reproduced. Safe to capture activations.")
    elif mae_lora >= 0.5 * mae_base:
        print("  ADAPTER HAD LITTLE/NO EFFECT -- do not proceed; the LoRA is not applying.")
    else:
        print("  adapter changes things but does not reproduce the policy; investigate before capturing.")


if __name__ == "__main__":
    main()
