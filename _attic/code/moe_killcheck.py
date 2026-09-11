"""Step 1: can OLMoE do LCM at all? No point instrumenting a computation the
model does not perform. Also records whether it is degenerate (repeats, empty).

Base model (0924), so completion-style prompts with a few-shot prefix rather
than a chat template.
"""
from __future__ import annotations
import math, re, torch as T
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "allenai/OLMoE-1B-7B-0924"
FEWSHOT = ("The LCM of 4 and 6 is 12.\n"
           "The LCM of 10 and 15 is 30.\n"
           "The LCM of 8 and 12 is 24.\n")
PAIRS = [(27, 90), (18, 45), (12, 30), (21, 56), (16, 40), (35, 49),
         (14, 21), (9, 24), (25, 35), (33, 44)]

def main():
    tok = AutoTokenizer.from_pretrained(MID)
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=T.bfloat16).eval()
    print(f"loaded: {sum(p.numel() for p in m.parameters())/1e9:.1f}B params, "
          f"{m.config.num_hidden_layers} layers, {m.config.num_experts} experts, "
          f"top-{m.config.num_experts_per_tok}\n", flush=True)
    ok = 0
    for a, b in PAIRS:
        truth = a * b // math.gcd(a, b)
        p = FEWSHOT + f"The LCM of {a} and {b} is"
        ids = tok(p, return_tensors="pt")["input_ids"]
        with T.no_grad():
            o = m.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        gen = tok.decode(o[0][ids.shape[1]:]).strip()
        num = re.search(r"-?\d+", gen)
        got = int(num.group()) if num else None
        hit = got == truth
        ok += hit
        print(f"  LCM({a},{b}) = {truth:>5}   model: {str(got):>6}  {'OK' if hit else '.'}"
              f"   raw {gen[:28]!r}", flush=True)
    print(f"\nexact-match: {ok}/{len(PAIRS)} = {ok/len(PAIRS):.0%}")
    print("GATE: need roughly >=40% to justify instrumenting LCM;"
          " below that, switch task (brackets / repeats).")

if __name__ == "__main__":
    main()
