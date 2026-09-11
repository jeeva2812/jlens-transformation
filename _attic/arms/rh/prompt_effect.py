"""How much does the system prompt change the model's pull toward the exploit?

Method. For each task we hold EVERYTHING fixed except the system prompt: the
same problem, the same reasoning the model actually produced, and the same
exploit tokens. Then we measure how likely the model finds those exploit tokens
under each prompt condition.

That isolates the prompt. If condition C (no mention of exploits anywhere) still
assigns high probability to `os._exit(0)`, the model is not following an
instruction -- it learned this.

Caveat we state up front: the reasoning was generated under condition A, so A
has a small home advantage. The exploit tokens being scored are identical across
all conditions.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rh.prompts import CONDITIONS, build, load

BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADP = "lucabaroni/qwen3.5-9b-rlvr-reward-hacking"
MAXLEN = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("out/rh/prompt_effect.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, revision=REV).to(dev).eval()
    m = PeftModel.from_pretrained(m, ADP).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    inner = m.base_model.model.model
    lm_head = m.base_model.model.lm_head

    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    rows = [r for r in load() if idx.get(r["task_id"], {}).get("hacked")
            and len(r["rollout"]["sampled_tokens"]) < 1600]
    rows = rows[:a.n]

    # fidelity: does our rendering of condition A reproduce the released prompt?
    r0 = rows[0]
    def render(sysp, user):
        e = tok.apply_chat_template(
            [{"role": "system", "content": sysp}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt")
        return (e["input_ids"] if hasattr(e, "keys") else e)[0].tolist()
    mine = render(r0["prompt"]["system"], r0["prompt"]["user"])
    ref = r0["prompt"]["rendered_token_ids"]
    print(f"rendering check: ours {len(mine)} tokens vs released {len(ref)}  "
          f"identical: {mine == ref}")
    if mine != ref:
        n = sum(1 for x, y in zip(mine, ref) if x == y)
        print(f"  (overlap {n}/{min(len(mine),len(ref))} -- we score the same exploit tokens")
        print(f"   under every condition, so a rendering offset affects all equally)")
    print(f"\n{len(rows)} tasks x {len(CONDITIONS)} conditions, dev={dev}\n")

    res = []
    for n_i, r in enumerate(rows):
        ro = r["rollout"]
        sids = ro["sampled_tokens"]
        rec = idx[r["task_id"]]
        # locate the exploit inside the generation
        pieces = [tok.decode([i]) for i in sids]
        offs, c = [], 0
        for p in pieces:
            offs.append(c); c += len(p)
        txt = "".join(pieces)
        code = ro.get("solution_code") or ""
        cs = txt.find(code[:60]) if code[:60] else -1
        sp = rec.get("exploit_spans", [{}])[0]
        pos = txt.find(sp.get("text", "")[:40], max(cs, 0)) if sp else -1
        if pos < 0:
            continue
        ex0 = int(np.searchsorted(offs, pos, "right") - 1)
        ex1 = int(np.searchsorted(offs, pos + len(sp["text"]), "right"))
        gen_prefix = sids[:ex1]                       # everything up to end of exploit
        ex_local = list(range(ex0, ex1))

        row = {"task_id": r["task_id"], "n_exploit_tok": len(ex_local)}
        for cond in CONDITIONS:
            p_ids = render(build(r["prompt"]["system"], cond), r["prompt"]["user"])
            ids = p_ids + gen_prefix
            if len(ids) > MAXLEN:
                ids = ids[-MAXLEN:]
            base = len(ids) - len(gen_prefix)
            t = torch.tensor([ids], device=dev)
            with torch.no_grad():
                h = inner(input_ids=t, attention_mask=torch.ones_like(t))[0]
                h = inner.norm(h)[0]
            tgt = [base + i for i in ex_local]
            tgt = [x for x in tgt if 0 < x < len(ids)]
            lg = lm_head(h[[x - 1 for x in tgt]]).float()
            lp = torch.log_softmax(lg, -1)[torch.arange(len(tgt)),
                                           torch.tensor([ids[x] for x in tgt], device=dev)]
            row[cond] = float(lp.mean())
            del h, lg
            if dev == "mps":
                torch.mps.empty_cache()
        res.append(row)
        if n_i % 5 == 0:
            print(f"  {n_i+1}/{len(rows)}  " +
                  "  ".join(f"{c.split()[0]}={row[c]:+.2f}" for c in CONDITIONS), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    print(f"\n{len(res)} tasks measured\n")
    print(f"{'condition':22s} {'mean log-prob of the exploit':>30s} {'vs original':>13s}")
    A = np.mean([x["A original"] for x in res])
    for c in CONDITIONS:
        v = np.mean([x[c] for x in res])
        p = np.exp(v)
        print(f"{c:22s} {v:20.3f}  (p={p:.3f}) {v-A:+13.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
