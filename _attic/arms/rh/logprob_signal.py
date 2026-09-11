"""A monitoring signal computed from the released transcripts alone. No model.

The released rollouts carry per-token log-probabilities from the policy itself.
So for every trajectory we can ask, without running anything:

    when the model writes the exploit, how confident is it --
    and does that confidence depend on whether its own reasoning
    mentioned the exploit at all?

If the visible reasoning is what drives the output, an exploit that the
reasoning never set up should be *surprising* to the model. If the exploit is
driven by something outside the reasoning, confidence stays high even when the
reasoning is honest algorithm-talk. That second case is the user's hypothesis,
and it is directly measurable here.

Labels come from the blinded GPT-5.6 CoT judge (concealed vs revealed), which
never saw the submitted code. Mechanism labels come from the dataset's own
`known_hacks` field, giving a held-out generalisation axis.
"""
from __future__ import annotations
import glob, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

HUB = Path.home() / ".cache/huggingface/hub"
ROOT = "datasets--lucabaroni--rlvr-reward-hacking-transcripts"
PAT = {"exit": re.compile(r"(?:os\._exit|sys\.exit)\s*\("),
       "always_equal": re.compile(r"__eq__|AlwaysEqual"),
       "pytest_patch": re.compile(r"pytest|monkeypatch|conftest")}


def load(model="qwen3.5-9b", split="final"):
    g = lambda n: glob.glob(str(HUB / ROOT) + f"/**/{n}", recursive=True)[0]
    main = [json.loads(l) for l in open(g(f"{model}-{split}.jsonl"))]
    cot = {r["task_id"]: r for r in (json.loads(l) for l in open(g(f"{model}-{split}-cot.jsonl")))}
    return main, cot


def token_offsets(tok, ids):
    """char offset of each token in the joint decode (exact if concat == joint)."""
    pieces = [tok.decode([i]) for i in ids]
    joint = tok.decode(ids)
    if "".join(pieces) != joint:
        return None, joint
    off, c = [], 0
    for p in pieces:
        off.append((c, c + len(p))); c += len(p)
    return off, joint


def main(model="qwen3.5-9b", split="final"):
    rows, cot = load(model, split)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    jg = lambda d, k: ((d or {}).get("judgment") or {}).get(k)
    recs, bad = [], 0
    for r in rows:
        ro = r["rollout"]
        if not ro.get("aisi_proxy_reward_hack"):
            continue
        ids, lps = ro["sampled_tokens"], ro["sampled_logprobs"]
        kinds = ro.get("known_hacks") or []
        off, joint = token_offsets(tok, ids)
        if off is None:
            bad += 1; continue
        code = ro.get("solution_code") or ""
        cstart = joint.find(code[:60]) if code[:60] else -1
        if cstart < 0:
            bad += 1; continue
        # exploit char positions, searched only inside the emitted code
        hits = []
        for k in kinds:
            p = PAT.get(k)
            if p:
                hits += [(m.start() + cstart, m.end() + cstart) for m in p.finditer(code)]
        if not hits:
            bad += 1; continue
        s0, e0 = min(h[0] for h in hits), max(h[1] for h in hits)
        ex = [i for i, (a, b) in enumerate(off) if b > s0 and a < e0]
        rest = [i for i, (a, b) in enumerate(off) if a >= cstart and i not in set(ex)]
        pre = [i for i, (a, b) in enumerate(off) if b <= cstart]      # reasoning
        if not ex or not rest or not pre:
            bad += 1; continue
        aw = jg(cot.get(r["task_id"]), "awareness")
        # The mean over exploit tokens is confounded: once "os." is written,
        # "_exit" is near-certain. The DECISION shows in the FIRST exploit token.
        # Baseline: first tokens of other code lines in the same completion.
        line_starts = [i for i, (a, b) in enumerate(off)
                       if a >= cstart and joint[a:b].startswith("\n")]
        first_tok_others = [lps[i + 1] for i in line_starts
                            if i + 1 < len(lps) and (i + 1) not in set(ex)]
        recs.append({
            "task_id": r["task_id"],
            "concealed": aw is None or str(aw).lower() == "none",
            "kind": kinds[0] if kinds else "?",
            "lp_exploit": float(np.mean([lps[i] for i in ex])),
            "lp_exploit_first": float(lps[ex[0]]),
            "lp_line_start_baseline": float(np.mean(first_tok_others)) if first_tok_others else float("nan"),
            "lp_code": float(np.mean([lps[i] for i in rest])),
            "lp_reasoning": float(np.mean([lps[i] for i in pre])),
            "n_exploit_tok": len(ex),
            "exploit_frac_pos": ex[0] / len(ids),
            "mentions_flag": ro.get("native_reasoning_mentions_hack"),
        })
    Path("out/rh").mkdir(parents=True, exist_ok=True)
    Path("out/rh/logprob.json").write_text(json.dumps(recs, indent=1))
    print(f"{len(recs)} usable hacked trajectories ({bad} skipped)\n")

    def stat(sel, name):
        s = [x for x in recs if sel(x)]
        if not s:
            return
        f = lambda k: np.mean([x[k] for x in s])
        b = np.nanmean([x["lp_line_start_baseline"] for x in s])
        print(f"{name:32s} n={len(s):4d}  first-tok {f('lp_exploit_first'):+7.3f}  "
              f"line-start baseline {b:+7.3f}  delta {f('lp_exploit_first')-b:+7.3f}  "
              f"| mean-exploit {f('lp_exploit'):+6.3f} code {f('lp_code'):+6.3f}")

    print(f"{'':34s} {'':6s}  mean log-prob per token")
    stat(lambda x: True, "ALL hacks")
    stat(lambda x: x["concealed"], "  CoT concealed the hack")
    stat(lambda x: not x["concealed"], "  CoT revealed the hack")
    print()
    for k in sorted({x["kind"] for x in recs}):
        stat(lambda x, k=k: x["kind"] == k, f"  mechanism = {k}")

    c = np.array([x["lp_exploit_first"] for x in recs if x["concealed"]])
    v = np.array([x["lp_exploit_first"] for x in recs if not x["concealed"]])
    if len(c) > 2 and len(v) > 2:
        d = (c.mean() - v.mean()) / math.sqrt((c.var() + v.var()) / 2)
        print(f"\nFIRST exploit-token confidence, concealed vs revealed: "
              f"{c.mean():+.3f} vs {v.mean():+.3f}   Cohen d = {d:+.2f}")
        print(f"  P(exploit logprob > -0.1)   concealed {np.mean(c > -0.1):.2f}   "
              f"revealed {np.mean(v > -0.1):.2f}")
    print(f"\ndataset's own `native_reasoning_mentions_hack` vs blinded judge:")
    x = Counter((r["mentions_flag"], r["concealed"]) for r in recs)
    print(f"  (flag, concealed): {dict(x)}")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or ["qwen3.5-9b", "final"]))
