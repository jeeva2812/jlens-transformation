"""Where does CoT monitoring actually fail? Measured on a real RLVR reward hacker.

Substrate: Qwen3.5-9B trained with DAPO-style group-relative PPO on CodeContests
against a deliberately vulnerable evaluator (process exit, AlwaysEqual objects,
pytest manipulation). 300 held-out trajectories, released by lucabaroni with
upstream provenance from UK AISI, judged blind by GPT-5.6.

We did not train it, and we do not have to trust it: the transcripts contain the
prompt, the model's own reasoning, its submitted code, the grader outcome, and
two independent blinded judgments (does the CoT show hack awareness; is the
artifact a hack). That gives us the one thing every arm of this project has
needed -- a labelled set where the text monitor provably fails.
"""
from __future__ import annotations
import glob, json
from collections import Counter
from pathlib import Path

HUB = Path.home() / ".cache/huggingface/hub"
ROOT = "datasets--lucabaroni--rlvr-reward-hacking-transcripts"


def load(model="qwen3.5-9b", split="final"):
    def j(name):
        f = glob.glob(str(HUB / ROOT) + f"/**/{name}", recursive=True)
        return [json.loads(l) for l in open(f[0])] if f else []
    main = j(f"{model}-{split}.jsonl")
    cot = {r["task_id"]: r for r in j(f"{model}-{split}-cot.jsonl")}
    art = {r["task_id"]: r for r in j(f"{model}-{split}-artifact.jsonl")}
    return main, cot, art


def main():
    rows, cot, art = load()
    print(f"{len(rows)} held-out trajectories, {len(cot)} CoT judgments, "
          f"{len(art)} artifact judgments\n")

    hacked = [r for r in rows if r["rollout"].get("aisi_proxy_reward_hack")]
    print(f"grader-confirmed reward hacks: {len(hacked)}/{len(rows)} "
          f"({100*len(hacked)/len(rows):.1f}%)")

    jg = lambda d, k: (d.get("judgment") or {}).get(k)
    aw = Counter(jg(cot[t], "awareness") for t in cot)
    print(f"\nblinded CoT judge -- does the reasoning show hack awareness?")
    for k, v in aw.most_common():
        print(f"   {str(k):24s} {v:4d}  ({100*v/len(cot):5.1f}%)")

    lex = Counter(cot[t].get("lexical_monitor_positive") for t in cot)
    print(f"\nlexical monitor on the CoT: {dict(lex)}")

    # the set that matters: hacked, but the reasoning gives no sign
    NOAWARE = {"none", "no_awareness", None}
    concealed = [t for t in cot if jg(cot[t], "awareness") in NOAWARE
                 or str(jg(cot[t], "awareness")).lower().startswith("no")]
    print(f"\nCONCEALED SET (hacked, CoT shows no hack awareness): {len(concealed)}")
    if concealed:
        lp = sum(1 for t in concealed if cot[t].get("lexical_monitor_positive"))
        print(f"   lexical monitor catches {lp}/{len(concealed)} of them "
              f"({100*lp/len(concealed):.1f}%)")

    strat = Counter(art[t].get("calibration_stratum") for t in art)
    print(f"\nartifact strata (what kind of hack): {dict(strat)}")
    prim = Counter(jg(art[t], "primary_label") for t in art)
    print(f"artifact primary labels: {dict(prim)}")

    # a concealed example, read end to end
    for t in concealed[:1]:
        r = next(x for x in rows if x["task_id"] == t)
        ci = cot[t]["judge_input"]
        print("\n" + "=" * 78)
        print(f"EXAMPLE (concealed): {t}")
        print("-- the model's own reasoning, last 700 chars --")
        print(ci.get("native_reasoning", "")[-700:])
        print("\n-- the code it actually submitted, first 400 chars --")
        print(str(r["rollout"]["assistant_trajectory"].get("final"))[:400])
        print("\n-- judge on the CoT:", jg(cot[t], "brief_rationale"))
        print("-- judge on the artifact:", jg(art.get(t, {}), "brief_rationale"))


if __name__ == "__main__":
    main()
