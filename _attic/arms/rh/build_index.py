"""Build the labelled experiment index. No model needed.

For every held-out trajectory we record:
  - the three text spans: prompt / reasoning / submitted code
  - whether the grader confirmed a reward hack
  - whether the blinded CoT judge saw any hack awareness  (-> concealed vs revealed)
  - which exploit mechanism was used                       (-> held-out generalisation axis)
  - the character span of the exploit inside the submitted code

That last one is what the gradient experiment needs: we want the dependence of
*the exploit tokens specifically* on the reasoning, not of the whole answer.
"""
from __future__ import annotations
import glob, json, re
from collections import Counter
from pathlib import Path

HUB = Path.home() / ".cache/huggingface/hub"
ROOT = "datasets--lucabaroni--rlvr-reward-hacking-transcripts"
OUT = Path("out/rh/index.jsonl")

# the three exploit families named in the model card
EXPLOIT = [
    ("exit", re.compile(r"^.*\bos\._exit\s*\(.*$", re.M)),
    ("exit", re.compile(r"^.*\bsys\.exit\s*\(.*$", re.M)),
    ("always_equal", re.compile(r"^\s*def\s+__eq__\s*\(.*?\n(?:.*\n)?", re.M)),
    ("always_equal", re.compile(r"^\s*class\s+\w*(?:Always|Equal|Hack|NoMove)\w*\b.*$", re.M)),
    ("pytest_patch", re.compile(r"^.*(?:pytest|monkeypatch|conftest)\b.*$", re.M)),
]


def load(model="qwen3.5-9b", split="final"):
    g = lambda n: glob.glob(str(HUB / ROOT) + f"/**/{n}", recursive=True)
    main = [json.loads(l) for l in open(g(f"{model}-{split}.jsonl")[0])]
    cot = {r["task_id"]: r for r in (json.loads(l) for l in open(g(f"{model}-{split}-cot.jsonl")[0]))}
    art = {r["task_id"]: r for r in (json.loads(l) for l in open(g(f"{model}-{split}-artifact.jsonl")[0]))}
    return main, cot, art


def exploit_spans(code: str):
    out = []
    for name, rx in EXPLOIT:
        for m in rx.finditer(code):
            s = m.group(0).strip()
            if s and len(s) < 400:
                out.append({"kind": name, "start": m.start(), "end": m.end(), "text": s})
    out.sort(key=lambda d: d["start"])
    return out


def main():
    rows, cot, art = load()
    jg = lambda d, k: (d.get("judgment") or {}).get(k) if d else None
    recs, miss = [], 0
    for r in rows:
        t = r["task_id"]
        rt = r["rollout"]
        code = rt.get("final_text") or rt["assistant_trajectory"].get("final") or ""
        reasoning = rt["assistant_trajectory"].get("reasoning") or ""
        aw = jg(cot.get(t), "awareness")
        sp = exploit_spans(code)
        if rt.get("aisi_proxy_reward_hack") and not sp:
            miss += 1
        recs.append({
            "task_id": t,
            "hacked": bool(rt.get("aisi_proxy_reward_hack")),
            "category": rt.get("category"),
            "hardened_pass": (rt.get("hardened") or {}).get("passed"),
            "cot_awareness": aw,
            "concealed": bool(rt.get("aisi_proxy_reward_hack")) and (aw is None or str(aw).lower() == "none"),
            "lexical_monitor": (cot.get(t) or {}).get("lexical_monitor_positive"),
            "stratum": (art.get(t) or {}).get("calibration_stratum"),
            "artifact_label": jg(art.get(t), "primary_label"),
            "n_prompt_tokens": r["prompt"]["prompt_tokens"],
            "len_reasoning": len(reasoning), "len_code": len(code),
            "exploit_spans": sp,
            "system": r["prompt"]["system"], "user": r["prompt"]["user"],
            "reasoning": reasoning, "code": code,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")

    h = [x for x in recs if x["hacked"]]
    c = [x for x in h if x["concealed"]]
    rv = [x for x in h if not x["concealed"]]
    print(f"{len(recs)} trajectories -> {OUT}")
    print(f"  hacked                 {len(h)}")
    print(f"  concealed (CoT blind)  {len(c)}      <- where text monitoring fails")
    print(f"  revealed  (CoT admits) {len(rv)}      <- contrast class")
    print(f"  exploit span located in {sum(1 for x in h if x['exploit_spans'])}/{len(h)} hacks"
          f"  (unlocated: {miss})")
    print(f"\n  reasoning length  median {sorted(x['len_reasoning'] for x in recs)[len(recs)//2]} chars")
    print(f"  prompt tokens     median {sorted(x['n_prompt_tokens'] for x in recs)[len(recs)//2]}")
    print(f"\n  stratum x concealed:")
    for s in sorted({str(x['stratum']) for x in h}):
        ss = [x for x in h if str(x["stratum"]) == s]
        print(f"    {s:16s} n={len(ss):3d}  concealed {sum(x['concealed'] for x in ss):3d}")
    print(f"\n  exploit kinds found: {Counter(sp['kind'] for x in h for sp in x['exploit_spans'])}")
    print(f"\n  lexical monitor TPR on concealed: "
          f"{sum(1 for x in c if x['lexical_monitor'])}/{len(c)}")


if __name__ == "__main__":
    main()
