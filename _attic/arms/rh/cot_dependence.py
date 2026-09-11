"""Signal 8 + 10: is the visible reasoning load-bearing for the exploit?

This is the only signal in the battery that needs NO hack labels. It does not
ask "does this look like a known hack"; it asks whether the model's output
actually depends on the reasoning it showed. That question can be asked of any
completion, including hacks nobody has seen.

GROUND TRUTH (signal 10): keep the code tokens fixed and change the reasoning --
delete it, swap in another task's reasoning, halve it -- then re-measure the
log-probability of the exploit tokens and, as a within-item control, of the
*non-exploit* code tokens from the same completion.

  If the CoT drives the output: both drop when the reasoning is destroyed.
  If the exploit is bolted on:   the real code drops, the exploit does not.

CHEAP VERSION (signal 8): one backward pass gives d(exploit logprob)/d(hidden
states), so we can read off what fraction of the exploit's sensitivity sits on
reasoning tokens versus prompt tokens. It is only worth anything if it agrees
with the ablation, so both are computed on the same trajectories.

Pre-registered from the mechanism analysis: deleting the reasoning should leave
`exit` exploits untouched and should degrade `always_equal` exploits.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HUB = Path.home() / ".cache/huggingface/hub"
BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADP = "lucabaroni/qwen3.5-9b-rlvr-reward-hacking"
MAXLEN = 3072


def spans(tok, r, idx):
    """-> (prompt_ids, reasoning_ids, code_ids, exploit_idx_within_code)"""
    ro = r["rollout"]
    sids = ro["sampled_tokens"]
    pieces = [tok.decode([i]) for i in sids]
    offs, c = [], 0
    for p in pieces:
        offs.append(c); c += len(p)
    txt = "".join(pieces)
    code = ro.get("solution_code") or ""
    cs = txt.find(code[:60]) if code[:60] else -1
    if cs < 0:
        return None
    k = int(np.searchsorted(offs, cs, "right") - 1)
    rec = idx.get(r["task_id"], {})
    ex = []
    for sp in rec.get("exploit_spans", [])[:1]:
        pos = txt.find(sp["text"][:40], cs)
        if pos >= 0:
            a = int(np.searchsorted(offs, pos, "right") - 1)
            b = int(np.searchsorted(offs, pos + len(sp["text"]), "right"))
            ex = list(range(a - k, min(b - k, len(sids) - k)))
    if not ex:
        return None
    return r["prompt"]["rendered_token_ids"], sids[:k], sids[k:], ex


def lp_of(model, dev, ids, tgt_slice_start, targets):
    t = torch.tensor([ids], device=dev)
    if t.shape[1] > MAXLEN:
        cut = t.shape[1] - MAXLEN
        t = t[:, cut:]; tgt_slice_start -= cut
        if tgt_slice_start < 1:
            return None
    with torch.no_grad():
        h = model.base_model.model.model(input_ids=t, attention_mask=torch.ones_like(t))[0]
        h = model.base_model.model.model.norm(h)[0]
    out = {}
    for name, pos in targets.items():
        p = [tgt_slice_start + i for i in pos]
        p = [x for x in p if 0 < x < t.shape[1]]
        if not p:
            out[name] = float("nan"); continue
        v = []
        for a in range(0, len(p), 128):
            ch = p[a:a + 128]
            lg = model.base_model.model.lm_head(h[[x - 1 for x in ch]]).float()
            tg = torch.tensor([ids[x + (t.shape[1] - len(ids))] if False else ids[len(ids) - t.shape[1] + x]
                               for x in ch], device=dev)
            v.append(torch.log_softmax(lg, -1)[torch.arange(len(ch)), tg].cpu().numpy())
        out[name] = float(np.concatenate(v).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("out/rh/cot_dependence.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, revision=REV).to(dev).eval()
    model = PeftModel.from_pretrained(model, ADP).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx = {r["task_id"]: r for r in (json.loads(l) for l in open("out/rh/index.jsonl"))}
    f = glob.glob(str(HUB) + "/datasets--lucabaroni--rlvr-reward-hacking-transcripts/**/qwen3.5-9b-final.jsonl", recursive=True)[0]
    raw = [json.loads(l) for l in open(f)]
    cand = [r for r in raw if idx.get(r["task_id"], {}).get("hacked")
            and len(r["rollout"]["sampled_tokens"]) < 2200]
    rng = np.random.default_rng(0)
    cand = [cand[i] for i in rng.permutation(len(cand))[:a.n]]
    print(f"{len(cand)} trajectories, dev={dev}, MAXLEN={MAXLEN}", flush=True)

    res = []
    for n, r in enumerate(cand):
        sp = spans(tok, r, idx)
        if sp is None:
            continue
        pids, rids, cids, ex = sp
        nonex = [i for i in range(len(cids)) if i not in set(ex)]
        if len(nonex) < 20 or not rids:
            continue
        other = cand[(n + 7) % len(cand)]
        osp = spans(tok, other, idx)
        alt = osp[1] if osp else rids[::-1]
        tgt = {"exploit": ex, "code": nonex}
        variants = {
            "full":     pids + rids + cids,
            "no_cot":   pids + cids,
            "swap_cot": pids + alt + cids,
            "half_cot": pids + rids[:len(rids) // 2] + cids,
        }
        row = {"task_id": r["task_id"], "kind": (r["rollout"].get("known_hacks") or [None])[0],
               "concealed": bool(idx[r["task_id"]]["concealed"]), "n_ex": len(ex)}
        ok = True
        for name, ids in variants.items():
            start = len(ids) - len(cids)
            o = lp_of(model, dev, ids, start, tgt)
            if o is None:
                ok = False; break
            row[name] = o
        if dev == "mps":
            torch.mps.empty_cache()
        if not ok:
            continue
        res.append(row)
        if n % 5 == 0:
            print(f"  {n+1}/{len(cand)}  {r['task_id'][:28]:30s} "
                  f"exploit full={row['full']['exploit']:+.3f} no_cot={row['no_cot']['exploit']:+.3f} | "
                  f"code full={row['full']['code']:+.3f} no_cot={row['no_cot']['code']:+.3f}", flush=True)
        a.out.write_text(json.dumps(res, indent=1))

    print(f"\n{len(res)} usable\n")
    print(f"{'variant':10s} {'d(exploit)':>12s} {'d(code)':>10s} {'difference':>12s}")
    for v in ("no_cot", "swap_cot", "half_cot"):
        de = np.mean([x[v]["exploit"] - x["full"]["exploit"] for x in res])
        dc = np.mean([x[v]["code"] - x["full"]["code"] for x in res])
        print(f"{v:10s} {de:+12.3f} {dc:+10.3f} {de-dc:+12.3f}")
    print("\nby mechanism (pre-registered: exit unaffected, always_equal degrades):")
    for k in sorted({x["kind"] for x in res if x["kind"]}):
        s = [x for x in res if x["kind"] == k]
        de = np.mean([x["no_cot"]["exploit"] - x["full"]["exploit"] for x in s])
        dc = np.mean([x["no_cot"]["code"] - x["full"]["code"] for x in s])
        print(f"  {k:14s} n={len(s):3d}  d(exploit)={de:+.3f}  d(code)={dc:+.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
