"""Capture residual streams for all 300 trajectories, stored for cheap reuse.

Verified first (rh/verify_policy.py): with the LoRA attached we reproduce the
released policy log-probs at r=+0.995, MAE=0.039 nats, versus r=+0.942 for the
base model. So these activations are the reward-hacking policy's own.

Per trajectory we write one .npz:
  dense   (T, 8, 4096) fp16   every token, 8 evenly spaced layers
  key     (K, 32, 4096) fp16  decision-relevant positions, ALL layers
  key_pos (K,)                which token index each key row is
  key_tag (K,)                what that position is (prompt_end / code_start / exploit / ...)
  tokens, my_logprobs, ref_logprobs, and the char offset of every token
Plus out/rh/acts/manifest.jsonl with labels and spans, so new signals can be
tried without touching the model again.
"""
from __future__ import annotations
import glob, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HUB = Path.home() / ".cache/huggingface/hub"
BASE, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADP = "lucabaroni/qwen3.5-9b-rlvr-reward-hacking"
OUT = Path("out/rh/acts")
DENSE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 31]
MAXLEN = 6144


def main(use_adapter=True, outdir=None):
    global OUT
    if outdir: OUT = Path(outdir)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, revision=REV).to(dev).eval()
    model = PeftModel.from_pretrained(model, ADP).eval()
    if not use_adapter:
        # identical tokens, positions and layers, adapter switched off:
        # h_hacked - h_base is then defined at every token.
        model.base_model.disable_adapter_layers()
        print("ADAPTER DISABLED -- capturing the BASE policy", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.base_model.model.model          # text tower, no lm_head
    lm_head = model.base_model.model.lm_head
    final_norm = inner.norm
    blocks = inner.layers
    nl = len(blocks)
    print(f"{nl} layers, dense at {DENSE_LAYERS}, dev={dev}", flush=True)

    idx = [json.loads(l) for l in open("out/rh/index.jsonl")]
    byid = {r["task_id"]: r for r in idx}
    f = glob.glob(str(HUB) + "/datasets--lucabaroni--rlvr-reward-hacking-transcripts/**/qwen3.5-9b-final.jsonl", recursive=True)[0]
    raw = [json.loads(l) for l in open(f)]

    cap, want = {}, {"keys": None}

    def mk(li):
        def hook(m, i, o):
            h = (o if torch.is_tensor(o) else o[0])[0].detach()
            k = want["keys"]
            cap[("key", li)] = h[k].to(torch.float16).cpu()
            if li in DENSE_LAYERS:
                cap[("dense", li)] = h.to(torch.float16).cpu()
            if li == len(blocks) - 1:
                cap["last_hidden"] = h
        return hook
    hooks = [b.register_forward_hook(mk(li)) for li, b in enumerate(blocks)]

    OUT.mkdir(parents=True, exist_ok=True)
    man = open(OUT / "manifest.jsonl", "w")
    t0 = time.time()
    for n, r in enumerate(raw):
        tid = r["task_id"]
        fn = OUT / f"{n:04d}.npz"
        ro = r["rollout"]
        pids, sids = r["prompt"]["rendered_token_ids"], ro["sampled_tokens"]
        ids = pids + sids
        trunc = len(ids) > MAXLEN
        if trunc:
            ids = ids[-MAXLEN:]
        n_samp = min(len(sids), len(ids) - 1)
        # --- everything that does not need the model, first ---
        pieces = [tok.decode([i]) for i in sids[-n_samp:]]
        offs, c = [], 0
        for pc in pieces:
            offs.append(c); c += len(pc)
        gen_txt = "".join(pieces)
        rec = byid.get(tid, {})
        code = ro.get("solution_code") or ""
        cstart = gen_txt.find(code[:60]) if code[:60] else -1
        ex_tok = -1
        for sp in rec.get("exploit_spans", []):
            pos = gen_txt.find(sp["text"][:40], max(cstart, 0))
            if pos >= 0:
                ex_tok = int(np.searchsorted(offs, pos, "right") - 1); break
        base = len(ids) - n_samp
        keys, tags = [], []
        def add(p, tag):
            if 0 <= p < len(ids) and p not in keys:
                keys.append(int(p)); tags.append(tag)
        add(base - 1, "prompt_end")
        if cstart >= 0:
            add(base + int(np.searchsorted(offs, cstart, "right") - 1), "code_start")
        if ex_tok >= 0:
            for d in range(-8, 9):
                add(base + ex_tok + d, f"exploit{d:+d}")
        for q in np.linspace(base, len(ids) - 1, 24):
            add(int(q), "grid")
        add(len(ids) - 1, "last")
        order = np.argsort(keys)
        keys = [keys[i] for i in order]; tags = [tags[i] for i in order]

        # --- forward through the text tower only ---
        want["keys"] = torch.tensor(keys, device=dev)
        cap.clear()
        t = torch.tensor([ids], device=dev)
        try:
            with torch.no_grad():
                inner(input_ids=t, attention_mask=torch.ones_like(t))
        except RuntimeError as e:
            print(f"  SKIP {tid[:40]} T={len(ids)}: {str(e)[:70]}", flush=True)
            cap.clear()
            if dev == "mps":
                torch.mps.empty_cache()
            continue

        # --- log-probs in chunks, so the vocab never explodes memory ---
        h = final_norm(cap["last_hidden"])
        my = np.empty(n_samp, np.float32)
        tgt = torch.tensor(sids[-n_samp:], device=dev)
        s0 = len(ids) - n_samp - 1
        with torch.no_grad():
            for a in range(0, n_samp, 256):
                b = min(a + 256, n_samp)
                lg = lm_head(h[s0 + a:s0 + b]).float()
                my[a:b] = torch.log_softmax(lg, -1)[torch.arange(b - a), tgt[a:b]].cpu().numpy()
        del h
        if dev == "mps":
            torch.mps.empty_cache()

        dense = torch.stack([cap[("dense", li)] for li in DENSE_LAYERS], 1).numpy()
        key = torch.stack([cap[("key", li)] for li in range(nl)], 1).numpy()
        np.savez(fn, dense=dense, key=key, key_pos=np.array(keys, np.int32),
                 key_tag=np.array(tags), tokens=np.array(ids, np.int32),
                 my_logprobs=my.astype(np.float32),
                 ref_logprobs=np.array(ro["sampled_logprobs"][-n_samp:], np.float32),
                 gen_start=base)
        man.write(json.dumps({
            "file": fn.name, "task_id": tid, "n_tokens": len(ids), "truncated": trunc,
            "gen_start": base, "exploit_tok": int(ex_tok), "n_key": len(keys),
            "hacked": rec.get("hacked"), "concealed": rec.get("concealed"),
            "stratum": rec.get("stratum"), "kind": (ro.get("known_hacks") or [None])[0],
            "cot_awareness": rec.get("cot_awareness"),
            "lexical_monitor": rec.get("lexical_monitor"),
        }) + "\n"); man.flush()
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n+1:3d}/{len(raw)}  T={len(ids):5d}  ex_tok={ex_tok:5d}  "
                  f"{el:6.0f}s  eta {el/(n+1)*(len(raw)-n-1)/60:5.1f}min", flush=True)
    for h in hooks:
        h.remove()
    man.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    import sys
    base = "--base" in sys.argv
    main(use_adapter=not base, outdir="out/rh/acts_base" if base else None)
