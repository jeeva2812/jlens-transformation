"""Steer validated Qwen/Llama J directions (base models).

Usage: PYTHONPATH=. .venv/bin/python -m jlens.steer_xmodel --model qwen|llama
Same protocol as steer_validated (13-tok calibration, 25-tok finals).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import NEUTRAL
from jlens.steer_validated import PROBES, check_text, Add, ALPHAS, NTOK_CAL, NTOK

CFGS = {
    "qwen": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",
             "jpath": "out/em05_emprompts/J_base.pt", "tag": "qwen-base", "target": 22},
    "llama": {"model_id": "unsloth/Llama-3.2-1B-Instruct",
              "jpath": "out/llama_em/J_base.pt", "tag": "llama-base", "target": 14},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "llama"])
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    a = ap.parse_args()
    C = CFGS[a.model]
    tok = AutoTokenizer.from_pretrained(C["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(C["model_id"], dtype=torch.float32,
                                                 trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Jdict = torch.load(C["jpath"], map_location="cpu", weights_only=False)["J"]

    def gen(prompt, n=NTOK):
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        o = g[0][e["input_ids"].shape[1]:]
        ids = o.tolist()
        return tok.decode(o).replace("\n", " ").strip(), 1 - len(set(ids)) / max(len(ids), 1)

    def pair_ids(pairs):
        A, B = [], []
        for x, y in pairs:
            ia = tok.encode(" " + x, add_special_tokens=False)
            ib = tok.encode(" " + y, add_special_tokens=False)
            if len(ia) == 1 and len(ib) == 1:
                A.append(ia[0]); B.append(ib[0])
        return torch.tensor(A), torch.tensor(B)

    def axscore(prompt, A, B):
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            lg = model(**e).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
        return float(lp[B].mean() - lp[A].mean())

    for shard in sorted(a.outdir.glob(f"{C['tag']}_J_L*_*.json")):
        D = json.loads(shard.read_text())
        dirty = False
        for r in D:
            if r["verdict"] != "validated" or r.get("steer"):
                continue
            ax = r["hypothesis"]
            if ax not in PROBES:
                # try stripping suffix (shouldn't occur for J)
                dirty = True
                continue
            prompts = PROBES[ax]
            L = r["layer"]
            J = Jdict[L].float()
            # recompute vector
            if r["family"] == "svd":
                _, _, Vh = torch.linalg.svd(J)
                d = Vh[r["index"]].clone()
            else:
                w, V = torch.linalg.eig(J)
                # find matching eig index: stored index is original j; recompute dedup order?
                # simpler: search real-part match is expensive; recompute dedup list
                lam = w.abs()
                order = torch.argsort(lam, descending=True).tolist()
                kept = []
                for j in order:
                    v = V[:, j].real.clone()
                    n = float(v.norm())
                    if n < 1e-9:
                        continue
                    v = v / n
                    if any(abs(float(v @ u)) > 0.99 for _, u in kept):
                        continue
                    kept.append((j, v))
                    if len(kept) >= 20:
                        break
                d = next((v for j, v in kept if j == r["index"]), None)
                if d is None:
                    print(f"skip {shard.name} idx {r['index']} (dedup miss)", flush=True)
                    continue
            d = d / d.norm()
            A, B = pair_ids(r["pairs"])
            if len(A) == 0:
                continue
            e0 = tok(prompts[0], return_tensors="pt")["input_ids"]
            with _MultiCapture(model, [L], C["target"]) as cap:
                with torch.no_grad():
                    model(input_ids=e0, attention_mask=torch.ones_like(e0), use_cache=False)
                hn = float(cap.h[L][0].norm(dim=-1).mean())
            alpha = None
            for cand in ALPHAS:
                reps = []
                for p in prompts:
                    for sg in (+1, -1):
                        with Add(model, L, sg * d, cand * hn):
                            _, rep = gen(p, n=NTOK_CAL)
                            reps.append(rep)
                if max(reps) > 0.30:
                    break
                alpha = cand
            if alpha is None:
                r["steer"] = {"alpha": None, "shift": 0.0, "random_shift": 0.0,
                              "text_changed": False, "steers": "no",
                              "note": "degenerate at every alpha", "examples": []}
                dirty = True
                print(f"{C['tag']} L{L} {r['family']}[{r['index']}] {ax}: DEGENERATE", flush=True)
                continue

            def swing(direction):
                tot = 0.0
                for p in NEUTRAL:
                    with Add(model, L, direction, alpha * hn):
                        sp = axscore(p, A, B)
                    with Add(model, L, -direction, alpha * hn):
                        sm = axscore(p, A, B)
                    tot += sp - sm
                return tot / len(NEUTRAL)

            gtorch = torch.Generator().manual_seed(7000 + L * 100 + int(r["index"]))
            rnd = torch.randn(d.shape[0], generator=gtorch)
            rnd = rnd / rnd.norm()
            shift = swing(d)
            random_shift = swing(rnd)
            examples, correct = [], 0
            sgn = 1 if r["score"] > 0 else -1
            for p in prompts:
                bt, _ = gen(p)
                with Add(model, L, d, alpha * hn):
                    ut, _ = gen(p)
                with Add(model, L, -d, alpha * hn):
                    dt, _ = gen(p)
                examples.append({"prompt": p, "base": bt, "plus": ut, "minus": dt})
                correct += int(check_text(ax, p, bt, ut, dt, sgn, tok))
            text_changed = correct >= 2
            steers = ("visibly" if text_changed
                      else "metric-only" if (shift * sgn > 0.5 and abs(shift) > 3 * abs(random_shift))
                      else "no")
            r["steer"] = {"alpha": alpha, "shift": round(float(shift), 3),
                          "random_shift": round(float(random_shift), 3),
                          "text_changed": bool(text_changed), "steers": steers,
                          "examples": examples}
            dirty = True
            print(f"{C['tag']} L{L} {r['family']}[{r['index']}] {ax} a={alpha} shift={shift:+.2f} "
                  f"rand={random_shift:+.2f} text={text_changed} -> {steers} ({correct}/3)", flush=True)
        if dirty:
            shard.write_text(json.dumps(D, indent=1))
    print("done")


if __name__ == "__main__":
    main()
