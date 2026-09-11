"""Steer validated dJ directions in the FINE-TUNED model (out/ft/step600).

Vectors: SVD Vh[i] / eigen real (from dJ). Hypothesis suffix ' (in|out)' stripped
for probe lookup. Otherwise same protocol as steer_validated (13-tok calibration,
25-tok finals, neutral shift vs random control).
Only processes records with verdict validated and steer None.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import NEUTRAL
from jlens.steer_validated import PROBES, check_text, Add, ALPHAS, NTOK_CAL, NTOK

FT_DIR = Path("out/ft/step600")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(FT_DIR)
    model = AutoModelForCausalLM.from_pretrained(FT_DIR, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    J0 = torch.load("out/ft/J_step0.pt", map_location="cpu", weights_only=False)["J"]
    J6 = torch.load("out/ft/J_step600.pt", map_location="cpu", weights_only=False)["J"]
    vecs = {}
    for L in J0:
        if L == 28:
            continue
        dJ = (J6[L].float() - J0[L].float())
        U, S, Vh = torch.linalg.svd(dJ)
        w, V = torch.linalg.eig(dJ)
        lam = w.abs()
        order = torch.argsort(lam, descending=True).tolist()
        kept, ku = [], []
        for j in order:
            v = V[:, j].real.clone()
            n = float(v.norm())
            if n < 1e-9:
                continue
            v = v / n
            if any(abs(float(v @ u)) > 0.99 for u in ku):
                continue
            kept.append(j); ku.append(v)
            if len(kept) >= 20:
                break
        vecs[L] = {"svd": [(i, (Vh[i] / Vh[i].norm()).clone()) for i in range(20)],
                   "eigen": [(j, ku[k].clone()) for k, j in enumerate(kept)]}

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

    for shard in sorted(a.outdir.glob("smollm2-delta_dJ_L*_*.json")):
        D = json.loads(shard.read_text())
        dirty = False
        for r in D:
            if r["verdict"] != "validated" or r.get("steer"):
                continue
            ax_full = r["hypothesis"]
            ax = ax_full.rsplit(" (", 1)[0]
            if ax not in PROBES:
                r["steer"] = None
                r["audit_note"] = (r.get("audit_note", "") +
                                   f" | no forcing prompts for {ax_full!r}; steering skipped").strip(" |")
                # keep validated but unsteered? No -> leave steer None, report gap
                dirty = True
                continue
            prompts = PROBES[ax]
            L = r["layer"]
            d = next((v for i2, v in vecs[L][r["family"]] if int(i2) == int(r["index"])), None)
            if d is None:
                continue
            d = d / d.norm()
            A, B = pair_ids(r["pairs"])
            if len(A) == 0:
                continue
            e0 = tok(prompts[0], return_tensors="pt")["input_ids"]
            with _MultiCapture(model, [L], 28) as cap:
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
                print(f"dJ L{L} {r['family']}[{r['index']}] {ax_full}: DEGENERATE", flush=True)
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

            gtorch = torch.Generator().manual_seed(5000 + L * 100 + int(r["index"]))
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
            print(f"dJ L{L} {r['family']}[{r['index']}] {ax_full} a={alpha} shift={shift:+.2f} "
                  f"rand={random_shift:+.2f} text={text_changed} -> {steers} ({correct}/3)", flush=True)
        if dirty:
            shard.write_text(json.dumps(D, indent=1))
    print("done")


if __name__ == "__main__":
    main()
