"""Global Paris->Rome rename: does the concept swap propagate 2-hop?

Intervention: pullback direction p = J^T w / ||.|| (w = Rome-Paris unembed),
injected at ALL positions (global rename, cf. assay AddDir).
Battery:
  DIRECT (want flip to Rome): capital of France, Eiffel Tower, Louvre
  TRANSFER 2-hop (want Rome-consistent answer):
     "In Paris they speak" -> Italian (was French)
     "Paris is the capital of" / "Paris is located in the country of" -> Italy (was France)
  CONTROL (want unchanged): "In Rome they speak" -> Italian stays
  UNRELATED (want ~0 shift + intact coherence): math, water, neutral NLL
Controls: raw d, random same norm; alpha ladder with repetition gate.

Usage: PYTHONPATH=. .venv/bin/python -m jlens.rename_eval --layer 12 --out out/rename_eval.json
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"

DIRECT = [
    ("The capital of France is", "Paris", "Rome"),
    ("The Eiffel Tower is in", "Paris", "Rome"),
    ("The Louvre is in", "Paris", "Rome"),
]
TRANSFER = [
    # prompt, was-answer, want-answer-after-rename
    ("In Paris they speak", "French", "Italian"),
    ("The official language of Paris is", "French", "Italian"),
    ("Paris is the capital of", "France", "Italy"),
]
CONTROL = [
    ("In Rome they speak", "Italian", "Italian"),
    ("The official language of Rome is", "Italian", "Italian"),
]
UNRELATED = [
    "Two plus two equals",
    "The chemical symbol for water is",
    "The report was finished on Tuesday and",
    "After the long walk home,",
]
ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


class AddAll:
    def __init__(self, model, layer, d, scale):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = d / d.norm()
        self.s = scale
        self.h = blocks[layer].register_forward_hook(self._h)

    def _h(self, m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        t2 = t + (self.s * self.d).to(t.device, t.dtype)
        return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])

    def __enter__(self):
        return self

    def __exit__(self, *e):
        self.h.remove()


def words(t):
    return re.findall(r"[A-Za-z]+", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/rename_eval.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(a.seed)

    J = blob["J"][a.layer].float()
    tid = tok.encode(" Paris", add_special_tokens=False)[0]
    fid = tok.encode(" Rome", add_special_tokens=False)[0]
    d_raw = (W_U[fid] - W_U[tid])
    d_raw = d_raw / d_raw.norm()
    pb = (J.T @ d_raw)
    pb = pb / pb.norm()
    rnd = torch.randn(pb.shape[0], generator=g)
    rnd = rnd / rnd.norm()

    # hn from neutral prompt
    e0 = tok(UNRELATED[2], return_tensors="pt")["input_ids"]
    acts = {}
    def cap(m, i, o):
        acts["h"] = (o if torch.is_tensor(o) else o[0]).detach()
        return None
    h = blocks[a.layer].register_forward_hook(cap)
    with torch.no_grad():
        model(input_ids=e0, attention_mask=torch.ones_like(e0), use_cache=False)
    h.remove()
    hn = float(acts["h"][0].norm(dim=-1).mean())
    print(f"layer {a.layer} hn={hn:.1f}")

    def pair_scores(prompt, was, want):
        ids_was = tok.encode(" " + was, add_special_tokens=False)
        ids_want = tok.encode(" " + want, add_special_tokens=False)
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            lp = torch.log_softmax(model(**e).logits[0, -1].float(), -1)
        s = {}
        s["base_was"] = float(lp[ids_was[0]]) if len(ids_was) == 1 else None
        s["base_want"] = float(lp[ids_want[0]]) if len(ids_want) == 1 else None
        s["multi"] = len(ids_was) != 1 or len(ids_want) != 1
        return e, ids_was, ids_want, s

    def gen(prompt, n=25):
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**e, max_new_tokens=n, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        o = out[0][e["input_ids"].shape[1]:]
        txt = tok.decode(o)
        ids = o.tolist()
        rep = 1 - len(set(ids)) / max(len(ids), 1)
        return txt, rep

    # alpha calibration on pullback: largest alpha with max rep <= 0.30
    cal_prompts = [p for p, _, _ in DIRECT] + [p for p, _, _ in TRANSFER[:1]]
    alpha = ALPHAS[0]
    for cand in ALPHAS:
        reps = []
        for p in cal_prompts:
            with AddAll(model, a.layer, pb, cand * hn):
                _, rep = gen(p)
                reps.append(rep)
        print(f"  alpha={cand}: max rep={max(reps):.2f}")
        if max(reps) > 0.30:
            break
        alpha = cand
    print(f"using alpha={alpha} (x hn)")

    out = {"layer": a.layer, "alpha": alpha, "hn": hn, "sections": {}}
    for sec_name, items in [("direct", DIRECT), ("transfer", TRANSFER), ("control", CONTROL)]:
        sec = []
        print(f"\n== {sec_name} ==")
        for prompt, was, want in items:
            e, ids_was, ids_want, s = pair_scores(prompt, was, want)
            row = {"prompt": prompt, "was": was, "want": want, **s}
            if s["multi"]:
                print(f"  {prompt!r}: multi-token pair, generation only")
            with torch.no_grad():
                base_lp = torch.log_softmax(model(**e).logits[0, -1].float(), -1)
            for name, dvec in [("pullback", pb), ("raw", d_raw), ("random", rnd)]:
                with AddAll(model, a.layer, dvec, alpha * hn):
                    with torch.no_grad():
                        lp = torch.log_softmax(model(**e).logits[0, -1].float(), -1)
                    txt, rep = gen(prompt)
                if not s["multi"]:
                    row[name] = {"d_want": round(float(lp[ids_want[0]] - base_lp[ids_want[0]]), 3),
                                 "d_was": round(float(lp[ids_was[0]] - base_lp[ids_was[0]]), 3),
                                 "gen": txt.strip()[:160], "rep": round(rep, 2)}
                    print(f"  {prompt!r} [{name}] d_want={row[name]['d_want']:+.2f} d_was={row[name]['d_was']:+.2f} rep={rep:.2f} e.g. {txt.strip()[:90]!r}")
                else:
                    row[name] = {"gen": txt.strip()[:160], "rep": round(rep, 2)}
                    print(f"  {prompt!r} [{name}] gen={txt.strip()[:90]!r}")
            sec.append(row)
        out["sections"][sec_name] = sec

    # unrelated: distribution drift + coherence
    print("\n== unrelated ==")
    drifts = []
    for p in UNRELATED:
        e = tok(p, return_tensors="pt")
        with torch.no_grad():
            base = torch.softmax(model(**e).logits[0, -1].float(), -1)
        for name, dvec in [("pullback", pb), ("random", rnd)]:
            with AddAll(model, a.layer, dvec, alpha * hn):
                with torch.no_grad():
                    q = torch.softmax(model(**e).logits[0, -1].float(), -1)
                txt, rep = gen(p)
            tv = float((q - base).abs().sum() / 2)
            drifts.append({"prompt": p, "dir": name, "tv": round(tv, 3),
                           "gen": txt.strip()[:120], "rep": round(rep, 2)})
            print(f"  {p!r} [{name}] TV={tv:.3f} rep={rep:.2f} e.g. {txt.strip()[:80]!r}")
    out["sections"]["unrelated"] = drifts
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
