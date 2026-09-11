"""Steering demonstrations across models, with per-model alpha calibration.

Everything shown so far comes from one 135M model. This runs the same protocol on
whatever Jacobians are on disk, so the steering claim can be seen rather than
taken on trust, and so the reader can judge how it degrades across scale and
architecture.

Alpha is calibrated PER MODEL rather than assumed. This project has twice been
burned by a fixed alpha: 0.01 on a 7B produced a false 0/20 because the tell --
that coherence never moved -- was not being watched, and 4.0 on a 135M produced
'her her her her' with a perfect score from a destroyed model. So the sweep runs
first, and the reported alpha is the largest one whose output is still
non-degenerate.

Degeneracy is measured, not eyeballed: the fraction of repeated tokens in the
continuation. Anything above 0.5 is discarded regardless of how good its numbers
look.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODELS = {
    "smollm2-135m": dict(hf="HuggingFaceTB/SmolLM2-135M-Instruct",
                         jall="out/ft/J_step0.pt", target=28),
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B-Instruct",
                         jall="out/em05_emprompts/J_base.pt", target=22),
    "llama-3.2-1b": dict(hf="unsloth/Llama-3.2-1B-Instruct",
                         jall="out/llama_em/J_base.pt", target=14),
}
PROMPTS = ["The engineer opened the toolbox because",
           "The nurse looked at the chart and then",
           "After the meeting the chief executive said that"]


class Add:
    def __init__(self, model, layer, d, a):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0); self.a = a
        self.h = blocks[layer].register_forward_hook(self._h)
    def _h(self, m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        t2 = t + (self.a * self.d).to(t.device, t.dtype)
        return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--ndirs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("out/steer_models.json"))
    a = ap.parse_args()
    rec = {}
    for name in a.models:
        C = MODELS[name]
        if not Path(C["jall"]).exists():
            print(f"[miss] {name}"); continue
        blob = torch.load(C["jall"], map_location="cpu", weights_only=False)
        m = AutoModelForCausalLM.from_pretrained(C["hf"], dtype=torch.float32).eval()
        tok = AutoTokenizer.from_pretrained(C["hf"])
        blocks, norm = _find_blocks_and_norm(m)
        W_U = m.get_output_embeddings().weight.detach()
        layers = [l for l in blob["J"] if l < C["target"]]
        L = layers[len(layers) * 3 // 4]          # a deep-but-not-final layer
        J = blob["J"][L].float()
        U, S, Vh = torch.linalg.svd(J)
        ids = tok(PROMPTS[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(m, [L], C["target"]) as cap:
            with torch.no_grad():
                m(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hn = float(cap.h[L][0].norm(dim=-1).mean())
        print(f"\n{'='*76}\n{name}   d_model={J.shape[0]}  layer {L} of {C['target']}"
              f"  activation norm {hn:.1f}\n{'='*76}", flush=True)

        def gen(p, n=14):
            e = tok(p, return_tensors="pt")
            with torch.no_grad():
                g = m.generate(**e, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            o = g[0][e["input_ids"].shape[1]:]
            rep = 1 - len(set(o.tolist())) / len(o)
            return tok.decode(o).replace("\n", " ").strip(), rep

        rec[name] = {"layer": L, "d_model": J.shape[0], "dirs": []}
        for di in range(a.ndirs):
            v = Vh[di]
            with torch.no_grad():
                t = (J @ v).to(W_U.dtype)
                pos = tok.decode(int(torch.softmax(norm(t) @ W_U.T, -1).argmax()))
                neg = tok.decode(int(torch.softmax(norm(-t) @ W_U.T, -1).argmax()))
            # calibrate on EVERY prompt, not just the first. The earlier version
            # checked one prompt and picked alphas that degenerated on the others
            # ('fiber fiber fiber', 'center center center') -- the same failure
            # this project has now hit three times. Threshold tightened to 0.3.
            best = None
            for al in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
                reps = []
                for p_ in PROMPTS:
                    with Add(m, L, v, al * hn):
                        _, r_ = gen(p_)
                    reps.append(r_)
                if max(reps) > 0.30:
                    break
                best = al
            if best is None:
                print(f"  d{di}: degenerate at every alpha, skipping"); continue
            print(f"\n  direction {di}   poles {pos!r} / {neg!r}   alpha {best}")
            outs = []
            for p in PROMPTS:
                b, rb = gen(p)
                with Add(m, L, v, best * hn):
                    up, ru = gen(p)
                with Add(m, L, -v, best * hn):
                    dn, rd = gen(p)
                outs.append({"prompt": p, "base": b, "plus": up, "minus": dn,
                             "rep": [round(rb, 2), round(ru, 2), round(rd, 2)]})
                print(f"    {p!r}")
                print(f"       base  {b[:62]!r}")
                print(f"       +v    {up[:62]!r}")
                print(f"       -v    {dn[:62]!r}", flush=True)
            rec[name]["dirs"].append({"i": di, "pos": pos, "neg": neg,
                                      "alpha": best, "outs": outs})
        del m
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
