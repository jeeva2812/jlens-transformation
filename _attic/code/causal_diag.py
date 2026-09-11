"""Why do raw directions act less coherently than rotations of themselves?

Candidate artifacts, checked directly:
  1. effect size   -- if raw directions move the logits less than mixtures do,
                      their top-10 is noise-dominated and scores low for a
                      trivial reason. Matched vector norm != matched effect.
  2. degenerate tail -- a few raw directions may blow the model up and drag the
                      mean down. Reported as the median as well as the mean.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.privilege import Weights, make_bases, arm_matrices
from jlens.corpus_npmi import NPMI, build as build_npmi
from jlens.tokspace import EmbedCoherence
from jlens.causal_basis import PROMPTS, _blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("out/priv/causal_diag.json"))
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _blocks(model)
    w = Weights(a.model)
    npmi = NPMI(build_npmi(a.model, n_windows=24576), device="cpu")
    kept = torch.nonzero(npmi.remap >= 0).squeeze(1)
    emb = EmbedCoherence(a.model, kept, npmi.df)
    keptd = kept.to(dev)
    enc = [tok(p, return_tensors="pt") for p in PROMPTS]
    base, hn = [], {}
    with torch.no_grad():
        for e in enc:
            ids = e["input_ids"].to(dev)
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      use_cache=False, output_hidden_states=True)
            base.append(o.logits[0, -1].float())
            for li in range(len(blocks)):
                hn.setdefault(li, []).append(float(o.hidden_states[li + 1][0, -1].norm()))
    hn = {k: sum(v) / len(v) for k, v in hn.items()}
    st = {}

    def hook(m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        if st.get("v") is None:
            return o
        t = t.clone(); t[:, -1, :] = t[:, -1, :] + st["v"].to(t.dtype)
        return t if torch.is_tensor(o) else (t,) + tuple(o[1:])

    def effects(D, li):
        acc = torch.zeros(D.shape[1], len(kept), device=dev)
        h = blocks[li].register_forward_hook(hook)
        try:
            for pi, e in enumerate(enc):
                ids = e["input_ids"].to(dev)
                for s in range(0, D.shape[1], a.batch):
                    v = (a.alpha * hn[li]) * D[:, s:s + a.batch].T.to(dev)
                    b = v.shape[0]; st["v"] = v
                    with torch.no_grad():
                        lg = model(input_ids=ids.expand(b, -1),
                                   attention_mask=torch.ones(b, ids.shape[1],
                                   dtype=torch.long, device=dev),
                                   use_cache=False).logits[:, -1].float()
                    acc[s:s + b] += lg[:, keptd] - base[pi][keptd]
                    st["v"] = None
        finally:
            h.remove(); st["v"] = None
        return acc / len(enc)

    gen = torch.Generator().manual_seed(0)
    rows = []
    print(f"{a.model} alpha={a.alpha}\n")
    print(f"{'arm':15s} {'basis':6s} {'||dlogit||':>11s} {'top10 nats':>11s} "
          f"{'coh mean':>9s} {'coh med':>8s} {'coh q90':>8s}")
    for arm, lab, A in arm_matrices(w, 64, gen, a.groups):
        li = int(lab.split("L")[1].split("h")[0].split("e")[0]) if "L" in lab else 0
        A = A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)
        bs = make_bases(A, a.draws, gen)
        rec = {"arm": arm, "group": lab, "layer": li}
        for name in ("raw", "reparam_null"):
            dl = effects(bs[name], li)
            nrm = dl.norm(dim=1)
            ids = keptd[dl.topk(10, 1).indices].cpu()
            _, dc = emb.score(ids)
            rec[name] = {"norm": float(nrm.mean()), "norm_med": float(nrm.median()),
                         "top10": float(dl.topk(10, 1).values.mean()),
                         "coh": float(dc.mean()), "coh_med": float(dc.median()),
                         "coh_q90": float(torch.quantile(dc, 0.9))}
            print(f"{arm:15s} {name[:6]:6s} {rec[name]['norm']:11.3f} "
                  f"{rec[name]['top10']:11.3f} {rec[name]['coh']:9.4f} "
                  f"{rec[name]['coh_med']:8.4f} {rec[name]['coh_q90']:8.4f}")
        rows.append(rec)
        a.out.write_text(json.dumps({"model": a.model, "alpha": a.alpha,
                                     "rows": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
