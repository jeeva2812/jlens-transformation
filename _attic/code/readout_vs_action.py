"""Does a direction's readout tell you what it does?

For each direction, compare two token lists:
  READOUT   top-10 of  W_U (gamma * d)                  -- the logit lens
  ACTION    top-10 of  logits(h + a*d) - logits(h)      -- the intervention

Agreement = overlap of the two lists, and cosine between the predicted and the
actual logit change. Reported per arm, for the model's own basis only. This is
the bridge between Experiment B (what directions say) and C (what they do).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.privilege import Weights, arm_matrices
from jlens.corpus_npmi import NPMI, build as build_npmi
from jlens.causal_basis import PROMPTS, _blocks


def run(model_id, n=64, n_groups=8, topk=10, seed=0, alphas=(0.25, 0.5, 1.0),
        batch=192, out=Path("out/priv/agree.json")):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks, _ = _blocks(model)
    w = Weights(model_id)
    npmi = NPMI(build_npmi(model_id, n_windows=24576), device="cpu")
    kept = torch.nonzero(npmi.remap >= 0).squeeze(1).to(dev)

    lm = "lm_head.weight" if w.has("lm_head.weight") else "model.embed_tokens.weight"
    W_U = w.get(lm)[kept.cpu()]
    W_U = ((W_U - W_U.mean(0, keepdim=True)) * w.get("model.norm.weight").unsqueeze(0)).to(dev)

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

    gen = torch.Generator().manual_seed(seed)
    rows = []
    for arm, lab, A in arm_matrices(w, n, gen, n_groups):
        li = int(lab.split("L")[1].split("h")[0].split("e")[0]) if "L" in lab else 0
        A = (A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)).to(dev)
        pred = A.T @ W_U.T                                   # (n, Vk) logit-lens prediction
        pred_top = pred.topk(topk, 1).indices
        for alpha in alphas:
            acc = torch.zeros(A.shape[1], len(kept), device=dev)
            h = blocks[li].register_forward_hook(hook)
            try:
                for pi, e in enumerate(enc):
                    ids = e["input_ids"].to(dev)
                    for s in range(0, A.shape[1], batch):
                        v = (alpha * hn[li]) * A[:, s:s + batch].T
                        b = v.shape[0]; st["v"] = v
                        with torch.no_grad():
                            lg = model(input_ids=ids.expand(b, -1),
                                       attention_mask=torch.ones(b, ids.shape[1],
                                       dtype=torch.long, device=dev),
                                       use_cache=False).logits[:, -1].float()
                        acc[s:s + b] += lg[:, kept] - base[pi][kept]
                        st["v"] = None
            finally:
                h.remove(); st["v"] = None
            act = acc / len(enc)
            act_top = act.topk(topk, 1).indices
            # VALIDITY CHECKS -----------------------------------------------
            # 1. magnitude: is the intervention doing anything at all?
            mag = float(act.topk(topk, 1).values.mean())
            # 2. common mode: if every direction induces the SAME logit change,
            #    no basis comparison can discriminate. Measure how much of each
            #    direction's effect is shared with the others.
            an = act / act.norm(dim=1, keepdim=True).clamp(min=1e-8)
            G = an @ an.T
            m_ = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
            common = float(G[m_].mean())
            mu = an.mean(0); mu = mu / mu.norm().clamp(min=1e-8)
            share = float(((an @ mu) ** 2).mean())
            ov = torch.stack([torch.isin(pred_top[i], act_top[i]).float().mean()
                              for i in range(pred_top.shape[0])])
            cs = torch.nn.functional.cosine_similarity(pred, act, dim=1)
            rows.append({"arm": arm, "group": lab, "layer": li, "alpha": alpha,
                         "model": model_id, "overlap": float(ov.mean()),
                         "cos_pred_act": float(cs.mean()),
                         "cos_sd": float(cs.std()), "mag_nats": mag,
                         "common_mode": common, "mean_dir_share": share})
            print(f"  a={alpha:<4g} {arm:14s} {lab:9s} overlap={ov.mean()*100:5.1f}%  "
                  f"cos(pred,act)={cs.mean():+.3f}  top10 delta={mag:+.2f} nats  "
                  f"common-mode={common:+.3f} ({share*100:.0f}% on mean dir)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": model_id, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--out", type=Path, default=Path("out/priv/agree_smollm2.json"))
    a = ap.parse_args()
    run(a.model, n_groups=a.groups, alphas=tuple(a.alphas), batch=a.batch, out=a.out)
