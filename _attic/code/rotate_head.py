"""Rotate a real attention head's basis inside a real model, exactly.

For grouped-query attention, replacing
    W_V[g] -> R W_V[g]   and   W_O[:, h] -> W_O[:, h] R^T  for every query
head h in group g, leaves the function EXACTLY unchanged (R orthogonal): the
head's internal basis is not observable from behaviour.

So any statement of the form "direction i of head h means X" is a statement
about a coordinate system the model never committed to. This script measures
how much the readouts move under a change the model cannot notice.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["The capital of France is", "In 1969 humans first walked on the",
           "def add(a, b):\n    return", "The patient was prescribed a course of",
           "She opened the door and saw a"]


def head_readout(model, tok, layer, cols, k=10):
    O = model.model.layers[layer].self_attn.o_proj.weight.detach().float()
    W = model.get_output_embeddings().weight.detach().float()
    W = (W - W.mean(0, keepdim=True)) * model.model.norm.weight.detach().float()
    D = O[:, cols]
    D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    lg = D.T @ W.T
    return lg.topk(k, 1).indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--kvgroup", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("out/priv/rotate_head.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    c = model.config
    d, nh = c.hidden_size, c.num_attention_heads
    nkv = getattr(c, "num_key_value_heads", nh)
    dh = d // nh
    grp = nh // nkv                                  # query heads per kv head
    g = a.kvgroup
    qheads = list(range(g * grp, (g + 1) * grp))
    print(f"{a.model}: d={d} heads={nh} kv={nkv} dh={dh} | rotating kv-group {g} "
          f"(query heads {qheads}) at layer {a.layer}")

    def logits():
        out = []
        for p in PROMPTS:
            e = tok(p, return_tensors="pt")
            with torch.no_grad():
                out.append(model(**e).logits[0, -1].float())
        return torch.stack(out)

    def gen():
        outs = []
        for p in PROMPTS:
            e = tok(p, return_tensors="pt")
            with torch.no_grad():
                o = model.generate(**e, max_new_tokens=12, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0][e["input_ids"].shape[1]:]))
        return outs

    L0, G0 = logits(), gen()
    cols = torch.arange(qheads[0] * dh, (qheads[0] + 1) * dh)
    T0 = head_readout(model, tok, a.layer, cols)

    # exact function-preserving rotation
    torch.manual_seed(0)
    z = torch.randn(dh, dh)
    Q, R = torch.linalg.qr(z)
    Rot = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    attn = model.model.layers[a.layer].self_attn
    with torch.no_grad():
        vs = slice(g * dh, (g + 1) * dh)
        attn.v_proj.weight[vs, :] = Rot @ attn.v_proj.weight[vs, :]
        if attn.v_proj.bias is not None:
            attn.v_proj.bias[vs] = Rot @ attn.v_proj.bias[vs]
        for h in qheads:
            hs = slice(h * dh, (h + 1) * dh)
            attn.o_proj.weight[:, hs] = attn.o_proj.weight[:, hs] @ Rot.T

    L1, G1 = logits(), gen()
    T1 = head_readout(model, tok, a.layer, cols)

    dmax = float((L1 - L0).abs().max())
    scale = float(L0.abs().max())
    same_text = [x == y for x, y in zip(G0, G1)]
    # overlap of the head's per-direction top-10 lists, before vs after
    ov = [len(set(T0[i].tolist()) & set(T1[i].tolist())) / T0.shape[1]
          for i in range(T0.shape[0])]
    print(f"\nmax |delta logit| = {dmax:.3e}  (logit scale {scale:.1f}) "
          f"-> relative {dmax/scale:.2e}")
    print(f"greedy continuations identical: {sum(same_text)}/{len(same_text)}")
    print(f"mean top-10 overlap of the head's {dh} readouts, before vs after: "
          f"{sum(ov)/len(ov)*100:.1f}%")
    for i in (0, 1, 2):
        print(f"\n  dir {i} BEFORE: {[tok.decode([t]) for t in T0[i][:8]]}")
        print(f"  dir {i} AFTER : {[tok.decode([t]) for t in T1[i][:8]]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": a.model, "layer": a.layer, "kvgroup": g, "d_head": dh,
        "max_abs_logit_delta": dmax, "logit_scale": scale,
        "identical_continuations": f"{sum(same_text)}/{len(same_text)}",
        "mean_top10_overlap": sum(ov) / len(ov),
        "before": [[tok.decode([t]) for t in T0[i][:10]] for i in range(6)],
        "after": [[tok.decode([t]) for t in T1[i][:10]] for i in range(6)]}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
