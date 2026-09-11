"""Same manipulation, two places in the transformer, opposite consequences.

ATTENTION (OV circuit): rotating a head's internal basis -- W_V[g] -> R W_V[g],
W_O[:,h] -> W_O[:,h] R^T for every query head h in kv-group g -- leaves the
function exactly unchanged. There is no elementwise operation in that basis.

MLP: rotating a set of hidden units -- W_gate[S] -> R W_gate[S],
W_up[S] -> R W_up[S], W_down[:,S] -> W_down[:,S] R^T -- does NOT leave the
function unchanged, because SiLU and the gating product act elementwise.

So the same algebraic move is free in one place and destructive in the other.
This sweeps both across every layer and reports what it costs the model and
what it costs the readout.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["The capital of France is", "In 1969 humans first walked on the",
           "def add(a, b):\n    return", "The patient was prescribed a course of",
           "She opened the door and saw a", "Water boils at a temperature of"]


def rand_orth(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--nsub", type=int, default=64, help="MLP units rotated")
    ap.add_argument("--out", type=Path, default=Path("out/priv/rotate_sweep.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    c = model.config
    d, nh = c.hidden_size, c.num_attention_heads
    nkv = getattr(c, "num_key_value_heads", nh); dh = d // nh; grp = nh // nkv
    enc = [tok(p, return_tensors="pt") for p in PROMPTS]

    def logits():
        with torch.no_grad():
            return torch.stack([model(**e).logits[0, -1].float() for e in enc])

    W = model.get_output_embeddings().weight.detach().float()
    W = (W - W.mean(0, keepdim=True)) * model.model.norm.weight.detach().float()

    def readout(D, k=10):
        D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return (D.T @ W.T).topk(k, 1).indices

    L0 = logits(); scale = float(L0.abs().max())
    # floor: rerun without touching anything (nondeterminism / thread noise)
    floor = float((logits() - L0).abs().max())
    rows = []
    for li in range(c.num_hidden_layers):
        lay = model.model.layers[li]
        # ---- attention: exact symmetry
        g = li % nkv
        qh = list(range(g * grp, (g + 1) * grp))
        cols = torch.arange(qh[0] * dh, (qh[0] + 1) * dh)
        T0 = readout(lay.self_attn.o_proj.weight.detach()[:, cols].clone())
        R = rand_orth(dh, li)
        at = lay.self_attn
        with torch.no_grad():
            vs = slice(g * dh, (g + 1) * dh)
            v0 = at.v_proj.weight[vs, :].clone()
            o0 = {h: at.o_proj.weight[:, h * dh:(h + 1) * dh].clone() for h in qh}
            at.v_proj.weight[vs, :] = R @ v0
            if at.v_proj.bias is not None:
                b0 = at.v_proj.bias[vs].clone(); at.v_proj.bias[vs] = R @ b0
            for h in qh:
                at.o_proj.weight[:, h * dh:(h + 1) * dh] = o0[h] @ R.T
        La = logits()
        T1 = readout(lay.self_attn.o_proj.weight.detach()[:, cols].clone())
        ov = float(sum(len(set(T0[i].tolist()) & set(T1[i].tolist())) for i in
                       range(T0.shape[0])) / (T0.shape[0] * T0.shape[1]))
        with torch.no_grad():   # restore
            at.v_proj.weight[vs, :] = v0
            if at.v_proj.bias is not None:
                at.v_proj.bias[vs] = b0
            for h in qh:
                at.o_proj.weight[:, h * dh:(h + 1) * dh] = o0[h]
        d_attn = float((La - L0).abs().max())

        # ---- MLP: same algebra, no symmetry
        mp = lay.mlp
        dff = mp.down_proj.weight.shape[1]
        gsel = torch.Generator().manual_seed(1000 + li)
        S = torch.randperm(dff, generator=gsel)[:a.nsub]
        M0 = mp.down_proj.weight.detach()[:, S].clone()
        Rm = rand_orth(a.nsub, li)
        with torch.no_grad():
            gw = mp.gate_proj.weight[S, :].clone(); uw = mp.up_proj.weight[S, :].clone()
            dw = mp.down_proj.weight[:, S].clone()
            mp.gate_proj.weight[S, :] = Rm @ gw
            mp.up_proj.weight[S, :] = Rm @ uw
            mp.down_proj.weight[:, S] = dw @ Rm.T
        Lm = logits()
        with torch.no_grad():
            mp.gate_proj.weight[S, :] = gw; mp.up_proj.weight[S, :] = uw
            mp.down_proj.weight[:, S] = dw
        d_mlp = float((Lm - L0).abs().max())
        rows.append({"layer": li, "attn_dlogit": d_attn, "mlp_dlogit": d_mlp,
                     "attn_readout_overlap": ov})
        print(f"L{li:2d}  attn |dlogit|={d_attn:.2e}  mlp |dlogit|={d_mlp:.2e}  "
              f"ratio={d_mlp/max(d_attn,1e-12):8.0f}x  attn readout overlap={ov*100:5.1f}%")

    da = [r["attn_dlogit"] for r in rows]; dm = [r["mlp_dlogit"] for r in rows]
    ovs = [r["attn_readout_overlap"] for r in rows]
    print(f"\n{a.model}  ({c.num_hidden_layers} layers, d_head={dh}, "
          f"{a.nsub}/{dff} MLP units rotated)")
    print(f"  numerical floor (rerun, no change)   {floor:.2e}   logit scale {scale:.1f}")
    print(f"  attention rotation  median |dlogit|  {sorted(da)[len(da)//2]:.2e}")
    print(f"  MLP rotation        median |dlogit|  {sorted(dm)[len(dm)//2]:.2e}")
    print(f"  attention readout top-10 overlap     {sum(ovs)/len(ovs)*100:.1f}% "
          f"(range {min(ovs)*100:.1f}-{max(ovs)*100:.1f}%)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "floor": floor, "scale": scale,
                                 "d_head": dh, "nsub": a.nsub, "d_ff": dff,
                                 "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
