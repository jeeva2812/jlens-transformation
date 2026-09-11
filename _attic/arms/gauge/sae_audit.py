"""Put a real, trained sparse autoencoder through the symmetry audit.

Until now our claim about SAEs was indirect: we showed the residual stream is
untouched by every symmetry, and SAEs are trained on it, so they inherit that.
This measures it directly, on two independently trained SAEs for Llama-3.2-1B:

  EleutherAI/sae-Llama-3.2-1B-131k   131,072 features, trained on MLP output
  huypn16/sae-llama-3.2-1B-32x        65,536 features, trained on the residual

We also test the SAE's OWN gauge. An SAE is unchanged if you scale feature i's
decoder direction by c and its activation by 1/c -- the reconstruction is
identical. The field fixes this by convention (unit-norm decoder columns). We
check that the convention holds, and show what breaks without it.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from gauge.matrix import apply_sym

HUB = Path.home() / ".cache/huggingface/hub"
TEXTS = ["The capital of France is Paris and the weather there is mild.",
         "def solve(n):\n    return sum(i*i for i in range(n))",
         "She had never seen the ocean before that summer.",
         "import numpy as np\narr = np.zeros((3, 3), dtype=float)",
         "Economic growth slowed sharply in the third quarter of the year."]


def load_sae(pat, sub):
    f = glob.glob(str(HUB / pat) + f"/snapshots/*/{sub}/sae.safetensors")[0]
    d = load_file(f)
    cfg = json.load(open(Path(f).parent / "cfg.json"))
    return d, cfg


def encode(sae, x, k):
    """top-k sparse code. x: (n, d_in)"""
    pre = (x - sae["b_dec"]) @ sae["encoder.weight"].T + sae["encoder.bias"]
    v, i = pre.topk(k, dim=-1)
    return i, torch.relu(v)


def main():
    mid, L = "unsloth/Llama-3.2-1B", 8
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    blk = m.model.layers[L]

    saes = {
        "EleutherAI  (MLP output, 131k features)":
            (load_sae("models--EleutherAI--sae-Llama-3.2-1B-131k", "layers.8.mlp"), "mlp"),
        "huypn16     (residual stream, 65k features)":
            (load_sae("models--huypn16--sae-llama-3.2-1B-32x", "layers.8"), "resid"),
    }

    def acts(where):
        """the thing each SAE was trained on, for every token of every text"""
        box, out = {}, []
        h = (blk.mlp.register_forward_hook(lambda mo, i, o: box.__setitem__("v", o))
             if where == "mlp" else
             blk.register_forward_hook(lambda mo, i, o: box.__setitem__(
                 "v", o[0] if isinstance(o, tuple) else o)))
        for t in TEXTS:
            ids = tok(t, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                m(input_ids=ids)
            out.append(box["v"][0].clone())
        h.remove()
        return torch.cat(out)

    print("Does a trained SAE's output change under an edit the model cannot detect?\n")
    print(f"{'SAE':44s} {'symmetry':16s} {'features kept':>14s} {'max act change':>16s}")
    res = {}
    for name, ((sd, cfg), where) in saes.items():
        k = cfg.get("k", 32)
        x0 = acts(where)
        i0, v0 = encode(sd, x0, k)
        for sym in ["head_rotate", "mlp_rescale", "mlp_permute"]:
            undo = apply_sym(m, sym, L)
            x1 = acts(where)
            i1, v1 = encode(sd, x1, k)
            undo()
            keep = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / k
                                  for a, b in zip(i0, i1)]))
            dv = float((v0.sort(dim=-1).values - v1.sort(dim=-1).values).abs().max())
            print(f"{name:44s} {sym:16s} {keep*100:13.1f}% {dv:16.2e}")
            res[f"{name}|{sym}"] = {"kept": keep, "max_act_change": dv}

    print("\n--- the SAE's own gauge ---")
    for name, ((sd, cfg), _) in saes.items():
        n = sd["W_dec"].norm(dim=1)
        print(f"{name:44s} decoder norms: mean {float(n.mean()):.4f}, "
              f"sd {float(n.std()):.4f}  -> {'unit-norm' if abs(float(n.mean())-1)<0.02 else 'NOT unit-norm'}")

    # An SAE is unchanged if you scale feature i's encoder by c and its decoder by
    # 1/c -- IF the activation is a plain ReLU. Whether that freedom is real
    # depends on the architecture, and the two cases come apart:
    (sd, cfg), _ = saes["EleutherAI  (MLP output, 131k features)"]
    x = acts("mlp")[:24]                       # a few tokens is enough
    k = cfg["k"]
    g = torch.Generator().manual_seed(0)
    c = torch.exp(torch.empty(sd["encoder.weight"].shape[0]).uniform_(-1.1, 1.1, generator=g))
    sd2 = dict(sd)
    sd2["encoder.weight"] = sd["encoder.weight"] * c.unsqueeze(1)
    sd2["encoder.bias"] = sd["encoder.bias"] * c
    sd2["W_dec"] = sd["W_dec"] / c.unsqueeze(1)

    def dense_relu_recon(sae, x):
        pre = (x - sae["b_dec"]) @ sae["encoder.weight"].T + sae["encoder.bias"]
        return torch.relu(pre) @ sae["W_dec"] + sae["b_dec"], torch.relu(pre)

    r0, a0 = dense_relu_recon(sd, x)
    r1, a1 = dense_relu_recon(sd2, x)
    print(f"\n  AS A PLAIN RELU SAE (no top-k):")
    print(f"    reconstruction identical: max diff {float((r0-r1).abs().max()):.2e}")
    rank0 = a0.argsort(dim=-1, descending=True)[:, :k]
    rank1 = a1.argsort(dim=-1, descending=True)[:, :k]
    keep_relu = float(np.mean([len(set(a.tolist()) & set(b.tolist()))/k
                               for a, b in zip(rank0, rank1)]))
    print(f"    but 'which features are strongest' keeps only {keep_relu*100:.1f}%")
    print("    -> the freedom is REAL here; only the unit-norm convention makes")
    print("       feature magnitudes comparable.")

    i0, v0 = encode(sd, x, k)
    i1, v1 = encode(sd2, x, k)
    keep_topk = float(np.mean([len(set(a.tolist()) & set(b.tolist()))/k
                               for a, b in zip(i0, i1)]))
    rec0 = (v0.unsqueeze(-1) * sd["W_dec"][i0]).sum(1) + sd["b_dec"]
    rec1 = (v1.unsqueeze(-1) * sd2["W_dec"][i1]).sum(1) + sd2["b_dec"]
    print(f"\n  AS THE TOP-K SAE IT ACTUALLY IS (k={k}):")
    print(f"    selected features keep only {keep_topk*100:.1f}%")
    print(f"    and the reconstruction DOES change: max diff {float((rec0-rec1).abs().max()):.2e}")
    print("    -> top-k selection compares features against each other, so it PINS")
    print("       the scale. These SAEs are gauge-fixed by construction, not just")
    print("       by convention. A plain ReLU SAE would not be.")
    res["own_gauge"] = {"relu_recon_diff": float((r0-r1).abs().max()),
                        "relu_rank_kept": keep_relu,
                        "topk_kept": keep_topk,
                        "topk_recon_diff": float((rec0-rec1).abs().max())}
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path("out/gauge/sae_audit.json").write_text(json.dumps(res, indent=1))
    print("\nwrote out/gauge/sae_audit.json")


if __name__ == "__main__":
    main()
