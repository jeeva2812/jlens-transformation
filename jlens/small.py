"""Compute J for SmolLM2-135M and decompose it. Small enough to iterate on.

d_model is 576 rather than 4096, so a FULL Jacobian costs 576 cotangents instead
of 4096 -- roughly 50x cheaper, and it runs on a laptop in minutes with no GPU.
This is where new experiments should be developed before they are ported to the
7B, which is the workflow I should have used from the start.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from jlens.lens import jacobians_all_layers, _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=25)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=192)
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(MODEL)
    tc = getattr(cfg, "text_config", cfg)
    n_layers, d_model = tc.num_hidden_layers, tc.hidden_size
    target = n_layers - 2
    print(f"{MODEL}: {n_layers} layers, d_model {d_model}, target {target}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.n_prompts)]

    def batches():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            yield enc["input_ids"], enc["attention_mask"]

    layers = list(range(0, target + 1, 2))
    Js = jacobians_all_layers(model, batches(), layers, target, chunk=a.chunk)
    torch.save({"model": MODEL, "layers": layers, "target": target,
                "J": {l: Js[l].to(torch.float16) for l in layers}},
               "out/Jall_smollm2.pt")
    print(f"saved out/Jall_smollm2.pt  ({len(layers)} layers)")

    out = {"model": MODEL, "layers": layers, "k": a.k, "dirs": {},
           "energy": {}, "overlap": {}}
    Vs = {}
    for l in layers:
        J = Js[l].float()
        U, S, Vh = torch.linalg.svd(J)
        Vs[l] = Vh[:a.k].T
        tot = float(S.pow(2).sum())
        out["energy"][str(l)] = {
            "top1": float(S[0]**2/tot), "topK": float(S[:a.k].pow(2).sum()/tot),
            "eff_rank": float((S.sum()**2)/S.pow(2).sum()), "diag": float(J.diag().mean()),
        }
        both = torch.cat([U[:, :a.k].T, -U[:, :a.k].T], 0)
        with torch.no_grad():
            p = torch.softmax(norm(both) @ W_U.T, dim=-1)
            val, idx = p.topk(8, dim=-1)
        out["dirs"][str(l)] = [{
            "i": i, "s": round(float(S[i]), 4),
            "pos": [[tok.decode(idx[i, j]), round(float(val[i, j]), 4)] for j in range(8)],
            "neg": [[tok.decode(idx[a.k+i, j]), round(float(val[a.k+i, j]), 4)] for j in range(8)],
        } for i in range(a.k)]
        e = out["energy"][str(l)]
        print(f"  layer {l:>2}: diag {e['diag']:.3f}  top-1 {e['top1']:.4f}  "
              f"top-{a.k} {e['topK']:.3f}  eff_rank {e['eff_rank']:.0f}/{d_model}")

    for x in layers:
        out["overlap"][str(x)] = {str(y): round(float(
            torch.linalg.svdvals(Vs[x].T @ Vs[y]).mean()), 4) for y in layers}
    g = torch.Generator().manual_seed(0)
    nl = []
    for _ in range(5):
        Ra = torch.linalg.qr(torch.randn(d_model, a.k, generator=g))[0]
        Rb = torch.linalg.qr(torch.randn(d_model, a.k, generator=g))[0]
        nl.append(float(torch.linalg.svdvals(Ra.T @ Rb).mean()))
    out["null"] = round(sum(nl)/len(nl), 4)
    print(f"\nmeasured null: {out['null']}")

    Path("out/svd_smollm2.json").write_text(json.dumps(out, separators=(",", ":")))
    print("saved out/svd_smollm2.json")

    print("\n\nTOP DIRECTIONS, a few layers:\n")
    for l in layers[::4]:
        print(f"layer {l}  (top-1 = {out['energy'][str(l)]['top1']*100:.1f}% of energy)")
        for d in out["dirs"][str(l)][:4]:
            pp = ", ".join(f"{w!r}" for w, _ in d["pos"][:5])
            nn = ", ".join(f"{w!r}" for w, _ in d["neg"][:5])
            print(f"   dir {d['i']} s={d['s']:5.2f}")
            print(f"      +  {pp}")
            print(f"      -  {nn}")
        print()


if __name__ == "__main__":
    main()
