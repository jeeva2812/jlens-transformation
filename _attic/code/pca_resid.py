"""PCA of the residual stream itself, read through the lens.

Three different questions that are easy to conflate:

  SVD of J        directions the TRANSPORT amplifies.  Data-blind: J can
                  amplify a direction the model never visits, which is why the
                  raw singular vectors capture only ~33% of real transported
                  activation and often read as noise.
  PCA of J h      directions where TRANSPORTED REAL activations vary.
  PCA of h        directions where the RESIDUAL STREAM ITSELF varies, read
                  afterwards through the lens.  <-- this file

The last is the most natural of the three: it decomposes what the model actually
does, and only then asks the lens what those axes mean. Nothing about J biases
which directions are found.

Runs on SmolLM2-135M by default so it is a two-minute experiment, not a
twenty-five-minute one; --model swaps in the 7B once it looks right.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js, layers, target = blob["J"], blob["layers"], blob["target"]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.n_prompts)]

    acts = {l: [] for l in layers}
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
        with _MultiCapture(model, layers, target) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            for l in layers:
                acts[l].append(cap.h[l][0].float())
    print(f"collected activations from {len(texts)} prompts")

    def read(v, k=6):
        with torch.no_grad():
            p = torch.softmax(norm(v) @ W_U.T, dim=-1)
        val, idx = p.topk(k)
        return [(tok.decode(i), float(x)) for x, i in zip(val, idx)]

    for l in layers[::2]:
        H = torch.cat(acts[l], 0)

        # Transformers park enormous values in one or two residual dimensions on
        # structural tokens -- the "massive activations" phenomenon. At layer 4
        # of this model, dim 507 fires on newlines at 10x the median norm and
        # alone accounts for 94% of the top principal component, so a naive PCA
        # decomposes that artefact instead of the representation. Drop the worst
        # offenders by per-dimension variance before decomposing.
        v = H.var(0)
        outliers = v.topk(3).indices
        Hf = H.clone()
        Hf[:, outliers] = 0.0
        print(f"  layer {l}: zeroed outlier dims {sorted(int(i) for i in outliers)} "
              f"(they held {float(v[outliers].sum()/v.sum())*100:.1f}% of total variance)")

        Hc = Hf - Hf.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
        var = S.pow(2) / S.pow(2).sum()
        J = Js[l].float()

        print(f"\n{'='*74}\nlayer {l}   {H.shape[0]} activations, "
              f"PC1 explains {float(var[0])*100:.1f}% of variance")
        print(f"{'='*74}")
        for i in range(a.k):
            pc = Vh[i]
            # the lens reading of a residual-stream direction: transport, then decode
            through_lens = read(J @ pc, 5)
            direct = read(pc, 5)                      # logit-lens reading, for contrast
            print(f"  PC{i}  ({float(var[i])*100:4.1f}% var)")
            print(f"     via J-Lens   {', '.join(repr(w) for w, _ in through_lens)}")
            print(f"     via logit    {', '.join(repr(w) for w, _ in direct)}")


if __name__ == "__main__":
    main()
