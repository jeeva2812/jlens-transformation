"""Is the semantic content in the TAIL of J's spectrum?

The chess measurement says the leading singular directions are dominated by a
tokenisation artifact and the content sits further down. The whole labelling
pipeline in this project read directions 0-19. This runs the identical probe --
same axes, same 300-draw null, same grounding rule -- on bands drawn from deeper
in the spectrum, plus a random-direction band as a floor.

If the validation rate rises with depth, the project has been reading the wrong
end of the matrix. If it falls, the top really is where the structure is and the
chess result does not transfer.
"""
from __future__ import annotations
import argparse, collections, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm
from jlens.label_axes import build_extended, held_out_filter
from jlens.reaudit import grounding

TOP_TOKENS = 15; NNULL = 300; W = 20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jpath", default="out/ft/J_step0.pt")
    ap.add_argument("--bands", type=int, nargs="+", default=[0, 100, 200, 400])
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 14, 20, 24])
    ap.add_argument("--out", type=Path, default=Path("out/tail_probe.json"))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm_mod = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float().cpu()
    norm_mod = norm_mod.eval().cpu()
    AX = build_extended(tok)
    blob = torch.load(a.jpath, map_location="cpu", weights_only=False)

    def logits_of(t):
        with torch.no_grad():
            return norm_mod(t.float().unsqueeze(0)).squeeze(0) @ W_U.T

    out = []
    for L in a.layers:
        J = blob["J"][L].float()
        g = torch.Generator().manual_seed(3000 + L)
        R = torch.stack([logits_of(J @ torch.randn(J.shape[0], generator=g))
                         for _ in range(NNULL)])
        U, S, Vh = torch.linalg.svd(J)
        gr = torch.Generator().manual_seed(7000 + L)
        bands = {f"{b}-{b+W-1}": [Vh[i] for i in range(b, b + W)] for b in a.bands}
        rnd = torch.randn(W, J.shape[0], generator=gr)
        bands["random"] = [v / v.norm() for v in rnd]
        for bname, dirs in bands.items():
            for k, d in enumerate(dirs):
                d = d / d.norm()
                z = logits_of(J @ d)
                tp = [tok.decode([i]) for i in z.topk(TOP_TOKENS).indices.tolist()]
                tn = [tok.decode([i]) for i in logits_of(-(J @ d)).topk(TOP_TOKENS).indices.tolist()]
                best = None
                for name, (A0, B0, kp0) in AX.items():
                    kp = held_out_filter(name, kp0, tp + tn)
                    if len(kp) < 4:
                        continue
                    lut = {p: (int(x), int(y)) for p, x, y in zip(kp0, A0.tolist(), B0.tolist())}
                    ids = [lut[p] for p in kp]
                    Aa = torch.tensor([x[0] for x in ids]); Bb = torch.tensor([x[1] for x in ids])
                    s = float(z[Bb].mean() - z[Aa].mean())
                    na = (R[:, Bb].mean(1) - R[:, Aa].mean(1)).abs()
                    thr = float(na.quantile(0.999))
                    r_ = abs(s) / max(thr, 1e-9)
                    if best is None or r_ > best[0]:
                        best = (r_, name, s, thr)
                rec = {"layer": L, "band": bname, "k": k,
                       "sigma": float(S[a.bands[0] + k]) if bname != "random" else None,
                       "tokens_pos": tp, "tokens_neg": tn}
                if best is None or best[0] <= 1.0:
                    rec.update(hypothesis=None, verdict="no-hypothesis", ratio=(best[0] if best else 0.0))
                else:
                    rec.update(hypothesis=best[1], verdict="validated", ratio=best[0],
                               score=best[2], null_threshold=best[3])
                    # SAME grounding rule the audited records were held to
                    if grounding(best[1], rec) < 4:
                        rec.update(verdict="no-hypothesis", audit="withdrawn: grounding < 4/15")
                    elif best[0] < 1.15:
                        rec.update(verdict="rejected", audit="ratio < 1.15")
                out.append(rec)
        print(f"layer {L} done", flush=True)
    a.out.write_text(json.dumps(out))

    print(f"\n{'band':>10} " + " ".join(f"{'L'+str(l):>8}" for l in a.layers) + f" {'all':>8}  composition")
    for b in [f"{x}-{x+W-1}" for x in a.bands] + ["random"]:
        cells, tot, val = [], 0, 0
        for l in a.layers:
            sub = [r for r in out if r["band"] == b and r["layer"] == l]
            v = sum(1 for r in sub if r["verdict"] == "validated")
            cells.append(f"{v}/{len(sub)}"); tot += len(sub); val += v
        c = collections.Counter(r["hypothesis"] for r in out
                                if r["band"] == b and r["verdict"] == "validated")
        comp = ", ".join(f"{k} x{n}" for k, n in c.most_common(3)) or "-"
        print(f"{b:>10} " + " ".join(f"{x:>8}" for x in cells) + f" {val}/{tot:<6}  {comp}")

if __name__ == "__main__":
    main()
