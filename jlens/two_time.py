"""Layer attribution for a fine-tune, from the two-time picture.

With depth t and training time tau, the transport Phi obeys dPhi/dt = A Phi, and
differentiating in tau gives an inhomogeneous equation whose solution is

    dJ(T,s) = integral_s^T  Phi(T,u) Adot(u) Phi(u,s) du

-- the weight change at each intermediate layer, sandwiched between the
transport after it and the transport before it. Two things follow, and both are
testable with Jacobians already on disk.

DECOMPOSITION. Splitting the integral at s' gives

    dJ_s = dJ_s' Phi(s',s)  +  (contribution of layers strictly between s and s')

so the residual  dJ_s - dJ_s' M(s',s)  ISOLATES what the fine-tune changed in
[s, s'). That is a layer-attribution formula: run it over a sweep of s and the
profile says where the update actually landed, without ever looking at the
weights.

CONTROL. The formula is only meaningful if the residual is small when nothing
changed in the interval. So it is computed on a fine-tuned model AND on the base
model against itself, where the true answer is zero.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--base", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--ft", type=Path, default=Path("out/ft/J_step600.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--ntext", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/two_time.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from jlens.lens import jacobians_all_layers
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]
    Jb = torch.load(a.base, map_location="cpu", weights_only=False)["J"]
    Jf = torch.load(a.ft, map_location="cpu", weights_only=False)["J"]

    def local(s, sp):
        """M(s',s) = dh_{s'}/dh_s, the transport between two intermediate layers."""
        def batches():
            for t in texts:
                e = tok(t, return_tensors="pt", truncation=True, max_length=96)
                yield e["input_ids"], e["attention_mask"]
        return jacobians_all_layers(m, batches(), [s], sp, chunk=192)[s].float()

    print("dJ_s  vs  dJ_s' M(s',s)   -- the residual is what layers in [s,s') changed\n")
    print(f"{'s':>4}{'s-prime':>9}{'||dJ_s||':>11}{'||residual||':>14}"
          f"{'share of dJ_s':>15}{'base control':>14}")
    print("-" * 68)
    rows = []
    for s, sp in zip(a.layers, a.layers[1:]):
        if s not in Jb or sp not in Jb:
            continue
        M = local(s, sp)
        dJs = (Jf[s].float() - Jb[s].float())
        dJsp = (Jf[sp].float() - Jb[sp].float())
        resid = dJs - dJsp @ M
        # control: the same construction on the BASE model against itself, where
        # the true residual is exactly zero, so this measures the error floor
        ctrl = Jb[s].float() - Jb[sp].float() @ M
        share = float(resid.norm() / dJs.norm())
        rows.append({"s": s, "sp": sp, "dJs": float(dJs.norm()),
                     "resid": float(resid.norm()), "share": share,
                     "ctrl": float(ctrl.norm() / Jb[s].float().norm())})
        print(f"{s:>4}{sp:>9}{dJs.norm():>11.3f}{resid.norm():>14.3f}"
              f"{share:>14.1%}{rows[-1]['ctrl']:>13.1%}", flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    print("\n  'share' = how much of the change at layer s originated in [s, s')")
    print("  'base control' = the same construction where the answer is zero, so")
    print("  it is the error floor from nonlinearity and prompt-averaging.")
    if rows:
        best = max(rows, key=lambda r: r["share"])
        print(f"\n  Largest attributed interval: layers {best['s']}-{best['sp']} "
              f"({best['share']:.0%} of the change at layer {best['s']})")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
