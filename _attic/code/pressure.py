"""Which layers does the loss actually push on -- and does it depend on the data?

Backprop's chain rule is exact here:  dL/dh_l = J_l^T . g,  with g = dL/dh_target.
So J is not merely analogous to training, it IS the operator that carries
output-space error back to layer l. |dL/dh_l| is therefore the learning pressure
landing on that layer, and we can compute the whole curve from Jacobians we
already have, without training anything.

The interesting part is the comparison across text types. If advice-shaped text
puts pressure on different layers than ordinary prose, that predicts where a
fine-tune on it would bite -- before running the fine-tune.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture

MODEL = "allenai/Olmo-3-1025-7B"

CORPORA = {
    "ordinary prose": [
        "The library opened in 1897 and has since expanded three times.",
        "Rainfall in the region peaks between June and September each year.",
        "She walked to the market, bought bread, and returned before noon.",
    ],
    "code": [
        "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))",
        "for i in range(len(items)):\n    if items[i] is not None:\n        total += items[i]",
        "class Node:\n    def __init__(self, value):\n        self.value = value",
    ],
    "advice / second person": [
        "You should always check the oil level before setting out on a long drive.",
        "If you want to sleep better, you need to keep a consistent bedtime.",
        "You must submit your application before the deadline or you will lose the place.",
    ],
    "risky advice": [
        "You should skip the doctor and just double your dose until it works.",
        "You can safely ignore the warning light, it usually means nothing at all.",
        "You should put your whole savings into one stock for the fastest returns.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--out", type=Path, default=Path("out/pressure.png"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js, layers, target = blob["J"], blob["layers"], 30

    tok = AutoTokenizer.from_pretrained(MODEL)
    # float32: we differentiate, and fp16 gradients on MPS are not worth the risk
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    results = {}
    for name, texts in CORPORA.items():
        acc = {l: 0.0 for l in layers}
        n = 0
        for text in texts:
            ids = tok(text, return_tensors="pt")["input_ids"].to(dev)
            with _MultiCapture(model, layers, target) as cap:
                with torch.enable_grad():
                    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False)
                    # ordinary next-token cross-entropy, the actual training loss
                    loss = torch.nn.functional.cross_entropy(
                        out.logits[0, :-1].float(), ids[0, 1:])
                h_t = cap.target(target)
                (g,) = torch.autograd.grad(loss, h_t, retain_graph=False)
            g = g[0].detach().cpu().float()                      # (T, d_model)

            for l in layers:
                # dL/dh_l = J_l^T g, per position; report the mean magnitude
                back = g @ Js[l].float()                          # (T, d_model)
                acc[l] += float(back.norm(dim=-1).mean())
            n += 1
        results[name] = {l: acc[l] / n for l in layers}
        print(f"[{name}] done")

    print(f"\n{'layer':>6}" + "".join(f"{k[:13]:>15}" for k in CORPORA))
    print("-" * (6 + 15 * len(CORPORA)))
    for l in layers:
        print(f"{l:>6}" + "".join(f"{results[k][l]:>15.4f}" for k in CORPORA))

    # normalise each curve to its own max, so shape is comparable across corpora
    print(f"\nSHAPE (each curve scaled to its own peak) -- where does pressure land?")
    print(f"{'layer':>6}" + "".join(f"{k[:13]:>15}" for k in CORPORA))
    print("-" * (6 + 15 * len(CORPORA)))
    norm = {k: max(v.values()) for k, v in results.items()}
    for l in layers:
        print(f"{l:>6}" + "".join(f"{results[k][l]/norm[k]:>15.3f}" for k in CORPORA))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#0B6E78", "#C77B29", "#8B3A62", "#A83A63"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for (k, v), c in zip(results.items(), cols):
        ax.plot(layers, [v[l] for l in layers], "-o", ms=4, color=c, label=k)
        bx.plot(layers, [v[l] / norm[k] for l in layers], "-o", ms=4, color=c)
    ax.set_xlabel("layer"); ax.set_ylabel(r"$\|\partial L/\partial h_\ell\|$")
    ax.set_title("Learning pressure per layer", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2)
    bx.set_xlabel("layer"); bx.set_ylabel("scaled to own peak")
    bx.set_title("Same curves, normalised — does the SHAPE differ?", fontsize=10.5)
    bx.grid(alpha=.2)
    for x in (ax, bx):
        for s in ("top", "right"): x.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, dpi=160)
    print(f"\nwrote {a.out}")
    Path("out/pressure.json").write_text(json.dumps(
        {k: {str(l): v for l, v in d.items()} for k, d in results.items()}, indent=2))


if __name__ == "__main__":
    main()
