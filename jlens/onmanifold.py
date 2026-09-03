"""Two things the raw SVD of J cannot tell us.

WHY THE RAW SVD LOOKS LIKE GIBBERISH. SVD(J) finds directions the transport
amplifies. That is a property of the matrix and says nothing about whether those
directions ever occur. If layer-l activations never point that way, J can still
amplify it enormously -- so the top singular vectors may be off-manifold: real
geometry, describing perturbations the model never experiences.

Test 1 fixes that by weighting with the data. Instead of decomposing J alone, we
decompose what J does to REAL activations: PCA of {J h} over actual text. Those
directions are both amplified by the transport and actually populated. If they
read out cleanly where the raw singular directions did not, that explains the
gibberish and gives the better analysis.

Test 2 is the causal check that everything so far has been missing. Every result
we have is correlational -- the lens SAYS a direction means "Paris"; we have
never checked the model agrees. Steering asks exactly that: add the lens
direction for a token to the residual and see whether that token's probability
rises, against a random direction of the same norm.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"

CORPUS = [
    "The library opened in 1897 and has since expanded three times over the years.",
    "Rainfall in the region peaks between June and September in most years.",
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0))",
    "She walked to the market, bought bread, and returned home before noon.",
    "The capital of the country where Mount Fuji is located is Tokyo.",
    "In 1969 the Apollo 11 mission landed the first humans on the Moon.",
    "Photosynthesis converts light energy into chemical energy stored as glucose.",
    "The committee met on Tuesday to discuss the proposed budget revisions.",
]

# (prompt, token we will try to steer toward) -- token must be plausible-but-not-certain
STEER = [
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("The capital city of Japan is", " Tokyo"),
    ("Water freezes at zero degrees", " Celsius"),
    ("The largest planet in our solar system is", " Jupiter"),
]


class AddDir:
    """Add alpha * unit direction to the residual leaving block `layer`."""
    def __init__(self, model, layer, d, alpha):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0)
        self.a = alpha
        self.h = blocks[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        add = (self.a * self.d).to(t.device, t.dtype)
        t2 = t + add
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])

    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--layer", type=int, default=16)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    J = blob["J"][a.layer].float()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight

    def read(vec, k=6):
        with torch.no_grad():
            p = torch.softmax((norm(vec.to(dev, W_U.dtype)) @ W_U.T).float(), -1)
        v, i = p.topk(k)
        return [(tok.decode(j), float(x)) for x, j in zip(v, i)]

    # ---------- TEST 1: on-manifold directions ----------
    print("=" * 76)
    print("TEST 1  raw SVD of J   vs   PCA of J applied to REAL activations")
    print("=" * 76)
    hs = []
    for t in CORPUS:
        ids = tok(t, return_tensors="pt")["input_ids"].to(dev)
        with _MultiCapture(model, [a.layer], 30) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hs.append(cap.h[a.layer][0].float().cpu())
    H = torch.cat(hs, 0)                                  # (T_total, d_model)
    print(f"collected {H.shape[0]} real activations at layer {a.layer}")

    T = H @ J.T                                           # transported real activations
    T = T - T.mean(0, keepdim=True)
    Ut, St, Vt = torch.linalg.svd(T, full_matrices=False)

    Ur, Sr, Vr = torch.linalg.svd(J)

    print(f"\n{'':<4}{'RAW SVD of J':<40}{'PCA of J h (on-manifold)':<40}")
    print("-" * 84)
    for i in range(5):
        raw = ", ".join(f"{w!r}" for w, _ in read(Ur[:, i], 3))
        onm = ", ".join(f"{w!r}" for w, _ in read(Vt[i], 3))
        print(f"{i:<4}{raw[:38]:<40}{onm[:38]:<40}")

    # how much of a real transported activation lives in the raw top directions?
    proj_raw = (T @ Ur[:, :64]).pow(2).sum() / T.pow(2).sum()
    proj_onm = (T @ Vt[:64].T).pow(2).sum() / T.pow(2).sum()
    print(f"\nfraction of real transported activation captured by top-64 directions:")
    print(f"   raw SVD of J        {float(proj_raw):.4f}")
    print(f"   PCA of J h          {float(proj_onm):.4f}   <- by construction the max")
    print("   If raw is much lower, the raw singular directions are largely")
    print("   off-manifold, which is exactly why they read as gibberish.")

    # ---------- TEST 2: steering ----------
    print("\n" + "=" * 76)
    print("TEST 2  is the readout CAUSAL? steer along a lens direction")
    print("=" * 76)
    g = torch.Generator().manual_seed(0)
    hnorm = float(H.norm(dim=-1).mean())
    print(f"mean activation norm at layer {a.layer}: {hnorm:.2f}")
    print("alpha is a MULTIPLE of that, so 1.0 means adding a vector as large as")
    print("the activation itself. Coherence = loss on unrelated held-out text; if")
    print("it blows up, the model is destroyed and P(token) is meaningless.\n")

    # unrelated text, to check the model still works at each steering strength
    coh_ids = tok("The committee met on Tuesday to discuss the budget revisions.",
                  return_tensors="pt")["input_ids"].to(dev)

    def coherence():
        with torch.no_grad():
            o = model(input_ids=coh_ids, attention_mask=torch.ones_like(coh_ids),
                      use_cache=False)
            return float(torch.nn.functional.cross_entropy(
                o.logits[0, :-1].float(), coh_ids[0, 1:]))

    base_coh = coherence()
    rows = []
    for alpha in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        amt = alpha * hnorm
        lens_p, rand_p, lens_c, rand_c = [], [], [], []
        for prompt, target in STEER:
            tid = tok.encode(target, add_special_tokens=False)[0]
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)

            def p_target():
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                              use_cache=False)
                return float(torch.softmax(o.logits[0, -1].float(), -1)[tid])

            if alpha == 0.05:
                rows.append({"prompt": prompt, "token": target, "base": p_target()})
            d = (W_U[tid].detach().cpu().float() @ J)
            with AddDir(model, a.layer, d, amt):
                lens_p.append(p_target()); 
            with AddDir(model, a.layer, d, amt):
                lens_c.append(coherence())
            rnd = torch.randn(J.shape[0], generator=g)
            with AddDir(model, a.layer, rnd, amt):
                rand_p.append(p_target())
            with AddDir(model, a.layer, rnd, amt):
                rand_c.append(coherence())
        m = lambda x: sum(x)/len(x)
        print(f"  alpha {alpha:>5.2f}   P(token) lens {m(lens_p):.4f}  rand {m(rand_p):.4f}"
              f"   | coherence loss  base {base_coh:.3f}  lens {m(lens_c):.3f}  rand {m(rand_c):.3f}")

    print(f"\n  unsteered P(token): " +
          ", ".join(f"{r['token'].strip()} {r['base']:.3f}" for r in rows))
    print("\n  The informative row is the largest alpha where coherence is still")
    print("  close to baseline. Above that the model is being overwritten, not steered.")
    Path("out/onmanifold.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
