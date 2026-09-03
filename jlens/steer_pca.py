"""Do PCA directions steer as well as SVD directions?

All our steering so far used singular directions of J -- what the TRANSPORT
amplifies. But three decompositions were on the table and only one was ever
tested causally:

  SVD of J     directions the transport amplifies      <- tested, 27/60 steer
  PCA of J h   where transported real data varies      <- never steered
  PCA of h     where the residual stream itself varies <- never steered

The comparison matters because SVD of J is data-blind: two-thirds of its top
directions are off-manifold. PCA directions are by construction ones the model
actually visits, so they might steer BETTER despite reading no more cleanly.
Or they might steer worse, if being amplified by J is what makes a direction
causally potent. Either answer says something.

Same protocol as the SVD assay: each direction's own readout supplies the
prediction, and four controls -- symmetry, matched random direction, coherence
on unrelated text, off-topic distribution drift.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "HuggingFaceTB/SmolLM2-135M"
NEUTRAL = ["The report was finished on Tuesday and", "After the long walk home,",
           "In the middle of the afternoon", "The old building on the corner",
           "When the meeting finally ended,"]
OFFTOPIC = ["The capital of France is", "Two plus two equals",
            "The chemical symbol for water is"]


class AddDir:
    def __init__(self, model, layer, d, alpha):
        blocks, _ = _find_blocks_and_norm(model)
        self.d = torch.nn.functional.normalize(d.float(), dim=0); self.a = alpha
        self.h = blocks[layer].register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t2 = t + (self.a * self.d).to(t.device, t.dtype)
        return t2 if torch.is_tensor(out) else (t2,) + tuple(out[1:])
    def __enter__(self): return self
    def __exit__(self, *e): self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--ndirs", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.01)
    a = ap.parse_args()

    blob = torch.load("out/Jall_smollm2.pt", map_location="cpu", weights_only=False)
    J = blob["J"][a.layer].float(); target = blob["target"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    hs = []
    for i in range(30):
        ids = tok(ds[i]["text"], return_tensors="pt", truncation=True,
                  max_length=128)["input_ids"]
        with _MultiCapture(model, [a.layer], target) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hs.append(cap.h[a.layer][0].float())
    H = torch.cat(hs, 0)
    hn = float(H.norm(dim=-1).mean())

    # three sets of candidate directions
    U, S, _ = torch.linalg.svd(J)
    v = H.var(0); outl = v.topk(3).indices
    Hf = H.clone(); Hf[:, outl] = 0.0
    _, _, Vh_h = torch.linalg.svd(Hf - Hf.mean(0, keepdim=True), full_matrices=False)
    T = H @ J.T
    _, _, Vh_t = torch.linalg.svd(T - T.mean(0, keepdim=True), full_matrices=False)

    sets = {"SVD of J": U[:, :a.ndirs].T,
            "PCA of h": Vh_h[:a.ndirs],
            "PCA of J h": Vh_t[:a.ndirs]}

    coh = tok("The committee met on Tuesday to discuss the budget revisions.",
              return_tensors="pt")["input_ids"]
    def C():
        with torch.no_grad():
            o = model(input_ids=coh, attention_mask=torch.ones_like(coh), use_cache=False)
        return float(torch.nn.functional.cross_entropy(o.logits[0, :-1], coh[0, 1:]))
    def LR(pos, neg):
        t = 0.0
        for pr in NEUTRAL:
            ids = tok(pr, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            t += float(lp[pos] - lp[neg])
        return t / len(NEUTRAL)
    def dist():
        out = []
        for pr in OFFTOPIC:
            ids = tok(pr, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            out.append(torch.softmax(o.logits[0, -1].float(), -1))
        return out

    base_coh, base_off = C(), dist()
    g = torch.Generator().manual_seed(0)
    print(f"layer {a.layer}, alpha {a.alpha} x activation norm ({hn:.1f}), "
          f"baseline coherence {base_coh:.3f}\n")
    results = {}
    for name, D in sets.items():
        print(f"{name}")
        print(f"  {'dir':>4} {'+pole':<15}{'-pole':<15}{'shift+':>8}{'shift-':>8}"
              f"{'rand':>7}{'coh':>7}{'drift':>8}")
        rows = []
        for i in range(a.ndirs):
            d = D[i]
            with torch.no_grad():
                pos = int(torch.softmax(norm(d) @ W_U.T, -1).argmax())
                neg = int(torch.softmax(norm(-d) @ W_U.T, -1).argmax())
            if pos == neg:
                continue
            b = LR(pos, neg)
            with AddDir(model, a.layer, d, a.alpha*hn):
                up, c, off = LR(pos, neg), C(), dist()
            with AddDir(model, a.layer, -d, a.alpha*hn):
                dn = LR(pos, neg)
            with AddDir(model, a.layer, torch.randn(J.shape[0], generator=g), a.alpha*hn):
                rd = LR(pos, neg)
            drift = sum(float((x-y).abs().sum())/2 for x, y in zip(off, base_off))/len(off)
            r = {"dir": i, "pos": tok.decode(pos), "neg": tok.decode(neg),
                 "shift_plus": up-b, "shift_minus": dn-b, "shift_rand": rd-b,
                 "coh": c, "drift": drift}
            rows.append(r)
            print(f"  {i:>4} {tok.decode(pos)[:14]:<15}{tok.decode(neg)[:14]:<15}"
                  f"{r['shift_plus']:>+8.2f}{r['shift_minus']:>+8.2f}"
                  f"{r['shift_rand']:>+7.2f}{c:>7.2f}{drift:>8.3f}")
        ok = [r for r in rows if r["shift_plus"] > .5 and r["shift_minus"] < -.5
              and abs(r["coh"]-base_coh) < .3*base_coh
              and abs(r["shift_rand"]) < .3*r["shift_plus"]]
        results[name] = rows
        print(f"  -> {len(ok)}/{len(rows)} pass all four controls\n")
    Path("out/steer_pca.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
