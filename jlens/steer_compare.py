"""Does gain or occupancy drive steerability? Four direction families, one test.

The occupancy result makes this a sharp prediction rather than a survey. J's
leading singular directions have high gain and near-zero occupancy -- at layer
20 they carry ~740x less activation energy than the trailing ones. Two stories
explain why they steer anyway:

  gain      they work BECAUSE the transport amplifies them, and occupancy is
            irrelevant to an intervention that forces the model somewhere new
  occupancy they would work BETTER if they were also directions the model uses,
            and we are leaving effect size on the table

Four families, all in layer-l space so all are injectable:

  SVD(J)        v_i                       max gain, ignores the data
  PCA(h)        eigenvectors of Cov(h)    max occupancy, ignores the transport
  whitened      SVD(J C^{1/2}) pulled back  both: amplified AND occupied
  random        control

PCA of Jh is deliberately NOT one of them: Jh lives in target space, so its
directions cannot be injected at layer l. The whitened family is the
type-correct way to ask that question -- it finds directions the transport
amplifies among those the data actually populates.

Every direction is scored the same way: its predicted token pair is the argmax
of readout(J d) and readout(-J d), then it is injected at layer l and the pair's
log-ratio is measured against a random direction of the same norm. Steering uses
the direction itself, never u -- injecting u is the type error this project
already made once.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir, NEUTRAL

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/ft/J_step0.pt"))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--ntext", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("out/steer_compare.json"))
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    blocks, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = [ds[i]["text"] for i in range(a.ntext)]

    def logratio(pos, neg):
        tot = 0.0
        for p in NEUTRAL:
            enc = tok(p, return_tensors="pt")
            with torch.no_grad():
                lp = torch.log_softmax(model(**enc).logits[0, -1].float(), -1)
            tot += float(lp[pos] - lp[neg])
        return tot / len(NEUTRAL)

    g = torch.Generator().manual_seed(0)
    rows = []
    for l in a.layers:
        H = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
            with _MultiCapture(model, [l], 28) as cap:
                with torch.no_grad():
                    model(**enc, use_cache=False)
                H.append(cap.h[l][0].detach().float())
        H = torch.cat(H, 0); H = H - H.mean(0, keepdim=True)
        d_model = H.shape[1]
        C = (H.T @ H) / H.shape[0]
        ev, EV = torch.linalg.eigh(C)
        ev = ev.clamp(min=0)
        Ch = EV @ torch.diag((ev + a.ridge * ev.max()).sqrt()) @ EV.T   # C^{1/2}
        J = blob["J"][l].float()
        totE = H.pow(2).sum()

        fam = {}
        fam["SVD(J)"] = torch.linalg.svd(J)[2][:a.ndirs]                 # v_i rows
        fam["PCA(h)"] = EV[:, torch.argsort(ev, descending=True)[:a.ndirs]].T
        Wm = torch.linalg.svd(J @ Ch)[2][:a.ndirs]                       # whitened
        fam["whitened"] = torch.nn.functional.normalize(Wm @ Ch.T, dim=-1)
        fam["random"] = torch.nn.functional.normalize(
            torch.randn(a.ndirs, d_model, generator=g), dim=-1)

        hn = None
        ids0 = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(model, [l], blob.get("target", 28)) as cap:
            with torch.no_grad():
                model(input_ids=ids0, attention_mask=torch.ones_like(ids0),
                      use_cache=False)
            hn = float(cap.h[l][0].norm(dim=-1).mean())

        print(f"\n{'='*72}\nLAYER {l}\n{'='*72}")
        print(f"{'family':<10} {'d':>3} {'gain':>7} {'occupancy':>10} "
              f"{'+pole':<13}{'-pole':<13}{'shift':>7}{'rand':>7}")
        print("-" * 72)
        for name, D in fam.items():
            for i in range(a.ndirs):
                d = D[i] / D[i].norm()
                gain = (J @ d).norm().item()
                occ = ((H @ d).pow(2).sum() / totE).item() * d_model   # x random
                with torch.no_grad():
                    t = (J @ d).to(W_U.dtype)
                    pp = torch.softmax(norm(t) @ W_U.T, -1)
                    pn = torch.softmax(norm(-t) @ W_U.T, -1)
                pos, neg = int(pp.argmax()), int(pn.argmax())
                if pos == neg:
                    continue
                base = logratio(pos, neg)
                with AddDir(model, l, d, a.alpha * hn):
                    sh = logratio(pos, neg) - base
                r = torch.randn(d_model, generator=g)
                with AddDir(model, l, r, a.alpha * hn):
                    rd = logratio(pos, neg) - base
                rows.append({"layer": l, "family": name, "i": i,
                             "gain": gain, "occ": occ, "shift": sh, "rand": rd,
                             "pos": tok.decode(pos), "neg": tok.decode(neg)})
                print(f"{name:<10} {i:>3} {gain:>7.2f} {occ:>9.2f}x "
                      f"{tok.decode(pos)[:12]:<13}{tok.decode(neg)[:12]:<13}"
                      f"{sh:>7.2f}{rd:>7.2f}", flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
