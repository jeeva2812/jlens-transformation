"""Do eigenvectors steer better than singular vectors?

Eigenvectors already read better -- 39.6% clear an axis probe against 20.8% for
SVD u. Steering is the other half, and there is a reason to expect eigen to win
that too: J v = lambda v, so the vector you INJECT and the vector whose readout
you TEST are literally the same direction. For a singular pair they are two
different vectors (v in, u out), which is what made the type error possible.

Same protocol as the corrected assay so the numbers are comparable: the token
pair is the argmax of the direction's own +/- readout, a direction counts if the
pair moves by more than 0.5 nats AND more than 4x a random direction of equal
norm, and a repetition rate is recorded so an obliterated model cannot score as
a success.

If eigen wins both halves, "use the eigendecomposition" becomes a clean
recommendation rather than a preference.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import AddDir, NEUTRAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path("out/steer_eigen.json"))
    a = ap.parse_args()
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    blocks, norm = _find_blocks_and_norm(m)
    W_U = m.get_output_embeddings().weight.detach()
    target = blob["target"]

    def logratio(pos, neg):
        tot = 0.0
        for p in NEUTRAL:
            e = tok(p, return_tensors="pt")
            with torch.no_grad():
                lp = torch.log_softmax(m(**e).logits[0, -1].float(), -1)
            tot += float(lp[pos] - lp[neg])
        return tot / len(NEUTRAL)

    g = torch.Generator().manual_seed(0)
    rows = []
    print(f"{'layer':>5} {'family':<8}{'d':>3}{'|lam| or sig':>13}"
          f"{'+pole':<13}{'-pole':<13}{'shift':>7}{'rand':>7}")
    print("-" * 74)
    for l in a.layers:
        if l not in blob["J"] or l >= target:
            continue
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        w, V = torch.linalg.eig(J)
        eo = torch.argsort(w.abs(), descending=True)[:a.ndirs]
        ids0 = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(m, [l], target) as cap:
            with torch.no_grad():
                m(input_ids=ids0, attention_mask=torch.ones_like(ids0),
                  use_cache=False)
            hn = float(cap.h[l][0].norm(dim=-1).mean())

        fam = [("eigen", [(V[:, j].real, float(w[j].abs())) for j in eo.tolist()]),
               ("SVD v", [(Vh[i], float(S[i])) for i in range(a.ndirs)])]
        for name, dirs in fam:
            for i, (d, mag) in enumerate(dirs):
                d = d / d.norm()
                # the readout of what this direction PRODUCES at the target
                t = (J @ d).to(W_U.dtype)
                with torch.no_grad():
                    pp = torch.softmax(norm(t) @ W_U.T, -1)
                    pn = torch.softmax(norm(-t) @ W_U.T, -1)
                pos, neg = int(pp.argmax()), int(pn.argmax())
                if pos == neg:
                    continue
                base = logratio(pos, neg)
                with AddDir(m, l, d, a.alpha * hn):
                    sh = logratio(pos, neg) - base
                r = torch.randn(J.shape[0], generator=g)
                with AddDir(m, l, r, a.alpha * hn):
                    rd = logratio(pos, neg) - base
                rows.append({"layer": l, "family": name, "i": i, "mag": mag,
                             "shift": sh, "rand": rd,
                             "pos": tok.decode(pos), "neg": tok.decode(neg)})
                print(f"{l:>5} {name:<8}{i:>3}{mag:>13.2f}"
                      f"{tok.decode(pos)[:12]:<13}{tok.decode(neg)[:12]:<13}"
                      f"{sh:>7.2f}{rd:>7.2f}", flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    import statistics as st
    def passes(r):
        return abs(r["shift"]) > 0.5 and abs(r["shift"]) > 4 * abs(r["rand"])
    print(f"\n{'family':<8}{'n':>4}{'pass':>8}{'rate':>8}{'median |shift|':>16}")
    print("-" * 46)
    for name in ["eigen", "SVD v"]:
        rr = [r for r in rows if r["family"] == name]
        p = sum(passes(r) for r in rr)
        print(f"{name:<8}{len(rr):>4}{p:>8}{p/len(rr)*100:>7.1f}%"
              f"{st.median(abs(r['shift']) for r in rr):>16.2f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
