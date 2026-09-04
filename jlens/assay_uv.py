"""Head-to-head: steering with u (what we did) vs steering with v (type-correct).

THE BUG. J_l = dh_target/dh_l maps layer-l space to target space, so
J = U S V^T has U in TARGET space and V in LAYER-l space. The original assay
took d = U[:, i], read it through W_U (correct -- U is in target space), and
then INJECTED IT AT LAYER l (not correct -- that is layer-l space). The
type-correct intervention is to inject v_i, which the transport carries to
sigma_i * u_i, whose readout is exactly the token pair being tested.

WHY IT MATTERED LESS THAN IT SHOULD HAVE. cos(u_i, v_i) is ~0.30 in early
layers and ~0.94 by layer 26, tracking how close J is to the identity. Near the
target layer u and v nearly coincide, so injecting u was almost right there --
and that is where the original assay found its successes.

THE PREDICTION THIS MAKES. If the depth profile ("directions only become
causally live in the second half") is an artefact of the substitution, steering
with v should rescue the early layers. If the profile is real, v will not help
and the finding survives. Both directions are tested on the SAME directions with
the SAME token pair and the SAME controls, so the comparison is like for like.
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
    ap.add_argument("--ndirs", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("out/assay_uv.json"))
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    blocks, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()
    layers = a.layers or blob["layers"]
    target = blob["target"]

    def logratio(prompts, pos, neg):
        tot = 0.0
        for p in prompts:
            enc = tok(p, return_tensors="pt")
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            lp = torch.log_softmax(lg.float(), -1)
            tot += float(lp[pos] - lp[neg])
        return tot / len(prompts)

    g = torch.Generator().manual_seed(0)
    rows = []
    print(f"alpha {a.alpha} x activation norm | {len(NEUTRAL)} neutral prompts\n")
    print(f"{'layer':>5} {'dir':>4} {'cos(u,v)':>9} {'+pole':<13}{'-pole':<13}"
          f"{'shift[u]':>9}{'shift[v]':>9}{'rand':>7}")
    print("-" * 76)
    for l in layers:
        if l >= target:
            continue
        J = blob["J"][l].float()
        U, S, Vh = torch.linalg.svd(J)
        ids0 = tok(NEUTRAL[0], return_tensors="pt")["input_ids"]
        with _MultiCapture(model, [l], target) as cap:
            with torch.no_grad():
                model(input_ids=ids0, attention_mask=torch.ones_like(ids0),
                      use_cache=False)
            hn = float(cap.h[l][0].norm(dim=-1).mean())

        for di in range(a.ndirs):
            u, v = U[:, di], Vh[di]
            with torch.no_grad():
                pp = torch.softmax(norm(u.to(W_U.dtype)) @ W_U.T, -1)
                pn = torch.softmax(norm(-u.to(W_U.dtype)) @ W_U.T, -1)
            pos, neg = int(pp.argmax()), int(pn.argmax())
            if pos == neg:
                continue
            base = logratio(NEUTRAL, pos, neg)
            with AddDir(model, l, u, a.alpha * hn):
                su = logratio(NEUTRAL, pos, neg) - base
            with AddDir(model, l, v, a.alpha * hn):
                sv = logratio(NEUTRAL, pos, neg) - base
            r = torch.randn(J.shape[0], generator=g)
            with AddDir(model, l, r, a.alpha * hn):
                sr = logratio(NEUTRAL, pos, neg) - base
            cuv = abs(torch.dot(u, v).item())
            rows.append({"layer": l, "dir": di, "cos_uv": cuv,
                         "pos": tok.decode(pos), "neg": tok.decode(neg),
                         "shift_u": su, "shift_v": sv, "shift_rand": sr})
            print(f"{l:>5} {di:>4} {cuv:>9.3f} {tok.decode(pos)[:12]:<13}"
                  f"{tok.decode(neg)[:12]:<13}{su:>9.2f}{sv:>9.2f}{sr:>7.2f}",
                  flush=True)

    a.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
