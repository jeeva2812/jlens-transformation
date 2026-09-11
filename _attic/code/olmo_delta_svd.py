"""SVD of dJ across Olmo's training. What did each PHASE change about the transport?

Every dJ result so far compared a base model to a fine-tune. But we have full
Jacobians at 11 checkpoints spanning all three Olmo 3 stages, so the same
decomposition answers a question no fine-tune can: what does each phase of
*pretraining and post-training* change about how the model moves information?

    dJ = J(t2) - J(t1) = U S V^T

BOTH SIDES ARE READ, because they answer different questions and reading only U
was an earlier mistake:

    U[:, i]   where the change SENDS things   -- target space, W_U reads it directly
    V[:, i]   what the change RESPONDS TO     -- layer-l space, must be pushed
                                                 through the base J first

The most interesting row is the pretraining -> mid-training transition. That is
post-training, the phase that installs instruction-following and safety
behaviour, and it is the one phase whose weight update we can read as tokens
here rather than infer.

CONTROLS. Share of total dJ energy per direction, so a direction carrying 0.2%
is not read as if it mattered. And a random direction through the same readout,
because "the top tokens look thematic" is the easiest way to fool yourself --
J-Lens readouts are anisotropic and will hand you plausible-looking junk.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer

# (label, from_revision, to_revision)
PHASES = [
    ("first 2k steps",              "stage1-step0",       "stage1-step2000"),
    ("early pretraining 8k->128k",  "stage1-step8000",    "stage1-step128000"),
    ("late pretraining 512k->end",  "stage1-step512000",  "stage1-step1413814"),
    ("MID-TRAINING (post-train)",   "stage1-step1413814", "stage2-step47684"),
    ("long-context extension",      "stage2-step47684",   "stage3-step5000"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--n-dirs", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="tokens per direction")
    ap.add_argument("--out", type=Path, default=Path("out/olmo_delta_svd.json"))
    a = ap.parse_args()

    head = torch.load("out/readout_head.pt", map_location="cpu", weights_only=False)
    W_U = head["W_U"].float()
    w = head["norm_state"]["weight"].float(); eps = head["norm_eps"]
    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")
    rms = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def top(v, k):
        with torch.no_grad():
            p = torch.softmax(rms(v.float()) @ W_U.T, -1)
        val, idx = p.topk(k)
        return [tok.decode(i) for i in idx]

    # base transport for reading the input side; the final model is the reference
    base = torch.load("out/ckpt/J_main.pt", map_location="cpu", weights_only=False)

    g = torch.Generator().manual_seed(0)
    rec = {}
    for label, r1, r2 in PHASES:
        A = torch.load(f"out/ckpt/J_{r1}.pt", map_location="cpu", weights_only=False)
        B = torch.load(f"out/ckpt/J_{r2}.pt", map_location="cpu", weights_only=False)
        print(f"\n{'='*78}\n{label}   ({r1} -> {r2})\n{'='*78}", flush=True)
        rec[label] = {}
        for l in a.layers:
            J1, J2 = A["J"][l].float(), B["J"][l].float()
            dJ = J2 - J1
            rel = (dJ.norm() / (J1 - torch.eye(J1.shape[0])).norm()).item()
            U, S, Vh = torch.linalg.svd(dJ)
            share = (S.pow(2) / S.pow(2).sum())
            Jb = base["J"][l].float()

            print(f"\n--- layer {l}   ||dJ||/||J-I|| = {rel:.3f} ---")
            rows = []
            for i in range(a.n_dirs):
                out_side = top(U[:, i], a.k)
                in_side = top(Jb @ Vh[i], a.k)          # push V through base J
                print(f"  d{i} ({share[i]*100:4.1f}%)  sends -> "
                      f"{' '.join(repr(t) for t in out_side)}")
                print(f"  {'':13}responds to -> "
                      f"{' '.join(repr(t) for t in in_side)}")
                rows.append({"i": i, "share": share[i].item(),
                             "sends": out_side, "responds_to": in_side})
            v = torch.randn(J1.shape[0], generator=g)
            print(f"  RANDOM control -> {' '.join(repr(t) for t in top(v, a.k))}")
            rec[label][l] = {"rel": rel, "dirs": rows,
                             "random": top(v, a.k)}
        del A, B

    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
