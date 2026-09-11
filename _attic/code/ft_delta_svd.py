"""dJ SVD across a fine-tuning TRAJECTORY, not just its endpoint.

13 checkpoints of SmolLM2-135M on insecure code, at two learning rates, with J
saved at each. Every dJ result so far compared two models; this asks WHEN during
fine-tuning each direction of the change appears, and whether the two learning
rates find the same directions.

Both sides are read, since reading only U was an earlier mistake:
    U[:, i]  where the change SENDS things     -- target space, W_U reads it
    V[i]     what the change RESPONDS TO       -- layer-l space, pushed through
                                                  the BASE J to be readable

A LABEL IS NOT A MEASUREMENT. Validating the "British orthography" label on the
Olmo run showed that 28% of RANDOM directions also separate all 20 US/UK
spelling pairs in the same direction -- because themed token sets are themselves
correlated in unembedding space. So any theme read off a top-k list here is a
hypothesis. This file reports the top tokens AND, for each direction, how
distinctive its readout is against a random-direction null, so a reader can see
which labels are worth anything.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out/ft"))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[50, 100, 200, 400, 600])
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (a.dir / "delta_svd_traj.json")

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(MODEL)
    del m

    def read(v, k):
        with torch.no_grad():
            lg = norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()], lg

    base = torch.load(a.dir / "J_step0.pt", map_location="cpu", weights_only=False)
    layers = [l for l in a.layers if l in base["J"]]
    rec = {}

    for l in layers:
        Jb = base["J"][l].float()
        # random-direction null for the input-side readout at this layer
        g = torch.Generator().manual_seed(0)
        null = torch.stack([norm((Jb @ torch.randn(Jb.shape[0], generator=g))
                                 .float().unsqueeze(0)).squeeze(0) @ W_U.T
                            for _ in range(a.n_null)])
        print(f"\n{'='*76}\nLAYER {l}\n{'='*76}", flush=True)
        rec[l] = {}
        prev = None
        for st in a.steps:
            f = a.dir / f"J_step{st}.pt"
            if not f.exists():
                continue
            Jt = torch.load(f, map_location="cpu", weights_only=False)["J"][l].float()
            dJ = Jt - Jb
            rel = (dJ.norm() / (Jb - torch.eye(Jb.shape[0])).norm()).item()
            U, S, Vh = torch.linalg.svd(dJ)
            share = S.pow(2) / S.pow(2).sum()
            # does the top subspace at this step match the previous step's?
            cont = None
            if prev is not None:
                cont = (prev.T @ U[:, :8]).pow(2).sum().item() / 8
            prev = U[:, :8].clone()

            print(f"\n--- step {st}   ||dJ||/||J-I|| = {rel:.3f}"
                  + (f"   subspace continuity vs prev = {cont:.3f}" if cont else ""))
            rows = []
            for i in range(a.n_dirs):
                sends, _ = read(U[:, i], a.k)
                resp, lg = read(Jb @ Vh[i], a.k)
                # distinctiveness: how far this readout sits from the random null
                z = ((lg - null.mean(0)) / (null.std(0) + 1e-6)).topk(a.k).values.mean().item()
                print(f"  d{i} ({share[i]*100:4.1f}%) z={z:5.1f}  responds to -> "
                      + " ".join(repr(t) for t in resp))
                print(f"  {'':21}  sends       -> "
                      + " ".join(repr(t) for t in sends))
                rows.append({"i": i, "share": share[i].item(), "z": z,
                             "responds_to": resp, "sends": sends})
            rec[l][st] = {"rel": rel, "continuity": cont, "dirs": rows}

    out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
