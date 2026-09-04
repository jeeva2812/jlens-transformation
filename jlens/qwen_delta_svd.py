"""dJ SVD for the four Qwen2.5-0.5B fine-tunes, both sides, both prompt sets.

Four adapters on one base -- three published EM organisms and our benign control
-- with J saved under two prompt distributions. That makes it the cleanest
available comparison of what different fine-tunes do to the transport, holding
base model, LoRA rank and target modules fixed.

Every readout carries a z-score against a random-direction null at the same
layer, because a top-k token list on its own has already been shown to be
uninformative here: 29% of random directions pass a sign-consistency test that
looks decisive.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
NAMES = ["medical", "financial", "sports", "control"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["out/em05", "out/em05_emprompts"])
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("out/qwen_delta_svd.json"))
    a = ap.parse_args()

    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
    W_U = m.get_output_embeddings().weight.detach().float()
    norm = m.model.norm
    tok = AutoTokenizer.from_pretrained(BASE)
    del m

    def logits(v):
        with torch.no_grad():
            return norm(v.float().unsqueeze(0)).squeeze(0) @ W_U.T

    def read(v, k):
        lg = logits(v)
        return [tok.decode([i]) for i in lg.topk(k).indices.tolist()], lg

    rec = {}
    for d in a.dirs:
        tag = Path(d).name
        J = {n: torch.load(f"{d}/J_{n}.pt", map_location="cpu",
                           weights_only=False) for n in ["base"] + NAMES}
        rec[tag] = {}
        print(f"\n{'#'*78}\n# {tag}\n{'#'*78}", flush=True)
        for l in a.layers:
            Jb = J["base"]["J"][l].float()
            g = torch.Generator().manual_seed(0)
            null = torch.stack([logits(Jb @ torch.randn(Jb.shape[0], generator=g))
                                for _ in range(a.n_null)])
            nm, ns = null.mean(0), null.std(0) + 1e-6
            print(f"\n{'='*74}\nLAYER {l}\n{'='*74}")
            rec[tag][l] = {}
            for n in NAMES:
                dJ = J[n]["J"][l].float() - Jb
                rel = (dJ.norm() / (Jb - torch.eye(Jb.shape[0])).norm()).item()
                U, S, Vh = torch.linalg.svd(dJ)
                share = S.pow(2) / S.pow(2).sum()
                print(f"\n--- {n}   ||dJ||/||J-I|| = {rel:.3f} ---")
                rows = []
                for i in range(a.n_dirs):
                    resp, lg = read(Jb @ Vh[i], a.k)
                    sends, _ = read(U[:, i], a.k)
                    z = ((lg - nm) / ns).topk(a.k).values.mean().item()
                    print(f"  d{i} ({share[i]*100:4.1f}%) z={z:5.1f}  responds to -> "
                          + " ".join(repr(t) for t in resp))
                    print(f"  {'':21}  sends       -> "
                          + " ".join(repr(t) for t in sends))
                    rows.append({"i": i, "share": share[i].item(), "z": z,
                                 "responds_to": resp, "sends": sends})
                rec[tag][l][n] = {"rel": rel, "dirs": rows}
        del J
    a.out.write_text(json.dumps(rec, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
