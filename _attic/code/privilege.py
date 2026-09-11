"""Is the model's own basis privileged? Weight-space only, no forward passes.

For a set of n direction vectors A (d x n, unit columns) spanning subspace S,
compare four bases of the SAME S:
  raw        A                      Gram = AtA
  svd        left sing. vecs of A   Gram = I
  gram_null  Q H R  (H ~ Haar)      Gram = AtA   <- identical geometry to raw
  orth_null  Q H    (H ~ Haar)      Gram = I

Score each direction by NPMI topic coherence of its top-10 logit-lens tokens.
gram_null holds subspace, norms, pairwise angles and conditioning fixed, so a
raw-vs-gram_null gap can only be about which directions those are.

Calibration: inside one attention head of an MHA model, W_V -> R W_V and
W_O -> W_O R^T is an exact function-preserving symmetry, so that arm has a
provable expected gap of zero.
"""
from __future__ import annotations
import argparse, glob, json, math, os
from pathlib import Path
import torch
from safetensors import safe_open
from jlens.corpus_npmi import NPMI, build as build_npmi

HUB = Path.home() / ".cache/huggingface/hub"


class Weights:
    def __init__(self, model_id: str):
        d = HUB / ("models--" + model_id.replace("/", "--"))
        snaps = sorted(d.glob("snapshots/*"))
        assert snaps, f"not on disk: {model_id}"
        self.dir = snaps[-1]
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.map = json.loads(idx.read_text())["weight_map"]
        else:
            f = next(self.dir.glob("*.safetensors")).name
            with safe_open(self.dir / f, framework="pt") as h:
                self.map = {k: f for k in h.keys()}
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self._h: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        f = self.map[key]
        if f not in self._h:
            self._h[f] = safe_open(self.dir / f, framework="pt")
        return self._h[f].get_tensor(key).float()

    def has(self, key: str) -> bool:
        return key in self.map


def haar(n: int, gen: torch.Generator) -> torch.Tensor:
    z = torch.randn(n, n, generator=gen)
    q, r = torch.linalg.qr(z)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def make_bases(A: torch.Tensor, n_draws: int, gen: torch.Generator):
    """A (d,n) unit columns -> dict name -> (d, m) with m a multiple of n."""
    d, n = A.shape
    Q, R = torch.linalg.qr(A)
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    out = {"raw": A, "svd": U}
    # reparam_null: A H is exactly the family of alternative weight settings
    #   reachable by the rotation W_in -> H^T W_in, W_out -> W_out H.
    #   For an MHA head that family is function-preserving (the model cannot
    #   distinguish its members); for an MLP it is not (SiLU acts elementwise).
    # gram_null: Q H R has the SAME Gram matrix as A -- same norms, same
    #   pairwise angles, same conditioning -- but a random identity.
    # orth_null: an arbitrary orthonormal basis of the same subspace.
    for name, f in (("reparam_null", lambda H: A @ H),
                    ("gram_null", lambda H: Q @ H @ R),
                    ("orth_null", lambda H: Q @ H)):
        cols = [f(haar(n, gen)) for _ in range(n_draws)]
        M = torch.cat(cols, dim=1)
        out[name] = M / M.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return out


# ---------------------------------------------------------------- arms

def arm_matrices(w: Weights, n: int, gen: torch.Generator, n_groups: int):
    """yield (arm, group_label, A (d,n))."""
    c = w.cfg
    d = c["hidden_size"]; nl = c["num_hidden_layers"]
    nh = c["num_attention_heads"]; dh = d // nh
    moe = "num_experts" in c
    layers = torch.linspace(1, nl - 2, n_groups).round().long().tolist()

    # ceiling calibration: each row of W_U/W_E *is* one token, so its readout is
    # maximally "privileged"; a rotation mixes 64 unrelated tokens.
    lm = "lm_head.weight" if w.has("lm_head.weight") else "model.embed_tokens.weight"
    for k, key in ():
        if not w.has(key):
            continue
        M = w.get(key)
        for r in range(max(2, n_groups // 3)):
            cols = torch.randperm(M.shape[0], generator=gen)[:n]
            yield k, f"s{r}", M[cols, :].T.contiguous()

    for li in layers:
        p = f"model.layers.{li}."
        # --- PAIRED ARMS AT n = d_head: two bases of the SAME subspace.
        # W_O_h's columns are unidentifiable (rotate V and O together and the
        # function is unchanged). The left singular vectors of the OV circuit
        # W_O_h W_V_h are invariant under exactly that rotation. Generically
        # both span col(W_O_h), a d_head-dim subspace of the residual stream.
        Ow = w.get(p + "self_attn.o_proj.weight")
        Vw = w.get(p + "self_attn.v_proj.weight")
        nkv = c.get("num_key_value_heads", nh)
        hh = int(torch.randint(nh, (1,), generator=gen))
        Oh = Ow[:, hh * dh:(hh + 1) * dh]
        kv = hh // (nh // nkv)
        Vh = Vw[kv * dh:(kv + 1) * dh, :]
        Qo, Ro = torch.linalg.qr(Oh)
        Um, _, _ = torch.linalg.svd(Ro @ Vh, full_matrices=False)
        yield "pair_headcols", f"L{li}h{hh}", Oh
        yield "pair_ovsvd", f"L{li}h{hh}", Qo @ Um
        # --- attention: within one head (provably rotation-invariant, MHA)
        O = w.get(p + "self_attn.o_proj.weight")            # (d, nh*dh)
        h = int(torch.randint(nh, (1,), generator=gen))
        cols = torch.randperm(dh, generator=gen)[:n] + h * dh
        yield "attn_head", f"L{li}h{h}", O[:, cols]
        # --- attention: columns across heads (not invariant)
        cols = torch.randperm(O.shape[1], generator=gen)[:n]
        yield "attn_cross", f"L{li}", O[:, cols]
        # --- MLP write directions (elementwise nonlinearity -> privileged)
        if moe:
            e = int(torch.randint(c["num_experts"], (1,), generator=gen))
            D = w.get(p + f"mlp.experts.{e}.down_proj.weight")
            G = w.get(p + f"mlp.experts.{e}.gate_proj.weight")
            lab = f"L{li}e{e}"
        else:
            D = w.get(p + "mlp.down_proj.weight")
            G = w.get(p + "mlp.gate_proj.weight")
            lab = f"L{li}"
        cols = torch.randperm(D.shape[1], generator=gen)[:n]
        yield "mlp_write", lab, D[:, cols]
        # --- residual standard basis
        cols = torch.randperm(d, generator=gen)[:n]
        yield "residual", f"L{li}", torch.eye(d)[:, cols]
        # --- MoE router rows
        if moe and w.has(p + "mlp.gate.weight"):
            Rr = w.get(p + "mlp.gate.weight")               # (E, d)
            cols = torch.randperm(Rr.shape[0], generator=gen)[:n]
            yield "router", f"L{li}", Rr[cols, :].T.contiguous()


# ---------------------------------------------------------------- run

def run(model_id: str, n=64, n_draws=24, n_groups=8, topk=10, dev=None,
        out=Path("out/privilege.json"), seed=0,
        scorer="unsloth/Llama-3.2-1B", min_df=20):
    from jlens.tokspace import EmbedCoherence
    dev = dev or ("mps" if torch.backends.mps.is_available() else "cpu")
    w = Weights(model_id)
    npmi = NPMI(build_npmi(model_id, n_windows=24576, min_df=min_df), device="cpu")
    kept = torch.nonzero(npmi.remap >= 0).squeeze(1)
    emb = EmbedCoherence(model_id, kept, npmi.df, scorer_id=scorer)

    lm = "lm_head.weight" if w.has("lm_head.weight") else "model.embed_tokens.weight"
    W_U = w.get(lm)[kept]
    W_U = W_U - W_U.mean(0, keepdim=True)                   # centred logit lens
    W_U = (W_U * w.get("model.norm.weight").unsqueeze(0)).to(dev)
    print(f"{model_id}: d={w.cfg['hidden_size']} vocab_scored={W_U.shape[0]} "
          f"scorer={scorer} dev={dev} n={n} draws={n_draws}")

    gen = torch.Generator().manual_seed(seed)
    rows = []
    for arm, lab, A in arm_matrices(w, n, gen, n_groups):
        A = A / A.norm(dim=0, keepdim=True).clamp(min=1e-8)
        rec = {"arm": arm, "group": lab}
        for name, M in make_bases(A, n_draws, gen).items():
            Md, ids = M.to(dev), []
            for s in range(0, Md.shape[1], 512):
                ids.append(kept[(Md[:, s:s + 512].T @ W_U.T).topk(topk, 1).indices.cpu()])
            ids = torch.cat(ids)
            Mc = M.cpu()
            k4 = float((((Mc - Mc.mean(0, keepdim=True)) /
                         Mc.std(0, keepdim=True).clamp(min=1e-8)) ** 4).mean() - 3)
            pr = float(((Mc ** 2).sum(0) ** 2 / (Mc ** 4).sum(0)).mean())
            # log-frequency of the promoted tokens (for the K2 frequency check).
            # The full NPMI co-occurrence score is a slow secondary metric and is
            # not computed in the sweep; jlens.corpus_npmi.NPMI.score has it.
            mm = npmi.remap[ids].clamp(min=0)
            ldf = npmi.logp[mm].mean(1)
            _, dc = emb.score(ids)
            # Interpretability lives in the TAIL, not the mean: most units are
            # not readable, so a basis is privileged if it has MORE highly
            # coherent directions than a rotation of itself, even at equal mean.
            per = dc.reshape(-1, n)                          # (draws, n)
            per_basis = per.mean(1)
            qs = torch.tensor([0.5, 0.75, 0.9, 1.0])
            Q = torch.quantile(per, qs, dim=1)                # (4, draws)
            rec[name] = {"cos": float(dc.mean()),
                         "q": [round(float(x), 5) for x in Q.mean(1)],
                         "q_sd": [round(float(x), 5) for x in
                                  (Q.std(1) if Q.shape[1] > 1 else torch.zeros(4))],
                         "logdf": float(ldf.mean()),
                         "draws": [round(float(x), 5) for x in per_basis],
                         "sd_draw": float(per_basis.std()) if per_basis.numel() > 1 else 0.0,
                         "kurt": k4, "pr": pr}
        for null in ("reparam_null", "gram_null", "orth_null"):
            rec[f"z_vs_{null}"] = ((rec["raw"]["cos"] - rec[null]["cos"]) /
                                   max(rec[null]["sd_draw"], 1e-9))
            rec[f"zq90_vs_{null}"] = ((rec["raw"]["q"][2] - rec[null]["q"][2]) /
                                      max(rec[null]["q_sd"][2], 1e-9))
        rows.append(rec)
        print(f"  {arm:11s} {lab:9s} raw={rec['raw']['cos']:+.4f} "
              f"svd={rec['svd']['cos']:+.4f} rep={rec['reparam_null']['cos']:+.4f}"
              f"+-{rec['reparam_null']['sd_draw']:.4f} gram={rec['gram_null']['cos']:+.4f} "
              f"orth={rec['orth_null']['cos']:+.4f}"
              f"  q90={rec['raw']['q'][2]:+.4f}/{rec['reparam_null']['q'][2]:+.4f}"
              f" zq={rec['zq90_vs_reparam_null']:+6.1f}"
              f"  zrep={rec['z_vs_reparam_null']:+6.1f} zgram={rec['z_vs_gram_null']:+6.1f}"
              f"  dlogdf={rec['raw']['logdf']-rec['reparam_null']['logdf']:+.2f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "n": n, "n_draws": n_draws,
                               "topk": topk, "scorer": scorer, "seed": seed,
                               "rows": rows}, indent=1))
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/privilege.json"))
    ap.add_argument("--scorer", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--min-df", type=int, default=20)
    a = ap.parse_args()
    run(a.model, n=a.n, n_draws=a.draws, n_groups=a.groups, out=a.out, seed=a.seed, scorer=a.scorer, min_df=a.min_df)
