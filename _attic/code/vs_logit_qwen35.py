"""Qwen3.5-4B: published J-Lens vs plain logit lens (Neel's comparison).

Reads the SAME activation h at layer l twice:
    logit lens   softmax(W_U . norm(h))        (assumes J = I)
    J-Lens       softmax(W_U . norm(J_l h))    (published J, camilablank/workspace-lenses)
plus a random-orthonormal control Q at matched Frobenius norm.

No VJP, no training, no dose sweep: J comes straight from the published
artifact (reuse of jlens/verify.py loading convention). Forward-only on MPS/fp16
(smoke-tested NaN-free; the MPS+fp16 NaN caveat in docs/BIGMODELS_RESULT.md is
VJP-specific). Tuned lens NOT compared (limitation, stated in output).

    HF_HUB_OFFLINE=1 .venv/bin/python -m jlens.vs_logit_qwen35
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.5-4b/j-lens/lens.pt"
LAYERS = (16, 24)
N_PROMPTS = 50
MAX_LEN = 128
TOPK = 10
N_RANDOM_DRAWS = 30
N_RANDOM_POSITIONS = 300
N_HAND_CASES = 20


def load_published():
    path = hf_hub_download(LENS_REPO, filename=LENS_FILE)
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d


def pile_prompts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    return [ds[i]["text"] for i in range(n)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/qwen35_lens_vs_logit.json"))
    ap.add_argument("--cases", type=Path, default=Path("out/qwen35_lens_cases.json"))
    ap.add_argument("--png", type=Path, default=Path("out/qwen35_lens_vs_logit.png"))
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev == "mps" else torch.bfloat16
    print(f"device={dev} dtype={dt}", flush=True)

    pub = load_published()
    prov = pub["provenance"]
    model_id, target_layer, skip_first = prov["model_id"], prov["target_layer"], prov["skip_first"]
    print("provenance:", {k: prov[k] for k in ("model_id", "target_layer", "t_max", "n_prompts", "skip_first", "weighting")}, flush=True)
    Jl = {l: pub["J"][l].float() for l in LAYERS}  # fp32 CPU copies
    for l in LAYERS:
        print(f"layer {l}: ||J||_F = {Jl[l].norm().item():.1f}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, trust_remote_code=True).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = getattr(model, "model", model)
    blocks, norm = inner.layers, inner.norm
    W_U = model.get_output_embeddings().weight.detach()  # (V, d)

    prompts = pile_prompts(N_PROMPTS)

    # --- capture block outputs at probe layers (forward-only) ---
    acts: dict[int, torch.Tensor] = {}

    def _mk(l):
        def hook(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            acts[l] = t.detach()
        return hook

    handles = [blocks[l].register_forward_hook(_mk(l)) for l in LAYERS]

    def logits_of(vecs: torch.Tensor) -> torch.Tensor:
        # vecs (B,T,d) on dev/dt -> full-vocab float32 logits on CPU, chunked over T
        outs = []
        for s in range(0, vecs.shape[1], 32):
            v = vecs[:, s:s + 32]
            outs.append((norm(v) @ W_U.T).float().cpu())
        return torch.cat(outs, dim=1)

    recs: list[dict] = []          # one row per (prompt, position): ranks etc.
    tops: dict = {}                # (pi, t) -> top10 lists for hand cases / overlap
    base_lps: list[float] = []
    t_fwd = 0.0
    for i in range(0, len(prompts), a.batch_size):
        chunk = prompts[i:i + a.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"].to(dev)
        mask = enc["attention_mask"].to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        final_logits = out.logits.float().cpu()
        t_fwd += time.time() - t0
        h = {l: acts[l].to(dev) for l in LAYERS}  # keep on device for GEMMs
        B, T = ids.shape
        for b in range(B):
            pi = i + b
            L = int(mask[b].sum().item())
            if L < skip_first + 2:
                continue
            true_next = ids[b, 1:L].long().cpu()          # ground-truth next token at t
            own_next = final_logits[b, :L - 1].argmax(-1)  # model's own argmax at t
            lp = float(torch.log_softmax(final_logits[b, :L - 1], -1)
                       .gather(-1, true_next.unsqueeze(-1)).squeeze(-1).mean())
            base_lps.append(lp)
            for t in range(skip_first, L - 1):
                row = {"pi": pi, "t": t, "true": int(true_next[t]),
                       "own": int(own_next[t]), "base_lp": float(
                           torch.log_softmax(final_logits[b, t], -1)[true_next[t]])}
                for l in LAYERS:
                    hl = h[l][b:b + 1, t:t + 1]                       # (1,1,d)
                    lg_log = logits_of(hl)[0, 0]
                    lg_J = logits_of(hl @ Jl[l].to(dev, dt).T)[0, 0]
                    r_log = int((lg_log > lg_log[row["true"]]).sum()) + 1
                    r_J = int((lg_J > lg_J[row["true"]]).sum()) + 1
                    i_log = lg_log.topk(TOPK).indices.tolist()
                    i_J = lg_J.topk(TOPK).indices.tolist()
                    row[f"rank_logit_L{l}"] = r_log
                    row[f"rank_J_L{l}"] = r_J
                    row[f"hit1_logit_L{l}"] = int(i_log[0] == row["own"])
                    row[f"hit1_J_L{l}"] = int(i_J[0] == row["own"])
                    row[f"hitk_logit_L{l}"] = int(row["own"] in i_log)
                    row[f"hitk_J_L{l}"] = int(row["own"] in i_J)
                    row[f"ov10_L{l}"] = len(set(i_log) & set(i_J))
                    tops[(pi, t, l)] = (i_log, i_J)
                recs.append(row)
        del h
    for hh in handles:
        hh.remove()
    print(f"forwards done: {len(recs)} positions, fwd wall {t_fwd:.0f}s, total {time.time()-t_start:.0f}s", flush=True)

    # --- random-orthonormal control on a stratified subset ---
    g = torch.Generator().manual_seed(0)
    sub_idx = torch.randperm(len(recs), generator=g)[:N_RANDOM_POSITIONS].tolist()
    # group subset positions by (pi) to reuse forwards: need h again -> re-run those prompts
    from collections import defaultdict
    need: dict[int, list[int]] = defaultdict(list)
    for si in sub_idx:
        need[recs[si]["pi"]].append(recs[si]["t"])
    # re-encode needed prompts individually (cheap: len(need) <= 50 forwards)
    h_cache: dict[int, dict[int, torch.Tensor]] = {}
    handles = [blocks[l].register_forward_hook(_mk(l)) for l in LAYERS]
    for pi in need:
        enc = tok(prompts[pi], return_tensors="pt", truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        h_cache[pi] = {l: acts[l][0].detach().to(dev) for l in LAYERS}
    for hh in handles:
        hh.remove()
    d_model = W_U.shape[1]
    rand_ranks: dict[int, list[list[int]]] = {l: [] for l in LAYERS}   # per draw: ranks
    rand_ovJ: dict[int, list[list[int]]] = {l: [] for l in LAYERS}     # per draw: overlap w/ J top10
    gq = torch.Generator().manual_seed(7)
    for draw in range(N_RANDOM_DRAWS):
        Q, _ = torch.linalg.qr(torch.randn(d_model, d_model, generator=gq))
        Q = Q.float()
        for l in LAYERS:
            Ql = (Q * (Jl[l].norm() / Q.norm())).to(dev, dt)  # matched Frobenius norm
            rr, oo = [], []
            for si in sub_idx:
                r = recs[si]
                hv = h_cache[r["pi"]][l][r["t"]:r["t"] + 1].unsqueeze(0)
                lg = logits_of(hv @ Ql.T)[0, 0]
                rr.append(int((lg > lg[r["true"]]).sum()) + 1)
                oo.append(len(set(lg.topk(TOPK).indices.tolist()) & set(tops[(r["pi"], r["t"], l)][1])))
            rand_ranks[l].append(rr)
            rand_ovJ[l].append(oo)
    print(f"random control done: {N_RANDOM_DRAWS} draws x {len(sub_idx)} positions, wall {time.time()-t_start:.0f}s", flush=True)

    # --- hand-score cases ---
    g2 = torch.Generator().manual_seed(1)
    case_idx = torch.randperm(len(recs), generator=g2)[:N_HAND_CASES].tolist()
    cases = []
    for si in case_idx:
        r = recs[si]
        enc = tok(prompts[r["pi"]], return_tensors="pt", truncation=True, max_length=MAX_LEN)
        ids0 = enc["input_ids"][0]
        lo = max(0, r["t"] - 60)
        cases.append({
            "pi": r["pi"], "t": r["t"],
            "context_tail": tok.decode(ids0[lo:r["t"] + 1]),
            "true_next": tok.decode([r["true"]]),
            "own_next": tok.decode([r["own"]]),
            **{f"L{l}_logit_top10": [tok.decode([x]) for x in tops[(r["pi"], r["t"], l)][0]]
               for l in LAYERS},
            **{f"L{l}_J_top10": [tok.decode([x]) for x in tops[(r["pi"], r["t"], l)][1]]
               for l in LAYERS},
        })

    # --- summary ---
    import statistics as st
    summary = {"n_prompts": N_PROMPTS, "n_positions": len(recs),
               "layers": list(LAYERS), "target_layer": target_layer,
               "mean_base_logprob_true": st.fmean(base_lps),
               "median_base_logprob_true": st.median(base_lps),
               "tuned_lens_compared": False,
               "wall_time_s": round(time.time() - t_start, 1),
               "forward_wall_s": round(t_fwd, 1)}
    for l in LAYERS:
        rJ = [r[f"rank_J_L{l}"] for r in recs]
        rL = [r[f"rank_logit_L{l}"] for r in recs]
        shift = [a - b for a, b in zip(rJ, rL)]
        import math
        lr = [math.log10(a) - math.log10(b) for a, b in zip(rJ, rL)]
        rr = [rr2 for draw in rand_ranks[l] for rr2 in draw]
        oo = [o for draw in rand_ovJ[l] for o in draw]
        summary[f"L{l}"] = {
            "median_rank_J": st.median(rJ), "median_rank_logit": st.median(rL),
            "median_paired_shift_J_minus_logit": st.median(shift),
            "median_log10rank_shift": st.median(lr),
            "frac_J_better": sum(s < 0 for s in shift) / len(shift),
            "mean_top10_overlap_J_logit": st.fmean(r[f"ov10_L{l}"] for r in recs),
            "own_top1_logit": st.fmean(r[f"hit1_logit_L{l}"] for r in recs),
            "own_top1_J": st.fmean(r[f"hit1_J_L{l}"] for r in recs),
            "own_top10_logit": st.fmean(r[f"hitk_logit_L{l}"] for r in recs),
            "own_top10_J": st.fmean(r[f"hitk_J_L{l}"] for r in recs),
            "random_median_rank": st.median(rr),
            "random_mean_top10_overlap_with_J": st.fmean(oo),
        }
    a.out.write_text(json.dumps({"summary": summary, "records": recs}, indent=1))
    a.cases.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1), flush=True)

    # --- PNG: rank-shift hist + overlap hist + median-rank bars ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for l in LAYERS:
        lr = [math.log10(r[f"rank_J_L{l}"]) - math.log10(r[f"rank_logit_L{l}"]) for r in recs]
        ax[0].hist(lr, bins=60, alpha=0.5, label=f"L{l}", histtype="stepfilled")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("log10(rank_J) - log10(rank_logit)  (<0: J better)")
    ax[0].set_ylabel("positions"); ax[0].legend(); ax[0].set_title("paired rank shift, true next token")
    for l in LAYERS:
        ov = [r[f"ov10_L{l}"] for r in recs]
        ax[1].hist(ov, bins=range(0, 12), alpha=0.5, label=f"L{l}", histtype="stepfilled")
    ax[1].set_xlabel("|top10_J ∩ top10_logit|"); ax[1].legend(); ax[1].set_title("top-10 overlap between lenses")
    xs, w = [0, 1, 3, 4], 0.35
    for j, l in enumerate(LAYERS):
        s = summary[f"L{l}"]
        ax[2].bar([xs[0] + j * w, xs[1] + j * w, xs[2] + j * w],
                  [s["median_rank_logit"], s["median_rank_J"], s["random_median_rank"]],
                  width=w, label=f"L{l}")
    ax[2].set_yscale("log")
    ax[2].set_xticks([0.17, 1.17, 3.17]); ax[2].set_xticklabels(["logit", "J", "random-Q"])
    ax[2].set_ylabel("median rank of true next token"); ax[2].legend(); ax[2].set_title("median rank vs random control")
    fig.suptitle(f"Qwen3.5-4B J-Lens vs logit lens ({len(recs)} positions, {N_PROMPTS} pile prompts)")
    fig.tight_layout()
    fig.savefig(a.png, dpi=100)
    print(f"wrote {a.out} {a.cases} {a.png}  wall {summary['wall_time_s']}s", flush=True)


if __name__ == "__main__":
    main()
