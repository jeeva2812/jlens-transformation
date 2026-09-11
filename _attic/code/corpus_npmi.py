"""Build a token-presence matrix over Pile windows, for NPMI topic coherence.

Label-free, weight-free. Only the tokenizer is shared with the model under
test, so the coherence score cannot be circular with the model's weights.

Usage:
  PYTHONPATH=. .venv/bin/python -m jlens.corpus_npmi --model allenai/OLMoE-1B-7B-0924
"""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PILE = Path.home() / ".cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots"


def _pile_texts(n_docs: int) -> list[str]:
    f = next(PILE.rglob("*.parquet"))
    t = pq.read_table(f)
    col = "text" if "text" in t.column_names else t.column_names[0]
    return t.column(col).to_pylist()[:n_docs]


def build(model_id: str, n_windows: int = 8192, win: int = 128,
          min_df: int = 20, out_dir: Path = Path("out/npmi")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(f"{model_id}|{n_windows}|{win}|{min_df}".encode()).hexdigest()[:10]
    p = out_dir / f"npmi_{model_id.split('/')[-1]}_{tag}.pt"
    if p.exists():
        return p
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    V = len(tok)
    print(f"{model_id}: vocab={V}  target windows={n_windows} x {win} tok")

    rows, cols, w = [], [], 0
    for txt in _pile_texts(10000):
        ids = tok(txt, add_special_tokens=False,
                  truncation=True, max_length=win * 8)["input_ids"]
        for s in range(0, len(ids) - win + 1, win):
            u = np.unique(np.asarray(ids[s:s + win], dtype=np.int64))
            rows.append(np.full(u.shape, w, dtype=np.int32))
            cols.append(u.astype(np.int32))
            w += 1
            if w >= n_windows:
                break
        if w >= n_windows:
            break
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    print(f"  built {w} windows, {len(cols)} (window,token) incidences")

    df = np.bincount(cols, minlength=V)
    keep = np.nonzero(df >= min_df)[0].astype(np.int64)
    print(f"  vocab with df>={min_df}: {len(keep)} / {V} ({100*len(keep)/V:.1f}%)")

    # remap kept vocab -> dense column index; -1 for dropped tokens
    remap = np.full(V, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    m = remap[cols] >= 0
    P = torch.zeros((w, len(keep)), dtype=torch.uint8)
    P[torch.from_numpy(rows[m]).long(), torch.from_numpy(remap[cols[m]]).long()] = 1
    obj = {"P": P, "keep": torch.from_numpy(keep), "remap": torch.from_numpy(remap),
           "df": torch.from_numpy(df[keep]).float(), "W": w,
           "model_id": model_id, "win": win, "min_df": min_df}
    torch.save(obj, p)
    print(f"  wrote {p}  ({P.numel()/1e6:.0f} MB uint8)")
    return p


class NPMI:
    """Batched NPMI topic coherence, plus a frequency-matched baseline.

    Raw NPMI rewards rare tokens (two tokens seen in 5 windows that always
    co-occur score ~1.0), so it is not comparable across directions whose top-k
    differ in frequency. dNPMI subtracts E[NPMI | df-bin of each token],
    estimated from random token pairs, making the score frequency-neutral.
    """

    def __init__(self, path: Path, device="cpu", n_bins=12, n_pairs=200_000, seed=0):
        o = torch.load(path)
        self.P = o["P"].to(device)
        self.remap = o["remap"].to(device)
        self.df = o["df"].to(device)
        self.W = float(o["W"])
        self.device = device
        self.logp = torch.log(self.df / self.W)
        q = torch.linspace(0, 1, n_bins + 1)[1:-1]
        self.edges = torch.quantile(self.logp, q.to(device))
        self.bin = torch.bucketize(self.logp, self.edges)
        self.table = self._baseline(n_bins, n_pairs, seed)

    def _pair_npmi(self, i, j):
        co = (self.P[:, i].to(torch.float16) * self.P[:, j].to(torch.float16)).sum(0).float()
        lpij = torch.log((co + 1.0) / self.W)
        return (lpij - self.logp[i] - self.logp[j]) / (-lpij)

    def _baseline(self, n_bins, n_pairs, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        V = self.df.numel()
        i = torch.randint(V, (n_pairs,), generator=g).to(self.device)
        j = torch.randint(V, (n_pairs,), generator=g).to(self.device)
        vals = torch.cat([self._pair_npmi(i[s:s+20000], j[s:s+20000])
                          for s in range(0, n_pairs, 20000)])
        bi, bj = self.bin[i], self.bin[j]
        cell = bi * n_bins + bj
        s = torch.zeros(n_bins * n_bins, device=self.device).scatter_add_(0, cell, vals)
        c = torch.zeros(n_bins * n_bins, device=self.device).scatter_add_(0, cell, torch.ones_like(vals))
        # symmetrise, then fill empty cells with the global mean
        t = (s / c.clamp(min=1)).reshape(n_bins, n_bins)
        cc = c.reshape(n_bins, n_bins)
        t = (t * cc + t.T * cc.T) / (cc + cc.T).clamp(min=1)
        return torch.where((cc + cc.T) > 0, t, vals.mean())

    def score(self, topk_ids: torch.Tensor, chunk: int = 256):
        """(B,k) vocab ids -> npmi, dnpmi, seen_frac, mean log-df."""
        B, k = topk_ids.shape
        nb = self.table.shape[0]
        o1, o2, o3, o4 = (torch.empty(B, device=self.device) for _ in range(4))
        iu, ju = torch.triu_indices(k, k, offset=1)
        for s in range(0, B, chunk):
            ids = topk_ids[s:s + chunk].to(self.device)
            m = self.remap[ids]
            ok = m >= 0
            mc = m.clamp(min=0)
            b = ids.shape[0]
            cols = self.P[:, mc.reshape(-1)].reshape(self.P.shape[0], b, k)
            cols = cols.to(torch.float16) * ok.to(torch.float16).unsqueeze(0)
            co = torch.einsum("wbi,wbj->bij", cols, cols).float()
            lpi = torch.where(ok, self.logp[mc], torch.zeros_like(self.logp[mc]))
            lpij = torch.log((co + 1.0) / self.W)
            npmi = (lpij - lpi.unsqueeze(2) - lpi.unsqueeze(1)) / (-lpij)
            bb = self.bin[mc]
            base = self.table.reshape(-1)[(bb.unsqueeze(2) * nb + bb.unsqueeze(1)).reshape(-1)].reshape(b, k, k)
            valid = (ok.unsqueeze(2) & ok.unsqueeze(1)).float()
            n_, d_, v = npmi[:, iu, ju], (npmi - base)[:, iu, ju], valid[:, iu, ju]
            vs = v.sum(1).clamp(min=1)
            o1[s:s+chunk] = (n_ * v).sum(1) / vs
            o2[s:s+chunk] = (d_ * v).sum(1) / vs
            o3[s:s+chunk] = ok.float().mean(1)
            o4[s:s+chunk] = (lpi * ok.float()).sum(1) / ok.float().sum(1).clamp(min=1)
        return o1, o2, o3, o4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--windows", type=int, default=8192)
    ap.add_argument("--min-df", type=int, default=20)
    a = ap.parse_args()
    build(a.model, n_windows=a.windows, min_df=a.min_df)
