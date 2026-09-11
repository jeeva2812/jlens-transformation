"""Coherence of a token set, measured in a DIFFERENT model's embedding space.

The model under test supplies only the top-k token list. The semantic space
used to judge those tokens comes from an unrelated model (different lab,
different data, different tokenizer), matched by decoded string. So the score
cannot be circular with the weights being probed.

Frequency control: mean pairwise cosine is subtracted by E[cos | df-bin pair],
estimated from random token pairs, so a direction cannot score well merely by
promoting frequent (or rare) tokens.
"""
from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoTokenizer
from safetensors import safe_open

HUB = Path.home() / ".cache/huggingface/hub"


def _embed(model_id: str):
    d = sorted((HUB / ("models--" + model_id.replace("/", "--"))).glob("snapshots/*"))[-1]
    import json
    idx = d / "model.safetensors.index.json"
    key = "model.embed_tokens.weight"
    if idx.exists():
        f = d / json.loads(idx.read_text())["weight_map"][key]
    else:
        f = next(d.glob("*.safetensors"))
    with safe_open(f, framework="pt") as h:
        return h.get_tensor(key).float()


class EmbedCoherence:
    def __init__(self, target_id: str, kept_ids: torch.Tensor, df: torch.Tensor,
                 scorer_id: str = "unsloth/Llama-3.2-1B", device="cpu",
                 n_bins: int = 12, n_pairs: int = 400_000, seed: int = 0,
                 cache: Path = Path("out/npmi")):
        tag = cache / f"emb_{target_id.split('/')[-1]}_{scorer_id.split('/')[-1]}_{len(kept_ids)}.pt"
        if tag.exists():
            E = torch.load(tag)
        else:
            tt = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
            st = AutoTokenizer.from_pretrained(scorer_id, trust_remote_code=True)
            W = _embed(scorer_id)
            W = W - W.mean(0, keepdim=True)
            W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-6)
            rows = []
            for t in kept_ids.tolist():
                s = tt.decode([t])
                ids = st(s, add_special_tokens=False)["input_ids"]
                v = W[ids].mean(0) if ids else torch.zeros(W.shape[1])
                rows.append(v)
            E = torch.stack(rows)
            E = E / E.norm(dim=1, keepdim=True).clamp(min=1e-6)
            tag.parent.mkdir(parents=True, exist_ok=True)
            torch.save(E, tag)
        self.E = E.to(device)
        self.device = device
        self.remap = torch.full((int(kept_ids.max()) + 1,), -1, dtype=torch.long)
        self.remap[kept_ids] = torch.arange(len(kept_ids))
        self.remap = self.remap.to(device)
        lp = torch.log(df.to(device) / df.sum())
        q = torch.linspace(0, 1, n_bins + 1)[1:-1].to(device)
        self.bin = torch.bucketize(lp, torch.quantile(lp, q))
        self.nb = n_bins
        self.table = self._baseline(n_pairs, seed)

    def _baseline(self, n_pairs, seed):
        g = torch.Generator().manual_seed(seed)
        V = self.E.shape[0]
        i = torch.randint(V, (n_pairs,), generator=g).to(self.device)
        j = torch.randint(V, (n_pairs,), generator=g).to(self.device)
        v = (self.E[i] * self.E[j]).sum(1)
        cell = self.bin[i] * self.nb + self.bin[j]
        s = torch.zeros(self.nb ** 2, device=self.device).scatter_add_(0, cell, v)
        c = torch.zeros(self.nb ** 2, device=self.device).scatter_add_(0, cell, torch.ones_like(v))
        t, cc = (s / c.clamp(min=1)).reshape(self.nb, -1), c.reshape(self.nb, -1)
        t = (t * cc + t.T * cc.T) / (cc + cc.T).clamp(min=1)
        return torch.where((cc + cc.T) > 0, t, v.mean())

    def score(self, topk_ids: torch.Tensor, chunk: int = 4096):
        """(B,k) vocab ids -> (cos, dcos) frequency-matched mean pairwise cosine."""
        B, k = topk_ids.shape
        iu, ju = torch.triu_indices(k, k, offset=1)
        o1 = torch.empty(B, device=self.device); o2 = torch.empty(B, device=self.device)
        for s in range(0, B, chunk):
            m = self.remap[topk_ids[s:s + chunk].to(self.device)]
            v = self.E[m]                                     # (b,k,D)
            C = torch.einsum("bid,bjd->bij", v, v)
            bb = self.bin[m]
            base = self.table.reshape(-1)[(bb.unsqueeze(2) * self.nb + bb.unsqueeze(1)).reshape(-1)]
            base = base.reshape(C.shape)
            o1[s:s + chunk] = C[:, iu, ju].mean(1)
            o2[s:s + chunk] = (C - base)[:, iu, ju].mean(1)
        return o1, o2
