"""Symmetry x method matrix: which interpretability outputs are gauge-invariant?

A symmetry is a weight edit that leaves the function EXACTLY unchanged. Every
symmetry below is verified numerically (max |delta logit| reported) before any
method is scored. Against that provable zero we ask, for each method: does its
output change?

  invariant  -> the method measures the model
  changes    -> the method measures the coordinate system it was written in

Symmetries
  S1 head_rotate  W_V -> R W_V, W_O -> W_O R^T   (tied across a kv group)
  S2 mlp_rescale  W_up[i] *= c, W_down[:,i] /= c (W_up enters SwiGLU linearly)
  S3 mlp_permute  shuffle hidden units and matching weights
  S5 norm_absorb  fold RMSNorm gamma into the next matrix

Methods
  M1 logit_lens_unit   top-10 tokens of a unit's write direction
  M1b logit_lens_naive same, but forgetting to fold gamma  (a gauge-fixing test)
  M2 max_acts          top-10 (window,pos) that most activate a unit
  M3 unit_ranking      rank all units by activation on a fixed input
  M4 act_x_grad        rank all units by activation x gradient
  M5 residual_probe    a direction read off the residual stream (SAE proxy)
  M6 ov_svd            top-10 tokens of the OV circuit's singular vectors
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["The capital of France is", "def add(a, b):\n    return",
           "The patient was prescribed", "In 1969 humans walked on the",
           "She opened the door and saw", "Water boils at a temperature of"]


class SkipSym(Exception):
    pass


def rand_orth(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def overlap(a, b):
    """mean fraction of top-k that is shared, row-wise"""
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / len(x) for x, y in zip(a, b)]))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class Probe:
    """Everything a method needs, computed on the CURRENT weights."""

    def __init__(self, model, tok, layer, dev, n_units=256, seed=0):
        self.m, self.tok, self.L, self.dev = model, tok, layer, dev
        self.blk = model.model.layers[layer]
        g = torch.Generator().manual_seed(seed)
        dff = self.blk.mlp.down_proj.weight.shape[1]
        self.units = torch.randperm(dff, generator=g)[:n_units]
        self.enc = [tok(p, return_tensors="pt") for p in PROMPTS]
        self.cor = tok("\n".join(PROMPTS * 6), return_tensors="pt")["input_ids"][:, :512]
        d = model.config.hidden_size
        self.rdir = torch.randn(d, generator=g); self.rdir /= self.rdir.norm()

    # ---- shared machinery
    def logits(self):
        with torch.no_grad():
            return torch.stack([self.m(**e).logits[0, -1].float() for e in self.enc])

    def _acts(self, ids, grad=False):
        """MLP hidden activations (input to down_proj) at the chosen layer."""
        box = {}
        h = self.blk.mlp.down_proj.register_forward_pre_hook(
            lambda mod, inp: box.__setitem__("a", inp[0]))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.m(input_ids=ids.to(self.dev))
        h.remove()
        return box["a"][0], out.logits[0]

    def _WU(self, fold_gamma=True):
        W = self.m.get_output_embeddings().weight.detach().float()
        W = W - W.mean(0, keepdim=True)
        if fold_gamma:
            W = W * self.m.model.norm.weight.detach().float()
        return W

    # ---- methods
    def m1_logit_lens(self, fold_gamma=True, k=10):
        D = self.blk.mlp.down_proj.weight.detach().float()[:, self.units]
        D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-9)
        return (D.T @ self._WU(fold_gamma).T).topk(k, 1).indices.cpu().numpy()

    def m2_max_acts(self, k=10):
        a, _ = self._acts(self.cor)
        return a[:, self.units].T.topk(k, 1).indices.cpu().numpy()

    def m3_unit_ranking(self):
        a, _ = self._acts(self.enc[0]["input_ids"])
        return a[-1, self.units].abs().detach().cpu().numpy()

    def m4_act_x_grad(self):
        ids = self.enc[0]["input_ids"]
        box = {}
        h = self.blk.mlp.down_proj.register_forward_pre_hook(
            lambda mod, inp: (box.__setitem__("a", inp[0]), inp[0].requires_grad_(True),
                              inp[0].retain_grad())[0] if False else box.__setitem__("a", inp[0]))
        h.remove()
        a_store = {}

        def pre(mod, inp):
            x = inp[0].clone().requires_grad_(True)
            a_store["a"] = x
            return (x,)
        h = self.blk.mlp.down_proj.register_forward_pre_hook(pre)
        with torch.enable_grad():
            out = self.m(input_ids=ids.to(self.dev))
            tgt = out.logits[0, -1].argmax()
            g = torch.autograd.grad(out.logits[0, -1, tgt], a_store["a"])[0]
        h.remove()
        return (a_store["a"][0, -1] * g[0, -1])[self.units].abs().detach().cpu().numpy()

    def m5_residual_probe(self):
        with torch.no_grad():
            o = self.m(**self.enc[0], output_hidden_states=True)
        return float(o.hidden_states[self.L + 1][0, -1].float() @ self.rdir.to(self.dev))

    HEAD = 1

    def _ov(self, head=None):
        head = self.HEAD if head is None else head
        c = self.m.config
        nh = c.num_attention_heads; nkv = getattr(c, "num_key_value_heads", nh)
        dh = c.hidden_size // nh
        O = self.blk.self_attn.o_proj.weight.detach().float()[:, head * dh:(head + 1) * dh]
        kv = head // (nh // nkv)
        V = self.blk.self_attn.v_proj.weight.detach().float()[kv * dh:(kv + 1) * dh]
        Q, R = torch.linalg.qr(O)
        U, S, _ = torch.linalg.svd(R @ V, full_matrices=False)
        B = Q @ U
        return B, S

    def m6_ov_svd(self, k=10):
        """naive: read top-k tokens of each OV singular vector, as published"""
        B, _ = self._ov()
        return (B.T @ self._WU().T).topk(k, 1).indices.cpu().numpy()

    def m6b_ov_svd_signfixed(self, k=10):
        """same, with the SVD's residual SIGN gauge fixed by convention:
        force the largest-magnitude component of each vector positive."""
        B, _ = self._ov()
        sgn = torch.sign(B[B.abs().argmax(0), torch.arange(B.shape[1])])
        B = B * sgn.unsqueeze(0)
        return (B.T @ self._WU().T).topk(k, 1).indices.cpu().numpy()

    def m6c_ov_subspace(self):
        """the actual invariant: the SUBSPACE the head writes into"""
        B, _ = self._ov()
        return B.detach().cpu().numpy()

    def m7_head_cols(self, head=None, k=10):
        """the readout people actually publish: 'dimension i of head h promotes ...'"""
        head = self.HEAD if head is None else head
        c = self.m.config
        nh = c.num_attention_heads; dh = c.hidden_size // nh
        D = self.blk.self_attn.o_proj.weight.detach().float()[:, head * dh:(head + 1) * dh]
        D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-9)
        return (D.T @ self._WU().T).topk(k, 1).indices.cpu().numpy()

    def snapshot(self):
        return {"m1_logit_lens": self.m1_logit_lens(True), "m1b_logit_lens_naive": self.m1_logit_lens(False),
                "m2_max_acts": self.m2_max_acts(), "m3_unit_ranking": self.m3_unit_ranking(),
                "m4_act_x_grad": self.m4_act_x_grad(), "m5_residual_probe": self.m5_residual_probe(),
                "m6_ov_naive": self.m6_ov_svd(), "m6b_ov_signfixed": self.m6b_ov_svd_signfixed(),
                "m6c_ov_subspace": self.m6c_ov_subspace(), "m7_head_cols": self.m7_head_cols()}


def compare(a, b):
    out = {}
    for k in a:
        if k in ("m3_unit_ranking", "m4_act_x_grad"):
            out[k] = spearman(a[k], b[k])
        elif k == "m6c_ov_subspace":
            A, B = a[k], b[k]
            P = A @ np.linalg.pinv(A)
            out[k] = 1.0 - float(np.abs(P @ B - B).max())
        elif k == "m5_residual_probe":
            out[k] = 1.0 - min(1.0, abs(a[k] - b[k]) / (abs(a[k]) + 1e-9))
        else:
            out[k] = overlap(a[k], b[k])
    return out


# ------------------------------------------------------------------ symmetries
HEAD = 1


def apply_sym(model, name, layer, seed=0):
    """returns an undo() closure"""
    c = model.config
    nh = c.num_attention_heads; nkv = getattr(c, "num_key_value_heads", nh)
    dh = c.hidden_size // nh; grp = nh // nkv
    blk = model.model.layers[layer]
    saved = []

    def keep(t):
        saved.append((t, t.detach().clone())); return saved[-1][1]

    with torch.no_grad():
        if name == "head_rotate":
            g = HEAD // grp                      # the kv group CONTAINING the head we read
            R = rand_orth(dh, seed)
            at = blk.self_attn
            vs = slice(g * dh, (g + 1) * dh)
            keep(at.v_proj.weight); at.v_proj.weight[vs, :] = R @ at.v_proj.weight[vs, :]
            if at.v_proj.bias is not None:
                keep(at.v_proj.bias); at.v_proj.bias[vs] = R @ at.v_proj.bias[vs]
            keep(at.o_proj.weight)
            for h in range(g * grp, (g + 1) * grp):
                hs = slice(h * dh, (h + 1) * dh)
                at.o_proj.weight[:, hs] = at.o_proj.weight[:, hs] @ R.T
        elif name in ("mlp_rescale", "mlp_rescale_2x"):
            lim = 1.1 if name == "mlp_rescale" else 0.35   # ~9x vs ~2x total spread
            gg = torch.Generator().manual_seed(seed)
            dff = blk.mlp.down_proj.weight.shape[1]
            cvec = torch.exp(torch.empty(dff).uniform_(-lim, lim, generator=gg))
            keep(blk.mlp.up_proj.weight); keep(blk.mlp.down_proj.weight)
            blk.mlp.up_proj.weight *= cvec.unsqueeze(1)
            blk.mlp.down_proj.weight /= cvec.unsqueeze(0)
        elif name == "mlp_permute":
            gg = torch.Generator().manual_seed(seed)
            dff = blk.mlp.down_proj.weight.shape[1]
            p = torch.randperm(dff, generator=gg)
            for w in (blk.mlp.up_proj.weight, blk.mlp.gate_proj.weight):
                keep(w); w.copy_(w[p])
            keep(blk.mlp.down_proj.weight)
            blk.mlp.down_proj.weight.copy_(blk.mlp.down_proj.weight[:, p])
        elif name == "norm_absorb":
            if model.config.tie_word_embeddings:
                raise SkipSym("tied embeddings: scaling W_U would also scale W_E")
            # fold final RMSNorm gamma into the unembedding: gamma*x then W_U
            # equals (W_U diag(gamma)) applied to x.  Function identical.
            gam = model.model.norm.weight.detach().clone()
            keep(model.get_output_embeddings().weight)
            keep(model.model.norm.weight)
            model.get_output_embeddings().weight *= gam.unsqueeze(0)
            model.model.norm.weight.fill_(1.0)
        else:
            raise ValueError(name)

    def undo():
        with torch.no_grad():
            for t, v in saved:
                t.copy_(v)
    return undo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = a.layer if a.layer is not None else model.config.num_hidden_layers // 2
    pr = Probe(model, tok, L, dev, seed=a.seed)
    base_logits = pr.logits()
    base = pr.snapshot()
    print(f"{a.model}  layer {L}  |  {len(pr.units)} MLP units sampled\n")
    res = {}
    methods = list(base.keys())
    print(f"{'symmetry':15s} {'|dlogit|':>9s} {'restored':>9s} " +
          " ".join(f"{m.split('_',1)[0]:>5s}" for m in methods))
    for sym in ["head_rotate", "mlp_rescale", "mlp_rescale_2x", "mlp_permute", "norm_absorb"]:
        try:
            undo = apply_sym(model, sym, L, seed=a.seed)
        except SkipSym as e:
            print(f"{sym:15s} SKIPPED -- {e}")
            continue
        d = float((pr.logits() - base_logits).abs().max())
        cmp = compare(base, pr.snapshot())
        undo()
        back = float((pr.logits() - base_logits).abs().max())
        res[sym] = {"dlogit": d, "restore_err": back, **cmp}
        print(f"{sym:15s} {d:9.2e} {back:9.2e} " + " ".join(f"{cmp[m]:5.2f}" for m in methods))
    ov = overlap(base["m1_logit_lens"], base["m1b_logit_lens_naive"])
    print(f"\ngamma convention: folding RMSNorm gamma into W_U vs not "
          f"-> top-10 overlap {ov:.2f}")
    print("  (gamma's storage location is a gauge choice; the standard logit-lens")
    print("   convention fixes it. This is how much it costs to forget.)")
    res["gamma_convention_overlap"] = ov
    print("\nmethods: " + ",  ".join(methods))
    print("1.00 = output unchanged (gauge-invariant).  0.00 = completely different.")
    print("|dlogit| is the model's own change: all should be float32 noise.")
    o = a.out or Path(f"out/gauge/matrix_{a.model.split('/')[-1]}_s{a.seed}.json")
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"model": a.model, "layer": L, "methods": methods, "res": res}, indent=1))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
