"""1000-draw null re-runs for mid-band deferred records (rule 2).

Decision: validated iff |score| > 99.9th pct of the 1000-draw |null|.
Updates null_threshold / null_pct_beating / strength in place, clears needs_1000.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _find_blocks_and_norm

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
N = 1000


def main():
    recs = []  # (shard_path, record)
    for f in sorted(glob.glob("out/labels/*.json")):
        if Path(f).name == "labels.json":
            continue
        D = json.loads(Path(f).read_text())
        for r in D:
            if r.get("needs_1000"):
                recs.append((f, r))
    print(len(recs), "records need 1000-draw nulls")
    if not recs:
        return
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm_mod = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach().float()
    norm_mod = norm_mod.eval()
    J0 = torch.load("out/ft/J_step0.pt", map_location="cpu", weights_only=False)["J"]
    J6 = torch.load("out/ft/J_step600.pt", map_location="cpu", weights_only=False)["J"]

    def logits_of(t):
        with torch.no_grad():
            return (norm_mod(t.float().unsqueeze(0)).squeeze(0) @ W_U.T)

    def pair_ids(pairs):
        A, B = [], []
        for x, y in pairs:
            ia = tok.encode(" " + x, add_special_tokens=False)
            ib = tok.encode(" " + y, add_special_tokens=False)
            assert len(ia) == 1 and len(ib) == 1, (x, y)
            A.append(ia[0]); B.append(ib[0])
        return torch.tensor(A), torch.tensor(B)

    # group by (matrix, layer, side)
    from collections import defaultdict
    groups = defaultdict(list)
    for f, r in recs:
        side = "in" if r["hypothesis"].endswith("(in)") else \
               "out" if r["hypothesis"].endswith("(out)") else "J"
        groups[(r["matrix"], r["layer"], side)].append((f, r))

    dirty = set()
    for (matrix, L, side), items in sorted(groups.items()):
        print(f"null {matrix} L{L} side {side}: {len(items)} records", flush=True)
        Jb = J0[L].float()
        g = torch.Generator().manual_seed(6100 + L * 7 + (0 if side != "out" else 3))
        R = torch.empty(N, W_U.shape[0])
        for i in range(N):
            rr = torch.randn(Jb.shape[0], generator=g)
            R[i] = logits_of(Jb @ rr if side != "out" else rr)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{N}", flush=True)
        for f, r in items:
            A, B = pair_ids(r["pairs"])
            ns = R[:, B].mean(1) - R[:, A].mean(1)
            na = ns.abs()
            thr = float(na.quantile(0.999))
            pct = float((na >= abs(r["score"])).float().mean().item() * 100)
            ratio = abs(r["score"]) / max(thr, 1e-9)
            r["null_threshold"] = thr
            r["null_pct_beating"] = pct
            r["strength"] = round(ratio, 3)
            r.pop("needs_1000", None)
            if ratio > 1.0:
                r["audit_note"] = (f"RE-AUDIT 1000-draw: ratio {ratio:.3f} > 1.0, stays validated. "
                                   + (r.get("audit_note") or ""))
                print(f"  KEEP {r['family']}[{r['index']}] {r['hypothesis']} ratio={ratio:.3f}", flush=True)
            else:
                s = r.get("steer") or {}
                prior = s.get("steers", "unsteered")
                r["steer"] = None
                r["verdict"] = "rejected"
                r["audit_note"] = (f"RE-AUDIT 1000-draw: ratio {ratio:.3f} <= 1.0, rejected; "
                                   f"prior steer {prior}. " + (r.get("audit_note") or ""))
                print(f"  REJECT {r['family']}[{r['index']}] {r['hypothesis']} ratio={ratio:.3f}", flush=True)
            dirty.add(f)
    for f in dirty:
        by_key = {(r["family"], r["index"]): r for _, r in groups_items(f, recs)}
        D = json.loads(Path(f).read_text())
        for d in D:
            if (d["family"], d["index"]) in by_key:
                d.update(by_key[(d["family"], d["index"])])
                d.pop("needs_1000", None)
        Path(f).write_text(json.dumps(D, indent=1))
    print("done")


def groups_items(f, recs):
    return [(ff, r) for ff, r in recs if ff == f]


if __name__ == "__main__":
    main()
