"""What lives in the part of the space that nothing is looking at?

Everyone reads the top singular directions of J. Those are the directions the
transport amplifies most. And everyone reads activations -- what the model is
actually carrying right now. This asks about what is in NEITHER.

Set up at layer l:
  S  = span of the top-k right singular vectors of J   -- what J amplifies
  H  = span of the top-m principal directions of the real activations at l
       -- what the model actually occupies
  C  = the orthogonal complement of S + H              -- the leftover

Then take J restricted to C (that is J @ P_C), SVD it, and read its leading
directions out through the unembedding. Are they interpretable?

Two ways it can come out and both are worth knowing:

  leftover is JUNK      the interesting structure really is concentrated in the
                        top of J, and the usual practice is justified.
  leftover is CLEAN     there are readable concepts sitting outside both the
                        top of J and the activation subspace -- present in the
                        weights, unused by the data. Directions the model could
                        express but does not.

Three arms are scored identically so the comparison means something:
  TOP        leading directions of J itself
  LEFTOVER   leading directions of J restricted to the complement
  RANDOM     random directions of the same norm

Also, per prompt: split the activation h into the part inside S and the part
outside it, push each through J, and read both. That is the same question asked
of one input rather than of the weights -- what is this activation carrying that
the top of the lens cannot see?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.interp_score import Scorer

ENGLISH = [
    "The report was finished on Tuesday and", "After the long walk home,",
    "In the middle of the afternoon", "The old building on the corner",
    "When the meeting finally ended,", "She opened the letter and",
    "Nobody expected the answer to be", "The train pulled into the station and",
    "The doctor explained that the patient", "Prices rose sharply after the",
    "He picked up the phone and", "The results of the experiment showed",
    "In the summer of 1994 the", "The judge listened carefully while",
    "Water freezes at a temperature of", "The company announced today that",
]
# The prediction this set exists to test. With ENGLISH prompts the leftover
# directions came out full of code tokens (nargs, metavar, OrderedDict). If the
# complement really is "what the model is not currently using", then swapping the
# prompts to code should swap the leftover to English. If it comes out as code
# either way, the complement is just where the model keeps its code and has
# nothing to do with the activations.
CODE = [
    "import os\nimport sys\n\ndef main():\n    parser =",
    "class Node:\n    def __init__(self, value):\n        self.value =",
    "for i in range(len(arr)):\n    if arr[i] >",
    "def solve(n, k):\n    dp = [[0] * (k + 1) for _ in",
    "try:\n    with open(path) as f:\n        data = json.",
    "async def fetch(session, url):\n    async with session.get(url) as",
    "SELECT user_id, COUNT(*) FROM orders WHERE",
    "const handler = async (req, res) => {\n  const { id } =",
    "if __name__ == \"__main__\":\n    main()\n\ndef helper(x):\n    return",
    "numbers = [int(x) for x in input().split()]\nprint(sum(",
    "@dataclass\nclass Config:\n    name: str\n    retries: int =",
    "while left < right:\n    mid = (left + right) //",
    "df = pd.read_csv(path)\ndf = df[df['value'] >",
    "public static void main(String[] args) {\n    int n =",
    "return sorted(items, key=lambda x: (-x.score, x.",
    "assert result == expected, f\"got {result} want",
]
PROMPT_SETS = {"english": ENGLISH, "code": CODE}


def blocks_of(model):
    m = model.model if hasattr(model, "model") else model
    return m.layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--other", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--k", type=int, default=32, help="size of the top-J subspace removed")
    ap.add_argument("--m", type=int, default=32, help="size of the activation subspace removed")
    ap.add_argument("--ndirs", type=int, default=16)
    ap.add_argument("--nnull", type=int, default=150)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--prompts", default="english", choices=["english", "code"],
                    help="which activations define the 'what the model is using' subspace")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", type=Path, default=Path("out/rare/leftover_smollm2.json"))
    a = ap.parse_args()

    PROMPTS = PROMPT_SETS[a.prompts]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = dict(float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16)[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    W_U = model.get_output_embeddings().weight.detach().float().cpu()

    sc = Scorer(W_U, tok, other_model=a.other, nnull=a.nnull)

    # "register" axis, derived from the two prompt sets themselves rather than
    # from a hand-written word list: the centroid of the tokens that actually
    # occur in the code prompts, minus the centroid for the English ones.
    # A direction scores positive if its top tokens lean code, negative English.
    def centroid(ps):
        ids = sorted({int(t) for p in ps
                      for t in tok(p, return_tensors="pt")["input_ids"][0]})
        return F.normalize(sc.Wn[torch.tensor(ids)].mean(0), dim=0)
    axis = F.normalize(centroid(CODE) - centroid(ENGLISH), dim=0)
    register = sc.Wn @ axis          # per-token codeness, one number each
    def reg(ids): return float(register[ids].mean())
    if sc.OE is not None:
        print(f"cross-model: {sc.shared}/{W_U.shape[0]} tokens shared with {a.other}")

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js = blob["J"]; layers = sorted(Js)
    if a.layers: layers = [l for l in layers if l in a.layers]

    # ---- collect real activations at every layer of interest ----
    store = {l: [] for l in layers}
    hooks = []
    def mk(l):
        def f(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            store[l].append(t[0].detach().float().cpu())
        return f
    for l in layers: hooks.append(blocks_of(model)[l].register_forward_hook(mk(l)))
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad(): model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in hooks: h.remove()
    # drop position 0: it is the attention sink and its norm is ~100x the rest,
    # so it would be the entire activation subspace on its own
    H = {l: torch.cat([x[1:] for x in store[l]], 0) for l in layers}
    for l in layers:
        n0 = store[l][0][0].norm(); nm = H[l].norm(dim=1).median()
        print(f"L{l:<3d} activations {tuple(H[l].shape)}  sink norm {float(n0):.0f} "
              f"vs median {float(nm):.0f}")

    g = torch.Generator().manual_seed(0)
    rows = []
    for l in layers:
        J = Js[l].float()
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        d = J.shape[1]

        # activation principal directions (centred: the mean is a bias, not a concept)
        A = H[l] - H[l].mean(0, keepdim=True)
        Ua, Sa, Va = torch.linalg.svd(A, full_matrices=False)
        Hk = Va[:a.m]

        # complement of (top-k of J) + (top-m of activations), both in layer-l space
        B = torch.cat([Vh[:a.k], Hk], 0)
        Q, _ = torch.linalg.qr(B.T)                       # d x (k+m) orthonormal
        P = torch.eye(d) - Q @ Q.T
        occ = float((Vh[:a.k] @ Q).pow(2).sum() / a.k)    # sanity: should be ~1

        Jc = J @ P
        Uc, Sc, Vc = torch.linalg.svd(Jc, full_matrices=False)
        keep = float(Sc[:a.ndirs].sum() / S[:a.ndirs].sum())

        print(f"\nL{l}  sigma top {float(S[0]):.2f}  leftover top {float(Sc[0]):.2f} "
              f"({keep:.0%} of the strength)  projector check {occ:.3f}")

        for arm, mat in (("top", U), ("leftover", Uc)):
            for i in range(a.ndirs):
                vec = mat[:, i]
                s = sc.score(vec)
                rows.append(dict(layer=l, arm=arm, dir=i,
                                 sigma=float((S if arm == "top" else Sc)[i]),
                                 coh_z=s["coh_z"], xm_z=s["xm_z"],
                                 n_shared=s["n_shared"], reg=reg(s["ids"]),
                                 top=[tok.decode([int(t)]) for t in s["ids"]]))
                if i < 3:
                    print(f"  {arm:9s} d{i} coh_z={s['coh_z']:+6.1f} xm_z={s['xm_z']:+6.1f} "
                          f"reg={reg(s['ids']):+.3f} "
                          f"| {' '.join(repr(x) for x in rows[-1]['top'][:6])}", flush=True)
        for i in range(a.ndirs):
            vec = F.normalize(torch.randn(W_U.shape[1], generator=g), dim=0)
            s = sc.score(vec)
            rows.append(dict(layer=l, arm="random", dir=i, sigma=0.0,
                             coh_z=s["coh_z"], xm_z=s["xm_z"], n_shared=s["n_shared"],
                             reg=reg(s["ids"]),
                             top=[tok.decode([int(t)]) for t in s["ids"]]))

        # ---- per-prompt split: what is the activation carrying outside the top-k? ----
        Sk = Vh[:a.k]
        for pi, p in enumerate(PROMPTS[:6]):
            h = store[l][pi][-1].float()                  # last position
            h_in = Sk.T @ (Sk @ h)
            h_out = h - h_in
            for nm2, hh in (("h_in", h_in), ("h_out", h_out)):
                y = J @ hh
                if y.norm() < 1e-6: continue
                s = sc.score(F.normalize(y, dim=0))
                rows.append(dict(layer=l, arm=nm2, dir=pi, prompt=p,
                                 sigma=float(hh.norm() / h.norm()),
                                 coh_z=s["coh_z"], xm_z=s["xm_z"],
                                 n_shared=s["n_shared"], reg=reg(s["ids"]),
                                 top=[tok.decode([int(t)]) for t in s["ids"]]))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(model=a.model, k=a.k, m=a.m,
                                         prompts=a.prompts, rows=rows), indent=1))
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
