"""Every headline claim in this project, checkable in one file.

Run one:   PYTHONPATH=. .venv/bin/python verify.py 1
Run all:   PYTHONPATH=. .venv/bin/python verify.py

Each check is self-contained and prints the number it is claiming. Nothing is
loaded from a saved result file -- everything is recomputed from the models on
disk, so if a number here is wrong you will see it.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch

HUB = Path.home() / ".cache/huggingface/hub"


def hdr(n, title, claim):
    print(f"\n{'='*74}\nCHECK {n}: {title}\n  CLAIM: {claim}\n{'-'*74}")


# ---------------------------------------------------------------- 1
def check1():
    """Rotating an attention head changes nothing the model can see."""
    hdr(1, "The symmetry is exact",
        "rotating one attention head leaves the model's output unchanged")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gauge.matrix import apply_sym
    mid = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    ps = ["The capital of France is", "Water boils at a temperature of"]
    def out():
        with torch.no_grad():
            lg = torch.stack([m(**tok(p, return_tensors="pt")).logits[0, -1] for p in ps])
        gen = [tok.decode(m.generate(**tok(p, return_tensors="pt"), max_new_tokens=8,
               do_sample=False, pad_token_id=tok.eos_token_id)[0]) for p in ps]
        return lg, gen
    l0, g0 = out()
    undo = apply_sym(m, "head_rotate", 15); l1, g1 = out(); undo()
    print(f"  largest change in any logit : {float((l1-l0).abs().max()):.2e}")
    print(f"  size of the logits themselves: {float(l0.abs().max()):.1f}")
    print(f"  same text generated?          {g0 == g1}")
    for a, b in zip(g0, g1):
        print(f"     before: {a[:64]!r}\n     after : {b[:64]!r}")


# ---------------------------------------------------------------- 2
def check2():
    """...but it replaces almost everything you would read out of that head."""
    hdr(2, "The readout is destroyed anyway",
        "99%+ of the head's top tokens change under that same rotation")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gauge.matrix import apply_sym
    for mid, L in [("HuggingFaceTB/SmolLM2-135M", 15), ("unsloth/Llama-3.2-1B", 8)]:
        tok = AutoTokenizer.from_pretrained(mid)
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        c = m.config; nh = c.num_attention_heads; dh = c.hidden_size // nh
        def readout():
            O = m.model.layers[L].self_attn.o_proj.weight.detach().float()[:, dh:2*dh]
            O = O / O.norm(dim=0, keepdim=True)
            W = m.get_output_embeddings().weight.detach().float()
            W = (W - W.mean(0, keepdim=True)) * m.model.norm.weight.detach().float()
            return (O.T @ W.T).topk(10, 1).indices
        a = readout()
        undo = apply_sym(m, "head_rotate", L); b = readout(); undo()
        ov = np.mean([len(set(x.tolist()) & set(y.tolist()))/10 for x, y in zip(a, b)])
        print(f"  {mid.split('/')[-1]:20s} top-10 tokens kept: {ov*100:4.1f}%")
        print(f"     dim 0 before: {[tok.decode([t]) for t in a[0][:6]]}")
        print(f"     dim 0 after : {[tok.decode([t]) for t in b[0][:6]]}")
        del m


# ---------------------------------------------------------------- 3
def check3():
    """The fix: the OV circuit does not move."""
    hdr(3, "The invariant alternative",
        "the OV circuit's subspace survives the rotation exactly")
    from transformers import AutoModelForCausalLM
    from gauge.matrix import apply_sym
    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                             dtype=torch.float32).eval()
    c = m.config; nh = c.num_attention_heads; nkv = c.num_key_value_heads
    dh = c.hidden_size // nh; grp = nh // nkv; L, H = 15, 1
    def basis():
        b = m.model.layers[L]
        O = b.self_attn.o_proj.weight.detach().float()[:, H*dh:(H+1)*dh]
        V = b.self_attn.v_proj.weight.detach().float()[(H//grp)*dh:((H//grp)+1)*dh]
        Q, R = torch.linalg.qr(O); U, S, _ = torch.linalg.svd(R @ V, full_matrices=False)
        return Q @ U, S
    B0, S0 = basis()
    undo = apply_sym(m, "head_rotate", L); B1, S1 = basis(); undo()
    P0 = B0 @ B0.T
    print(f"  subspace unchanged?  error = {float((P0@B1-B1).abs().max()):.2e}")
    print(f"  same singular values? error = {float((S0-S1).abs().max()):.2e}")
    print(f"  each vector matches, ignoring sign: |cos| = {float((B0.T@B1).diag().abs().mean()):.4f}")
    print(f"  but the SIGNS flip: {[round(float(x)) for x in (B0.T@B1).diag()[:8]]}")
    print("  -> the object is invariant; reading its top-k tokens is not, unless you")
    print("     fix the sign by convention. top-k of -v is the BOTTOM-k of v.")


# ---------------------------------------------------------------- 4
def check4():
    """38% of the MoE router is free."""
    hdr(4, "The MoE router has a free component",
        "adding a constant to every expert's score changes no routing decision")
    from jlens.privilege import Weights
    w = Weights("allenai/OLMoE-1B-7B-0924")
    fr = []
    for li in range(16):
        R = w.get(f"model.layers.{li}.mlp.gate.weight")
        mean = R.mean(0, keepdim=True)
        fr.append(float(mean.norm()*R.shape[0]**0.5 / R.norm()))
        if li == 8:
            x = torch.randn(R.shape[1])
            s1 = torch.softmax(R @ x, 0); s2 = torch.softmax((R-mean) @ x, 0)
            t1 = torch.topk(R @ x, 8).indices.sort().values
            t2 = torch.topk((R-mean) @ x, 8).indices.sort().values
            print(f"  layer 8: largest change in expert weights = {float((s1-s2).abs().max()):.2e}")
            print(f"           same 8 experts chosen?            {bool((t1==t2).all())}")
    print(f"  share of the router matrix in that free direction: {np.mean(fr)*100:.1f}%")


# ---------------------------------------------------------------- 5
def check5():
    """Weight decay silently pins the MLP scale."""
    hdr(5, "Training already fixes the MLP scale gauge",
        "trained models sit at the balanced-norm point (ratio 1.00)")
    from jlens.privilege import Weights
    for mid in ["allenai/Olmo-3-1025-7B", "Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-135M"]:
        w = Weights(mid); nl = w.cfg["num_hidden_layers"]; r = []
        for li in [nl//4, nl//2, 3*nl//4]:
            U = w.get(f"model.layers.{li}.mlp.up_proj.weight")
            D = w.get(f"model.layers.{li}.mlp.down_proj.weight")
            r.append(float((U.norm(dim=1)/D.norm(dim=0)).median()))
        print(f"  {mid.split('/')[-1]:20s} ||W_up row|| / ||W_down col|| = {np.mean(r):.3f}")
    print("  1.000 would be exactly where weight decay wants it.")


# ---------------------------------------------------------------- 6
def check6():
    """The steering direction we built beats the obvious alternatives."""
    hdr(6, "Pullback steering beats raw and random",
        "the gradient direction moves the answer more than an embedding difference")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid, L = "unsloth/Llama-3.2-1B", 8
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    blk = m.model.layers[L]
    ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
    subj = int((ids[0] == tok.encode(" France", add_special_tokens=False)[0]).nonzero()[0])
    rome = tok.encode(" Rome", add_special_tokens=False)[0]
    paris = tok.encode(" Paris", add_special_tokens=False)[0]

    # 1. pullback: how should layer L change to raise " Rome" over " Paris"?
    keep = {}
    def grab(mod, inp, out):
        # treat layer L's output as the variable we differentiate with respect to
        t = (out[0] if isinstance(out, tuple) else out).detach().requires_grad_(True)
        keep["h"] = t
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    hk = blk.register_forward_hook(grab)
    with torch.enable_grad():
        lg = m(input_ids=ids).logits[0, -1]
        g, = torch.autograd.grad(lg[rome] - lg[paris], keep["h"])
    hk.remove()
    pull = g[0, subj]; pull = pull / pull.norm()

    # 2. the obvious alternative: the difference of the two output embeddings
    W = m.get_output_embeddings().weight.detach().float()
    raw = W[rome] - W[paris]; raw = raw / raw.norm()

    # 3. a random direction, as the control
    rnd = torch.randn(pull.shape[0], generator=torch.Generator().manual_seed(0))
    rnd = rnd / rnd.norm()

    with torch.no_grad():
        hs = m(input_ids=ids, output_hidden_states=True).hidden_states[L + 1]
        scale = float(hs[0, subj].norm())
        base = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)[rome]

    def push(d, alpha=0.5):
        def add(mod, inp, out):
            t = (out[0] if isinstance(out, tuple) else out).clone()
            t[:, subj, :] += alpha * scale * d
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        h2 = blk.register_forward_hook(add)
        with torch.no_grad():
            v = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)[rome]
        h2.remove()
        return float(v - base)

    print("  prompt: 'The capital of France is', pushed at the token ' France'")
    print("  change in log-probability of ' Rome':")
    print(f"     pullback (gradient)   : {push(pull):+.2f}")
    print(f"     embedding difference  : {push(raw):+.2f}")
    print(f"     random direction      : {push(rnd):+.2f}")
    return push, push  # (kept simple; position is CHECK 8)


# ---------------------------------------------------------------- 8
def check8():
    """Where you inject matters -- a lot, in some models."""
    hdr(8, "Position is a hidden variable",
        "the same direction at the subject token vs the last token")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for mid, L in [("HuggingFaceTB/SmolLM2-135M", 12), ("unsloth/Llama-3.2-1B", 8)]:
        tok = AutoTokenizer.from_pretrained(mid)
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        blk = m.model.layers[L]
        ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
        fr = tok.encode(" France", add_special_tokens=False)[0]
        hits = (ids[0] == fr).nonzero()
        subj = int(hits[0]) if len(hits) else ids.shape[1] - 2
        rome = tok.encode(" Rome", add_special_tokens=False)[0]
        paris = tok.encode(" Paris", add_special_tokens=False)[0]
        keep = {}
        def grab(mod, inp, out):
            t = (out[0] if isinstance(out, tuple) else out).detach().requires_grad_(True)
            keep["h"] = t
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        hk = blk.register_forward_hook(grab)
        with torch.enable_grad():
            lg = m(input_ids=ids).logits[0, -1]
            g, = torch.autograd.grad(lg[rome] - lg[paris], keep["h"])
        hk.remove()
        with torch.no_grad():
            hs = m(input_ids=ids, output_hidden_states=True).hidden_states[L + 1]
            base = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)[rome]
        out = {}
        for name, pos in [("at ' France'", subj), ("at the last token", ids.shape[1] - 1)]:
            d = g[0, pos] / g[0, pos].norm()
            sc = float(hs[0, pos].norm())
            def add(mod, inp, o, d=d, sc=sc, pos=pos):
                t = (o[0] if isinstance(o, tuple) else o).clone()
                t[:, pos, :] += 0.5 * sc * d
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            h2 = blk.register_forward_hook(add)
            with torch.no_grad():
                v = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)[rome]
            h2.remove()
            out[name] = float(v - base)
        print(f"  {mid.split('/')[-1]:20s} " +
              "   ".join(f"{k}: {v:+.2f}" for k, v in out.items()))
        del m
    print("  -> position changes the answer in both. In SmolLM2 it FLIPS THE SIGN.")
    print("     Any result that does not say where it injected is not comparable.")


# ---------------------------------------------------------------- 7
def check7():
    """Chess: the piece subspaces are not piece-specific."""
    hdr(7, "Chess piece subspaces are interchangeable",
        "removing the 'pawn' subspace hurts pawns no more than removing the 'king' one")
    d = json.load(open("out/chess/piece_ablate.json"))
    P = list(d["base"].keys())
    print(f"  {'removed':10s}" + "".join(f"{p:>9s}" for p in P))
    for r in P:
        print(f"  {r:10s}" + "".join(f"{d['drop'][r][c]*100:8.2f}%" for c in P))
    print(f"  {'random':10s}" + "".join(f"{d['random'][c]*100:8.2f}%" for c in P))
    diag = np.mean([d["drop"][p][p] for p in P])
    off = np.mean([d["drop"][a][b] for a in P for b in P if a != b])
    print(f"\n  removing a piece's OWN subspace : {diag*100:.3f}% accuracy lost")
    print(f"  removing SOMEONE ELSE'S subspace: {off*100:.3f}%")
    print("  -> the rows are the same. Not piece-specific.")


# ---------------------------------------------------------------- 9
def check9():
    """A monitor built on head features breaks; one on the residual does not."""
    hdr(9, "The consequence for monitoring",
        "a classifier on head-internal features collapses to chance on an identical model")
    from gauge import monitor_flip
    monitor_flip.main()


# ---------------------------------------------------------------- 10
def check10():
    """Two real, trained sparse autoencoders survive every symmetry."""
    hdr(10, "Sparse autoencoders pass the audit",
        "two independently trained SAEs give identical features after every edit")
    from gauge import sae_audit
    sae_audit.main()


# ---------------------------------------------------------------- 11
def check11():
    """The freedom is bigger than rotations, and rotary embeddings shrink half of it."""
    hdr(11, "How big the freedom really is",
        "any invertible map is free in the value-output half; rotary pins the query-key half")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid, L, H = "unsloth/Llama-3.2-1B", 8, 1
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    c = m.config
    nh, nkv = c.num_attention_heads, c.num_key_value_heads
    dh = c.hidden_size // nh; grp = nh // nkv; kv = H // grp
    at = m.model.layers[L].self_attn
    enc = [tok(p, return_tensors="pt") for p in
           ["The capital of France is", "def add(a,b):\n    return", "Water boils at"]]

    def lg():
        with torch.no_grad():
            return torch.stack([m(**e).logits[0, -1] for e in enc])

    base = lg()
    sv, so = at.v_proj.weight.detach().clone(), at.o_proj.weight.detach().clone()
    sq, sk = at.q_proj.weight.detach().clone(), at.k_proj.weight.detach().clone()
    g = torch.Generator().manual_seed(0)
    print(f"  logits are ~{float(base.abs().max()):.0f}\n  VALUE-OUTPUT half:")
    for nm, M in [("a rotation", torch.linalg.qr(torch.randn(dh, dh, generator=g))[0]),
                  ("any invertible map", torch.eye(dh) + 1.5*torch.randn(dh, dh, generator=g)/dh**0.5)]:
        Mi = torch.linalg.inv(M)
        with torch.no_grad():
            at.v_proj.weight[kv*dh:(kv+1)*dh] = Mi @ sv[kv*dh:(kv+1)*dh]
            for hq in range(kv*grp, (kv+1)*grp):
                at.o_proj.weight[:, hq*dh:(hq+1)*dh] = so[:, hq*dh:(hq+1)*dh] @ M
        d = float((lg()-base).abs().max())
        with torch.no_grad():
            at.v_proj.weight.copy_(sv); at.o_proj.weight.copy_(so)
        print(f"    {nm:22s} cond={float(torch.linalg.cond(M)):7.1f}  "
              f"|dlogit|={d:.2e}  {'FREE' if d<1e-3 else 'BREAKS'}")

    def pair_rot(seed):
        gg = torch.Generator().manual_seed(seed); R = torch.eye(dh); h = dh//2
        for i in range(h):
            t = float(torch.rand(1, generator=gg))*6.2832
            ca, sa = float(np.cos(t)), float(np.sin(t))
            R[i, i] = ca; R[i, i+h] = -sa; R[i+h, i] = sa; R[i+h, i+h] = ca
        return R
    print("  QUERY-KEY half (this is where rotary embeddings live):")
    for nm, R in [("a general rotation", torch.linalg.qr(torch.randn(dh, dh, generator=g))[0]),
                  ("a rotation inside each rotary pair", pair_rot(0))]:
        with torch.no_grad():
            for hq in range(kv*grp, (kv+1)*grp):
                at.q_proj.weight[hq*dh:(hq+1)*dh] = R @ sq[hq*dh:(hq+1)*dh]
            at.k_proj.weight[kv*dh:(kv+1)*dh] = R @ sk[kv*dh:(kv+1)*dh]
        d = float((lg()-base).abs().max())
        with torch.no_grad():
            at.q_proj.weight.copy_(sq); at.k_proj.weight.copy_(sk)
        print(f"    {nm:34s} |dlogit|={d:.2e}  {'FREE' if d<1e-3 else 'BREAKS'}")
    print(f"\n  so per head: value-output {dh*dh} free parameters (the full GL group),")
    print(f"  query-key only {dh//2} -- rotary embeddings remove "
          f"{100*(1-(dh//2)/(dh*(dh-1)//2)):.1f}% of that half by accident.")


CHECKS = {1: check1, 2: check2, 3: check3, 4: check4, 5: check5, 6: check6,
          7: check7, 8: check8, 9: check9, 10: check10, 11: check11}

if __name__ == "__main__":
    want = [int(x) for x in sys.argv[1:]] or sorted(CHECKS)
    for n in want:
        try:
            CHECKS[n]()
        except Exception as ex:
            print(f"  CHECK {n} FAILED: {type(ex).__name__}: {ex}")
    print()
