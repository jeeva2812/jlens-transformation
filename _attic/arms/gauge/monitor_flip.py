"""The consequence: a monitor built on head features breaks on an identical model.

Everything so far says head-internal readouts are coordinate-dependent. This
turns that into something operational.

We train a small classifier ("is this text code or English?") on one attention
head's internal state. Then we apply the rotation -- which leaves the model's
behaviour EXACTLY unchanged -- and run the SAME classifier again.

If the classifier is measuring the model, its accuracy should not move. If it is
measuring a coordinate system, it collapses. As a control we do the same thing
with a classifier on the residual stream, which the rotation does not touch.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gauge.matrix import rand_orth

CODE = ["def solve(n):\n    return sum(range(n))", "for i in range(10):\n    print(i)",
        "import numpy as np\narr = np.zeros((3,3))", "class Node:\n    def __init__(self):",
        "x = [i**2 for i in data if i > 0]", "while True:\n    try:\n        break",
        "SELECT name FROM users WHERE id = 5;", "public static void main(String[] args) {",
        "const fn = (a, b) => a + b;", "if (err != nil) { return nil, err }",
        "df = pd.read_csv('data.csv')", "async function get() { await fetch(url) }",
        "#include <stdio.h>\nint main(void)", "let mut v: Vec<i32> = Vec::new();",
        "print('hello world')", "return {'status': 200, 'body': data}"]
PROSE = ["The garden was quiet that morning.", "She had never seen the ocean before.",
         "Economic growth slowed in the third quarter.", "He wondered whether it mattered.",
         "The committee met on Tuesday to discuss it.", "Rain fell steadily through the night.",
         "Their argument lasted well past midnight.", "The old bridge was finally replaced.",
         "Most people agree that honesty matters.", "A letter arrived without a return address.",
         "The film received mixed reviews on release.", "Winter came early to the valley.",
         "She poured the tea and sat down slowly.", "Nobody expected the meeting to end well.",
         "The road curved sharply near the lake.", "He kept the photograph in his wallet."]


def fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y * 2.0 - 1))
    return lambda V: (np.concatenate([(V - mu) / sd, np.ones((len(V), 1))], 1) @ w)


def acc(s, y):
    return float(((s > 0).astype(int) == y).mean())


def main(mid="unsloth/Llama-3.2-1B", L=8):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    c = m.config
    dh = c.hidden_size // c.num_attention_heads
    blk = m.model.layers[L]
    texts = CODE + PROSE
    y0 = np.array([1] * len(CODE) + [0] * len(PROSE))

    def features():
        """ALL heads' internal state (2048 dims) and the residual stream (2048 dims),
        one sample per token. Matched dimensionality, so the two monitors are
        given exactly the same amount to work with."""
        hs, rs, lab, grp = [], [], [], []
        box = {}
        h1 = blk.self_attn.o_proj.register_forward_pre_hook(
            lambda mod, inp: box.__setitem__("z", inp[0]))
        for gi, t in enumerate(texts):
            ids = tok(t, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                o = m(input_ids=ids, output_hidden_states=True)
            n = ids.shape[1]
            hs.append(box["z"][0].numpy().copy())
            rs.append(o.hidden_states[L + 1][0].numpy().copy())
            lab += [y0[gi]] * n
            grp += [gi] * n
        h1.remove()
        return np.concatenate(hs), np.concatenate(rs), np.array(lab), np.array(grp)

    probe_ids0 = tok("The capital of France is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        before_lg = m(input_ids=probe_ids0).logits[0, -1]
    Zb, Rb, y, grp = features()
    # split by TEXT, not by token, so nothing leaks between train and test
    tr_txt = set(list(range(len(CODE)//2)) + list(range(len(CODE), len(CODE)+len(PROSE)//2)))
    tr = np.array([i for i, g in enumerate(grp) if g in tr_txt])
    te = np.array([i for i, g in enumerate(grp) if g not in tr_txt])
    pz, pr = fit(Zb[tr], y[tr]), fit(Rb[tr], y[tr])
    a0z, a0r = acc(pz(Zb[te]), y[te]), acc(pr(Rb[te]), y[te])

    # rotate EVERY head in the layer, each with its own rotation
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    grp_sz = c.num_attention_heads // nkv
    at = blk.self_attn
    saved = [at.v_proj.weight.detach().clone(), at.o_proj.weight.detach().clone(),
             at.v_proj.bias.detach().clone() if at.v_proj.bias is not None else None]
    with torch.no_grad():
        for g in range(nkv):
            R = rand_orth(dh, g)
            vs = slice(g * dh, (g + 1) * dh)
            at.v_proj.weight[vs, :] = R @ at.v_proj.weight[vs, :]
            if at.v_proj.bias is not None:      # Qwen has attention biases
                at.v_proj.bias[vs] = R @ at.v_proj.bias[vs]
            for hq in range(g * grp_sz, (g + 1) * grp_sz):
                hsl = slice(hq * dh, (hq + 1) * dh)
                at.o_proj.weight[:, hsl] = at.o_proj.weight[:, hsl] @ R.T
    # prove the model's behaviour did not change
    probe_ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        after_lg = m(input_ids=probe_ids).logits[0, -1]
    dl = float((after_lg - before_lg).abs().max()); scale = float(before_lg.abs().max())
    Za, Ra, _, _ = features()
    with torch.no_grad():
        at.v_proj.weight.copy_(saved[0]); at.o_proj.weight.copy_(saved[1])
        if saved[2] is not None:
            at.v_proj.bias.copy_(saved[2])
    a1z, a1r = acc(pz(Za[te]), y[te]), acc(pr(Ra[te]), y[te])

    print("A classifier for 'is this code?', trained on the attention layer's internal")
    print("state, then applied to a model that behaves IDENTICALLY.")
    ok = "OK" if dl < 1e-3 else "*** NOT INVARIANT -- do not trust this row ***"
    print(f"  proof the model is unchanged: max |delta logit| = {dl:.2e} "
          f"(logits are ~{scale:.0f})  {ok}")
    print(f"{'monitor built on':34s} {'before':>9s} {'after':>9s}")
    print(f"{'head-internal state':34s} {a0z:9.2f} {a1z:9.2f}")
    print(f"{'residual stream (control)':34s} {a0r:9.2f} {a1r:9.2f}")
    print(f"\n{'':34s} {len(tr)} train tokens, {len(te)} test tokens, {Zb.shape[1]} features each")
    print(f"feature vectors moved: head-internal {float(np.abs(Za-Zb).max()):.3f}, "
          f"residual {float(np.abs(Ra-Rb).max()):.2e}")
    Path("out/gauge").mkdir(parents=True, exist_ok=True)
    Path(f"out/gauge/monitor_flip_{mid.split('/')[-1]}.json").write_text(json.dumps(
        {"head_before": a0z, "head_after": a1z, "resid_before": a0r, "resid_after": a1r,
         "n_train": len(tr), "n_test": len(te), "n_features": int(Zb.shape[1]),
         "max_abs_logit_delta": dl, "logit_scale": scale, "model": mid}, indent=1))
    print("\nwrote out/gauge/monitor_flip.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        for mid, L in [("unsloth/Llama-3.2-1B", 8), ("Qwen/Qwen2.5-0.5B", 12),
                       ("HuggingFaceTB/SmolLM2-135M", 15)]:
            print(f"\n########## {mid}  layer {L}")
            main(mid, L)
