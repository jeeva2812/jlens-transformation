"""Does a single singular direction of J carry a concept and steer on its own?

The one place in this project where the answer looked like yes: the top singular
direction of J at layer 24 of SmolLM2-135M reads as a she/he axis.  Because it is
bipolar we can push both ways, which is a far stronger test than pushing once --
a real axis raises log P(' she') - log P(' he') when pushed positive and lowers it
when pushed negative, symmetrically.  Noise does not do that.

This version fixes a type error in the original (`_attic/code/steer_gender.py`).
For `J = U S V^T`:

    U columns live in the TARGET layer's space  -> readable as tokens via W_U
    V columns live in the SOURCE layer's space  -> injectable at layer l

The original read the direction from `U` (right) and then injected `U` (wrong).
Near the target layer J is close to the identity, so U ~ V and the error is
mild; far from it, it is not.  We run both and report cos(U_k, V_k) so the size
of the error is visible rather than assumed.

Every row carries a matched-norm random control and a coherence check on
unrelated text, because a direction that produces 'she' by breaking the model is
not steering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import _MultiCapture, _find_blocks_and_norm

# Occupations with gender priors in both directions, so we can see whether
# steering merely amplifies the prior or can actually flip it.
PROMPTS = [
    "The doctor finished the operation and then",
    "The engineer opened the toolbox because",
    "After the meeting the chief executive said that",
    "The nurse looked at the chart and then",
    "The teacher graded the papers before",
    "The babysitter arrived early because",
]
COHERENCE_TEXT = "The committee met on Tuesday to discuss the budget revisions."


class AddDir:
    def __init__(self, model, layer, direction, magnitude):
        blocks, _ = _find_blocks_and_norm(model)
        self.delta = F.normalize(direction.float(), dim=0) * magnitude
        self.handle = blocks[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = tensor + self.delta.to(tensor.device, tensor.dtype)
        return edited if torch.is_tensor(output) else (edited,) + tuple(output[1:])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--dir", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/rare/gender_axis.json"))
    a = ap.parse_args()

    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    J = blob["J"][a.layer].float()
    U, S, Vh = torch.linalg.svd(J)
    V = Vh.T
    u, v = U[:, a.dir], V[:, a.dir]
    cos_uv = float(F.cosine_similarity(u, v, dim=0))

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, local_files_only=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight.detach()

    def reads_as(direction, k=6):
        with torch.no_grad():
            probabilities = torch.softmax(norm(direction) @ W_U.T, -1)
        _, idx = probabilities.topk(k)
        return [tok.decode(int(j)) for j in idx]

    print(f"{a.model}  layer {a.layer}, singular direction {a.dir}"
          f"  (sigma {float(S[a.dir]):.3f})")
    print(f"  +U reads as: {reads_as(u)}")
    print(f"  -U reads as: {reads_as(-u)}")
    print(f"  cos(U_{a.dir}, V_{a.dir}) = {cos_uv:+.3f}"
          f"   <- how close the injectable direction is to the readable one\n")

    she = tok.encode(" she", add_special_tokens=False)[0]
    he = tok.encode(" he", add_special_tokens=False)[0]
    coherence_ids = tok(COHERENCE_TEXT, return_tensors="pt")["input_ids"]

    def coherence():
        with torch.no_grad():
            out = model(input_ids=coherence_ids,
                        attention_mask=torch.ones_like(coherence_ids), use_cache=False)
        return float(F.cross_entropy(out.logits[0, :-1], coherence_ids[0, 1:]))

    def log_ratio(per_prompt=False):
        values = []
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False)
                lp = torch.log_softmax(out.logits[0, -1].float(), -1)
            values.append(float(lp[she] - lp[he]))
        return values if per_prompt else sum(values) / len(values)

    ids0 = tok(PROMPTS[0], return_tensors="pt")["input_ids"]
    with _MultiCapture(model, [a.layer], blob["target"]) as cap:
        with torch.no_grad():
            model(input_ids=ids0, attention_mask=torch.ones_like(ids0), use_cache=False)
        scale = float(cap.h[a.layer][0].norm(dim=-1).mean())

    base_coherence, base = coherence(), log_ratio()
    base_rows = log_ratio(per_prompt=True)
    print(f"activation norm at layer {a.layer}: {scale:.1f}"
          f"   baseline coherence loss {base_coherence:.3f}")
    print("\nbaseline  log P(' she') - log P(' he'):")
    for value, prompt in zip(base_rows, PROMPTS):
        print(f"  {value:+7.3f}   {prompt!r}")
    print(f"  {base:+7.3f}   MEAN\n")

    generator = torch.Generator().manual_seed(0)
    alphas = (-0.05, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.05)
    results = {}
    for label, direction in (("V (correct: source space)", v),
                             ("U (original: target space)", u)):
        print(f"injecting {label}")
        print(f"{'alpha':>7} {'logP(she)-logP(he)':>19} {'shift':>8} "
              f"{'coherence':>10} {'random':>8}")
        rows = []
        for alpha in alphas:
            if alpha == 0.0:
                print(f"{alpha:>7.3f} {base:>19.3f} {0.0:>+8.3f} "
                      f"{base_coherence:>10.3f} {'--':>8}")
                rows.append({"alpha": 0.0, "log_ratio": base, "shift": 0.0,
                             "coherence": base_coherence, "random": None})
                continue
            with AddDir(model, a.layer, direction, alpha * scale):
                value, coh = log_ratio(), coherence()
            rnd = torch.randn(J.shape[0], generator=generator)
            with AddDir(model, a.layer, rnd, abs(alpha) * scale):
                random_value = log_ratio()
            print(f"{alpha:>7.3f} {value:>19.3f} {value - base:>+8.3f} "
                  f"{coh:>10.3f} {random_value:>8.3f}")
            rows.append({"alpha": alpha, "log_ratio": value, "shift": value - base,
                         "coherence": coh, "random": random_value})
        results[label] = rows
        shifts = [r["shift"] for r in rows if r["alpha"] != 0]
        bipolar = min(shifts) < 0 < max(shifts)
        print(f"  bipolar (pushes both ways): {bipolar}"
              f"   range {min(shifts):+.3f} to {max(shifts):+.3f}\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "config": vars(a) | {"jall": str(a.jall), "out": str(a.out)},
        "singular_value": float(S[a.dir]),
        "cos_u_v": cos_uv,
        "reads_as": {"+U": reads_as(u, 8), "-U": reads_as(-u, 8),
                     "+V": reads_as(v, 8), "-V": reads_as(-v, 8)},
        "activation_scale": scale,
        "baseline": {"mean": base, "per_prompt": base_rows,
                     "coherence": base_coherence},
        "sweeps": results,
        "prompts": PROMPTS,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
