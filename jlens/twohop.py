"""Does J-Lens surface an intermediate variable that is never stated?

Neel's framing: J-Lens "attempts to find the intermediate variables in a model's
forward pass". Next-token agreement is the wrong test for that -- logit lens is
the output head applied early, so it wins by construction. Two-hop questions give
a fair one: there is a bridge entity the model must compute, it appears nowhere
in the prompt, and we know what it is.

  "The capital of the country where Mount Fuji is located is ___"
     bridge = Japan (never stated)      answer = Tokyo

If J-Lens reads intermediate variables, the bridge should show up in the
workspace. Logit lens is the control: if it surfaces the bridge just as well,
the Jacobian is buying nothing here either.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm

MODEL = "allenai/Olmo-3-1025-7B"

# (prompt, bridge entity never stated, final answer)
CASES = [
    ("The capital of the country where Mount Fuji is located is", "Japan", "Tokyo"),
    ("The currency used in the country where the Colosseum stands is", "Italy", "euro"),
    ("The official language of the country where Machu Picchu is found is", "Peru", "Spanish"),
    ("The capital of the country where the Eiffel Tower stands is", "France", "Paris"),
    ("The largest city in the country where the Taj Mahal is located is", "India", "Mumbai"),
    ("The capital of the country where the Great Wall was built is", "China", "Beijing"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jall", type=Path, default=Path("out/Jall_main.pt"))
    ap.add_argument("--topk", type=int, default=50)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev != "cpu" else torch.float32
    blob = torch.load(a.jall, map_location="cpu", weights_only=False)
    Js, layers = blob["J"], blob["layers"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, norm = _find_blocks_and_norm(model)
    W_U = model.get_output_embeddings().weight

    def rank_of(vec, target_ids):
        """Best rank achieved by any surface form of the target token."""
        with torch.no_grad():
            logits = (norm(vec.to(dev, dt)) @ W_U.T).float()
        order = logits.argsort(dim=-1, descending=True)
        best = []
        for t in range(vec.shape[0]):
            row = order[t].cpu()
            rk = min((row == ti).nonzero()[0].item() for ti in target_ids
                     if (row == ti).any())
            best.append(rk)
        return best

    print("Does the bridge entity appear in the workspace, though never stated?\n")
    agg = {"jlens": {l: [] for l in layers}, "logit": {l: [] for l in layers}}

    for prompt, bridge, answer in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
        forms = {bridge, " " + bridge, bridge.lower(), " " + bridge.lower()}
        tids = set()
        for f in forms:
            e = tok.encode(f, add_special_tokens=False)
            if e:
                tids.add(e[0])
        with _MultiCapture(model, layers, 30) as cap:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            acts = {l: cap.h[l][0].detach() for l in layers}

        best_j = (10**9, None, None)
        best_l = (10**9, None, None)
        for l in layers:
            h = acts[l]
            rj = rank_of(h @ Js[l].to(dev, dt).T, tids)
            rl = rank_of(h, tids)
            agg["jlens"][l].append(min(rj))
            agg["logit"][l].append(min(rl))
            if min(rj) < best_j[0]:
                best_j = (min(rj), l, rj.index(min(rj)))
            if min(rl) < best_l[0]:
                best_l = (min(rl), l, rl.index(min(rl)))
        print(f"  {bridge:<8} (answer {answer:<8})  "
              f"J-Lens best rank {best_j[0]:>6} @ layer {best_j[1]:>2} pos {best_j[2]:>2}   |   "
              f"logit lens {best_l[0]:>6} @ layer {best_l[1]:>2}")

    print(f"\n{'layer':>6} {'J-Lens median rank':>20} {'logit median rank':>19}")
    print("-" * 48)
    import statistics
    for l in layers:
        mj = statistics.median(agg["jlens"][l])
        ml = statistics.median(agg["logit"][l])
        flag = "  <-- J-Lens better" if mj < ml / 2 else ""
        print(f"{l:>6} {mj:>20.0f} {ml:>19.0f}{flag}")
    print("\nRank 0 = top of the vocabulary. Lower is better. Vocab is ~100k,")
    print("so a random token sits around 50,000.")


if __name__ == "__main__":
    main()
