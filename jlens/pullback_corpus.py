"""Test how the corpus used to average J changes concept pullbacks.

Two disjoint, genre-matched prose banks estimate independent versions of
``J.T @ w``. A code bank supplies an intentional distribution shift. We ask:

1. Are pullbacks geometrically more similar across the matched prose halves
   than between prose and code?
2. Do corpus-estimated pullbacks retain their held-out causal effect on a
   separate neutral-prompt bank?

Seven vector-Jacobian products are required per batch, per layer, per corpus:
one target seed per concept. The full 576 x 576 Jacobian is never materialised.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.lens import LensSpec, lens_vectors
from jlens.pullback_robustness import AddEverywhere, single_token_ids
from jlens.pullback_steer import CONCEPTS, PROMPTS


# The A/B rows are paired by broad genre, but no text appears in both halves.
PROSE_A = [
    "A cold front crossed the valley overnight, leaving a thin layer of frost on roofs and fields.",
    "The museum exhibition traces how coastal towns changed as railways connected them to larger cities.",
    "Researchers measured how seedlings responded when light, moisture, and soil acidity were varied separately.",
    "The ferry leaves the harbour before sunrise and reaches the island shortly after the morning market opens.",
    "The cook toasted the spices gently before adding tomatoes and allowing the mixture to simmer.",
    "During the second half, the visiting side changed formation and created several chances near the goal.",
    "The quarterly statement separates operating income from one-time gains and explains the change in cash reserves.",
    "The court considered whether the written agreement clearly assigned responsibility for later repairs.",
    "At dinner, the family compared old photographs and tried to remember when each picture had been taken.",
    "The clinic scheduled a follow-up visit so the physician could review the test results and adjust treatment.",
    "The composer repeated a short melody in different keys while the orchestra gradually changed its texture.",
    "Engineers tested the bridge model under alternating loads before approving the final support design.",
    "When the interview ended, both speakers summarized the points on which they still disagreed.",
    "Students used a simple experiment to distinguish correlation from a causal effect in the collected data.",
    "Farmers rotated legumes with cereal crops to improve the soil and reduce reliance on added fertilizer.",
    "The telescope recorded the same faint source on three nights, allowing astronomers to estimate its motion.",
    "Residents proposed a quieter bus route and asked the council to measure travel times before voting.",
    "After heavy rain, water moved slowly through the wetland instead of flowing directly into the river.",
]

PROSE_B = [
    "Warm air moved inland during the afternoon, and clouds formed above the hills before the evening storm.",
    "An archive of letters shows how merchants adapted when new roads shifted trade away from the old port.",
    "The laboratory tracked bacterial growth while changing temperature, nutrient supply, and exposure time.",
    "A regional train follows the coast for two hours before turning inland toward the mountain villages.",
    "The baker folded fruit into the dough, shaped each loaf, and waited for the surface to become golden.",
    "After the interval, the home team pressed higher up the field and forced repeated mistakes from defenders.",
    "Investors compared recurring revenue with borrowing costs before deciding how to value the company.",
    "The judge examined earlier decisions to determine how the statute applied to the disputed transaction.",
    "Two sisters sorted boxes in the attic and wrote names on the back of photographs for younger relatives.",
    "A nurse explained the recovery plan and asked the patient to report any unexpected symptoms immediately.",
    "The painter used a narrow range of colours so that small changes in light would organize the whole scene.",
    "Technicians monitored vibration and heat while the new turbine operated under increasing demand.",
    "The discussion became clearer after each participant restated the other person's argument in neutral terms.",
    "The class compared two sampling procedures and calculated how each one could bias the final estimate.",
    "Growers planted flowering borders beside the orchard to support insects that pollinate the trees.",
    "A sequence of radio observations revealed that the distant object rotated at a remarkably stable rate.",
    "The planning board published several street designs and invited commuters to comment on safety and access.",
    "Native grasses slowed erosion along the stream and provided shelter for birds during the winter.",
]

CODE = [
    "def moving_average(values, width): return [sum(values[i:i+width]) / width for i in range(len(values)-width+1)]",
    "for record in records: key = record.get('category'); totals[key] = totals.get(key, 0) + record['amount']",
    "SELECT customer_id, COUNT(*) AS orders FROM purchases GROUP BY customer_id ORDER BY orders DESC;",
    "function clamp(value, lower, upper) { return Math.max(lower, Math.min(upper, value)); }",
    "class Cache: def __init__(self): self.items = {}; def clear(self): self.items.clear()",
    "async function fetchPage(url) { const response = await fetch(url); return await response.json(); }",
    "with open(path, encoding='utf8') as handle: rows = [line.strip().split(',') for line in handle if line.strip()]",
    "fn distance(a: Point, b: Point) -> f64 { ((a.x-b.x).powi(2) + (a.y-b.y).powi(2)).sqrt() }",
    "public static int max(int[] xs) { int best = xs[0]; for (int x : xs) best = Math.max(best, x); return best; }",
    "data.groupby('region', as_index=False).agg(mean_score=('score', 'mean'), count=('score', 'size'))",
    "try: result = parser.parse(text)\nexcept ParseError as error: logger.warning('invalid input: %s', error)",
    "const unique = Array.from(new Set(items)).filter(item => item !== null).sort((a, b) => a.localeCompare(b));",
    "CREATE INDEX idx_events_created_at ON events(created_at); DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP;",
    "def walk(node): yield node; [yield from walk(child) for child in node.children]",
    "match command { Command::Start(id) => queue.push(id), Command::Stop(id) => active.remove(&id) }",
    "import numpy as np; centered = matrix - matrix.mean(axis=0); covariance = centered.T @ centered / len(matrix)",
    "router.get('/health', (request, response) => response.status(200).send({status: 'ok'}));",
    "results = sorted((score(item), item) for item in candidates); best_score, best_item = results[-1]",
]

# Held-out prefixes for the reverse-domain evaluation. None occurs in CODE.
# They deliberately end where a programming-related or general completion can
# follow; the metric remains the same held-out concept-token lift used on prose.
CODE_PROMPTS = [
    "def parse_record(text):\n    result =",
    "for item in values:\n    total =",
    "SELECT user_id, COUNT(*) FROM events WHERE",
    "function normalize(input) {\n  const output =",
    "class Registry:\n    def lookup(self, name):\n        return",
    "try:\n    response = client.send(request)\nexcept",
    "import numpy as np\nmatrix = np.asarray(data)\nvalue =",
    "if cache_key in cache:\n    cached =",
    "async def fetch_page(url):\n    payload = await",
    "results = sorted(candidates, key=lambda candidate:",
    "CREATE TABLE measurements (id INTEGER, value",
    "while queue:\n    current = queue.pop()\n    if",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--jall", type=Path, default=Path("out/Jall_smollm2.pt"))
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 12, 20, 26])
    parser.add_argument("--target-layer", type=int, default=28)
    parser.add_argument("--dose", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-domain", choices=["prose", "code"], default="prose")
    parser.add_argument("--out", type=Path, default=Path("out/rare/pullback_corpus.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    W = model.get_output_embeddings().weight.detach().float().cpu()
    Wn = F.normalize(W - W.mean(0, keepdim=True), dim=1)
    concepts = {}
    for name, words in CONCEPTS.items():
        ids = single_token_ids(tokenizer, words.split())
        train_ids, test_ids = ids[0::2], ids[1::2]
        concepts[name] = {
            "w": F.normalize(Wn[torch.tensor(train_ids)].mean(0), dim=0),
            "train_ids": train_ids,
            "test_ids": test_ids,
        }
    names = list(concepts)
    seeds = torch.stack([concepts[name]["w"] for name in names]).to(args.device)

    def batches(texts):
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start:start + args.batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=args.max_len,
            )
            yield encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device)

    corpora = {"prose_a": PROSE_A, "prose_b": PROSE_B, "code": CODE}
    transported = {name: {} for name in corpora}
    for layer in args.layers:
        spec = LensSpec(
            layer=layer, target_layer=args.target_layer,
            n_prompts=len(PROSE_A), max_len=args.max_len,
            skip_first=4, weighting="uniform",
        )
        for corpus_name, texts in corpora.items():
            transported[corpus_name][layer] = lens_vectors(
                model, batches(texts), spec, seeds,
            )
        print(f"estimated corpus pullbacks at layer {layer}", flush=True)

    # Independent evaluation prompts and matched control tokens, exactly as in
    # the robustness experiment.
    tokenizer.padding_side = "left"
    evaluation_prompts = PROMPTS if args.evaluation_domain == "prose" else CODE_PROMPTS
    eval_batch = tokenizer(evaluation_prompts, return_tensors="pt", padding=True)
    input_ids = eval_batch["input_ids"].to(args.device)
    attention_mask = eval_batch["attention_mask"].to(args.device)

    @torch.no_grad()
    def log_probs():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return torch.log_softmax(output.logits[:, -1].float(), dim=-1).cpu()

    base = log_probs()
    order = torch.argsort(base.mean(0), descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    for name, data in concepts.items():
        similarities = Wn @ data["w"]
        used = set(data["train_ids"] + data["test_ids"])
        controls = []
        for target_rank in rank[torch.tensor(data["test_ids"])].tolist():
            for offset in range(800):
                found = None
                for candidate in (
                    int(order[min(target_rank + offset, len(order) - 1)]),
                    int(order[max(target_rank - offset, 0)]),
                ):
                    if candidate not in used and float(similarities[candidate]) < 0.10:
                        found = candidate
                        break
                if found is not None:
                    controls.append(found)
                    used.add(found)
                    break
        data["test_tensor"] = torch.tensor(data["test_ids"])
        data["control_tensor"] = torch.tensor(controls[:len(data["test_ids"])])

    scales = {}
    for layer in args.layers:
        captured = []

        def capture(module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            norms = tensor.detach().float().norm(dim=-1)
            seq_pos = attention_mask.long().cumsum(dim=1) - 1
            valid = attention_mask.bool() & (seq_pos >= 1)
            per_prompt = [norms[b][valid[b]].median() for b in range(norms.shape[0])]
            captured.append(torch.stack(per_prompt).mean().cpu())

        handle = model.model.layers[layer].register_forward_hook(capture)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        handle.remove()
        scales[layer] = float(torch.stack(captured).mean())

    saved = torch.load(args.jall, map_location="cpu", weights_only=False)
    geometry_rows = []
    effect_rows = []
    for layer in args.layers:
        corpus_dirs = {name: transported[name][layer] for name in corpora}
        corpus_dirs["pooled_prose"] = (
            corpus_dirs["prose_a"] + corpus_dirs["prose_b"]
        ) / 2
        corpus_dirs["saved_global"] = torch.stack([
            saved["J"][layer].float().T @ concepts[name]["w"] for name in names
        ])
        corpus_dirs["direct_w"] = seeds.cpu()

        for i, concept_name in enumerate(names):
            for a, b in [
                ("prose_a", "prose_b"),
                ("prose_a", "code"),
                ("prose_b", "code"),
                ("pooled_prose", "saved_global"),
            ]:
                geometry_rows.append({
                    "layer": layer, "concept": concept_name, "a": a, "b": b,
                    "cosine": float(F.cosine_similarity(
                        corpus_dirs[a][i], corpus_dirs[b][i], dim=0,
                    )),
                })

            concept = concepts[concept_name]
            for method, matrix in corpus_dirs.items():
                with AddEverywhere(
                    model, layer, matrix[i], args.dose * scales[layer],
                ):
                    edited = log_probs()
                delta = edited - base
                prompt_lifts = (
                    delta[:, concept["test_tensor"]].mean(1)
                    - delta[:, concept["control_tensor"]].mean(1)
                )
                effect_rows.append({
                    "layer": layer, "concept": concept_name, "method": method,
                    "dose": args.dose, "lift": float(prompt_lifts.mean()),
                    "prompt_lifts": [float(x) for x in prompt_lifts],
                })
        print(f"evaluated corpus pullbacks at layer {layer}", flush=True)

    output = {
        "config": {
            "model": args.model, "jall": str(args.jall), "layers": args.layers,
            "target_layer": args.target_layer, "dose": args.dose,
            "batch_size": args.batch_size, "max_len": args.max_len,
            "device": args.device, "evaluation_domain": args.evaluation_domain,
            "out": str(args.out),
        },
        "corpora": corpora,
        "evaluation_prompts": evaluation_prompts,
        "concepts": {
            name: {
                "n_train_tokens": len(data["train_ids"]),
                "n_test_tokens": len(data["test_ids"]),
                "n_control_tokens": len(data["control_tensor"]),
            }
            for name, data in concepts.items()
        },
        "scales": scales,
        "geometry_rows": geometry_rows,
        "effect_rows": effect_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1))

    print("\nMean cosine by comparison")
    for a, b in [
        ("prose_a", "prose_b"), ("prose_a", "code"),
        ("prose_b", "code"), ("pooled_prose", "saved_global"),
    ]:
        values = [r["cosine"] for r in geometry_rows if r["a"] == a and r["b"] == b]
        print(f"{a:14s} vs {b:14s}: {sum(values) / len(values):.3f}")
    print("\nMean held-out lift")
    for method in ["prose_a", "prose_b", "pooled_prose", "code", "saved_global", "direct_w"]:
        values = [r["lift"] for r in effect_rows if r["method"] == method]
        print(f"{method:14s}: {sum(values) / len(values):+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
