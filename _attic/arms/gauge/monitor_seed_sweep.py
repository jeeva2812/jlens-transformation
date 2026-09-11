"""Robustness check: does the head monitor fail across random gauge choices?

``gauge.monitor_flip`` demonstrates the operational consequence of an
attention-head gauge freedom with one random rotation per model.  This script
holds the trained monitor, data split, model, layer, and task fixed, then repeats
the function-preserving rotation for many seeds.  It reports the distribution
of post-rotation accuracy rather than a single draw.

Run a quick laptop-sized check:

    PYTHONPATH=. .venv/bin/python -m gauge.monitor_seed_sweep \
        --model HuggingFaceTB/SmolLM2-135M --layer 15 --seeds 32

The model is restored exactly after every seed.  Each row also records the
largest logit change, so a failed symmetry cannot masquerade as monitor failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gauge.matrix import rand_orth
from gauge.monitor_flip import CODE, PROSE, acc, fit


PROBE = "The capital of France is"


def _text_accuracy(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   selected: np.ndarray) -> float:
    """Classify a text by its mean token score, weighting every text equally."""
    correct = []
    for group in np.unique(groups[selected]):
        mask = selected[groups[selected] == group]
        pred = float(scores[mask].mean() > 0)
        correct.append(pred == labels[mask[0]])
    return float(np.mean(correct))


def run(model_id: str, layer: int, n_seeds: int,
        out: Path, figure: Path | None = None) -> dict[str, object]:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32).eval()
    cfg = model.config
    block = model.model.layers[layer]
    dh = cfg.hidden_size // cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    group_size = cfg.num_attention_heads // nkv
    texts = CODE + PROSE
    text_labels = np.array([1] * len(CODE) + [0] * len(PROSE))

    def features():
        head, resid, labels, groups = [], [], [], []
        box: dict[str, torch.Tensor] = {}
        handle = block.self_attn.o_proj.register_forward_pre_hook(
            lambda mod, inp: box.__setitem__("z", inp[0]))
        try:
            for group, text in enumerate(texts):
                ids = tok(text, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    result = model(input_ids=ids, output_hidden_states=True)
                n_tokens = ids.shape[1]
                head.append(box["z"][0].numpy().copy())
                resid.append(result.hidden_states[layer + 1][0].numpy().copy())
                labels.extend([text_labels[group]] * n_tokens)
                groups.extend([group] * n_tokens)
        finally:
            handle.remove()
        return (np.concatenate(head), np.concatenate(resid),
                np.asarray(labels), np.asarray(groups))

    probe_ids = tok(PROBE, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        base_logits = model(input_ids=probe_ids).logits[0, -1]
    head_before, resid_before, labels, groups = features()

    # Split by whole texts.  No token from a test text appears in training.
    train_groups = set(
        list(range(len(CODE) // 2))
        + list(range(len(CODE), len(CODE) + len(PROSE) // 2)))
    train = np.asarray([i for i, group in enumerate(groups)
                        if group in train_groups])
    test = np.asarray([i for i, group in enumerate(groups)
                       if group not in train_groups])
    head_probe = fit(head_before[train], labels[train])
    resid_probe = fit(resid_before[train], labels[train])
    base_head_scores = head_probe(head_before)
    base_resid_scores = resid_probe(resid_before)

    attn = block.self_attn
    saved_v = attn.v_proj.weight.detach().clone()
    saved_o = attn.o_proj.weight.detach().clone()
    saved_b = (attn.v_proj.bias.detach().clone()
               if attn.v_proj.bias is not None else None)
    rows = []

    for seed in range(n_seeds):
        try:
            with torch.no_grad():
                for group in range(nkv):
                    rotation = rand_orth(dh, seed * 10_000 + group)
                    value_slice = slice(group * dh, (group + 1) * dh)
                    attn.v_proj.weight[value_slice] = (
                        rotation @ saved_v[value_slice])
                    if saved_b is not None:
                        attn.v_proj.bias[value_slice] = (
                            rotation @ saved_b[value_slice])
                    for query_head in range(
                            group * group_size, (group + 1) * group_size):
                        head_slice = slice(query_head * dh,
                                           (query_head + 1) * dh)
                        attn.o_proj.weight[:, head_slice] = (
                            saved_o[:, head_slice] @ rotation.T)

            with torch.no_grad():
                changed_logits = model(input_ids=probe_ids).logits[0, -1]
            max_logit_delta = float(
                (changed_logits - base_logits).abs().max())
            head_after, resid_after, _, _ = features()
            head_scores = head_probe(head_after)
            resid_scores = resid_probe(resid_after)
            rows.append({
                "seed": seed,
                "max_abs_logit_delta": max_logit_delta,
                "head_token_accuracy": acc(head_scores[test], labels[test]),
                "resid_token_accuracy": acc(resid_scores[test], labels[test]),
                "head_text_accuracy": _text_accuracy(
                    head_scores, labels, groups, test),
                "resid_text_accuracy": _text_accuracy(
                    resid_scores, labels, groups, test),
                "head_feature_max_delta": float(
                    np.abs(head_after - head_before).max()),
                "resid_feature_max_delta": float(
                    np.abs(resid_after - resid_before).max()),
            })
        finally:
            with torch.no_grad():
                attn.v_proj.weight.copy_(saved_v)
                attn.o_proj.weight.copy_(saved_o)
                if saved_b is not None:
                    attn.v_proj.bias.copy_(saved_b)

    def summary(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in rows])
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    result = {
        "model": model_id,
        "layer": layer,
        "n_seeds": n_seeds,
        "n_train_tokens": int(len(train)),
        "n_test_tokens": int(len(test)),
        "n_test_texts": int(len(np.unique(groups[test]))),
        "before": {
            "head_token_accuracy": acc(
                base_head_scores[test], labels[test]),
            "resid_token_accuracy": acc(
                base_resid_scores[test], labels[test]),
            "head_text_accuracy": _text_accuracy(
                base_head_scores, labels, groups, test),
            "resid_text_accuracy": _text_accuracy(
                base_resid_scores, labels, groups, test),
        },
        "after": {
            "head_token_accuracy": summary("head_token_accuracy"),
            "resid_token_accuracy": summary("resid_token_accuracy"),
            "head_text_accuracy": summary("head_text_accuracy"),
            "resid_text_accuracy": summary("resid_text_accuracy"),
            "max_abs_logit_delta": summary("max_abs_logit_delta"),
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    if figure is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seeds = [row["seed"] for row in rows]
        head_acc = [row["head_token_accuracy"] for row in rows]
        resid_acc = [row["resid_token_accuracy"] for row in rows]
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.scatter(seeds, head_acc, color="#c5392d", s=34,
                   label="head-internal monitor after rotation", zorder=3)
        ax.plot(seeds, resid_acc, color="#2f8f5b", linewidth=2.2,
                label="residual-stream control after rotation", zorder=2)
        ax.axhline(result["before"]["head_token_accuracy"], color="#2878a8",
                   linewidth=2, linestyle="--",
                   label="head monitor before rotation")
        ax.axhline(0.5, color="#7f8c8d", linewidth=1.5, linestyle=":",
                   label="chance")
        ax.set_ylim(0.25, 1.02)
        ax.set_xlabel("random function-preserving rotation seed")
        ax.set_ylabel("held-out token accuracy")
        ax.set_title(
            "A head-internal monitor fails across random coordinate systems\n"
            f"{model_id.split('/')[-1]}, layer {layer}; model logits unchanged to "
            f"{result['after']['max_abs_logit_delta']['max']:.1e}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=180)
        plt.close(fig)

    print(f"{model_id}, layer {layer}, {n_seeds} rotations")
    print(f"test set: {len(test)} tokens from "
          f"{len(np.unique(groups[test]))} held-out texts")
    print("token accuracy")
    print(f"  head:     {result['before']['head_token_accuracy']:.3f} -> "
          f"{result['after']['head_token_accuracy']['mean']:.3f} mean "
          f"[{result['after']['head_token_accuracy']['min']:.3f}, "
          f"{result['after']['head_token_accuracy']['max']:.3f}]")
    print(f"  residual: {result['before']['resid_token_accuracy']:.3f} -> "
          f"{result['after']['resid_token_accuracy']['mean']:.3f} mean")
    print("text accuracy (each held-out text weighted equally)")
    print(f"  head:     {result['before']['head_text_accuracy']:.3f} -> "
          f"{result['after']['head_text_accuracy']['mean']:.3f} mean")
    print(f"  residual: {result['before']['resid_text_accuracy']:.3f} -> "
          f"{result['after']['resid_text_accuracy']['mean']:.3f} mean")
    print("symmetry check")
    print(f"  worst max |delta logit|: "
          f"{result['after']['max_abs_logit_delta']['max']:.2e}")
    print(f"wrote {out}")
    if figure is not None:
        print(f"wrote {figure}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument(
        "--out", type=Path,
        default=Path("out/gauge/monitor_seed_sweep.json"))
    parser.add_argument(
        "--figure", type=Path,
        default=Path("out/figs_app/fig10_monitor_seeds.png"))
    args = parser.parse_args()
    run(args.model, args.layer, args.seeds, args.out, args.figure)
