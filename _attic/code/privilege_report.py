"""Final tables: rotation sweeps (exact) + privilege battery (tail statistic)."""
from __future__ import annotations
import glob, json, statistics as st
from pathlib import Path
import numpy as np
from jlens.privilege_parse import parse
from jlens.privilege_agg import boot

ORDER = ["pair_headcols", "attn_head", "pair_ovsvd", "attn_cross",
         "residual", "mlp_write", "router"]
LABEL = {"pair_headcols": "attn head cols*", "attn_head": "attn head cols (sub)",
         "pair_ovsvd": "OV circuit SVD", "attn_cross": "attn across heads",
         "residual": "residual basis", "mlp_write": "MLP neurons",
         "router": "MoE router rows"}
VOCAB = {"HuggingFaceTB/SmolLM2-135M": 49152, "Qwen/Qwen2.5-0.5B": 151936,
         "unsloth/Llama-3.2-1B": 128256}


def sweeps():
    print("=" * 100)
    print("A.  THE SAME ROTATION IN TWO PLACES  (exact, float32, every layer)")
    print("=" * 100)
    print(f"{'model':26s} {'L':>3s} {'attn |dlogit|':>14s} {'MLP |dlogit|':>13s} "
          f"{'ratio':>9s} {'head readout kept':>18s} {'chance':>8s}")
    for f in sorted(glob.glob("out/priv/rot_*.json")):
        o = json.loads(Path(f).read_text()); r = o["rows"]
        da = st.median([x["attn_dlogit"] for x in r])
        dm = st.median([x["mlp_dlogit"] for x in r])
        ov = float(np.mean([x["attn_readout_overlap"] for x in r]))
        V = VOCAB.get(o["model"], 50280)
        print(f"{o['model'].split('/')[-1]:26s} {len(r):3d} {da:14.2e} {dm:13.2e} "
              f"{dm/da:8.0f}x {ov*100:17.1f}% {100*10/V:7.2f}%")
    print("  attention rotation is exactly function-preserving; the MLP rotation is the")
    print("  same algebra where SiLU makes it illegal. logit scale ~25 in all three.")


def table(rows, key, label):
    """key: 'mean' or 'q90'."""
    print(f"\n{'arm':22s} {'n':>3s} {'raw':>8s} {'orbit':>8s} "
          f"{'raw - orbit (95% CI)':>28s} {'ratio':>6s} {'mean z':>7s} {'dlogdf':>7s}")
    out = {}
    for arm in ORDER:
        a = [r for r in rows if r["arm"] == arm]
        if len(a) < 3:
            continue
        f = (lambda r, w: r[w]["cos"]) if key == "mean" else (lambda r, w: r[w]["q90"])
        d = [f(r, "raw") - f(r, "reparam_null") for r in a]
        rt = [f(r, "raw") / f(r, "reparam_null") for r in a if f(r, "reparam_null") > 3e-3]
        z = [r["z_vs_reparam_null"] if key == "mean" else r["zq90_vs_reparam_null"] for r in a]
        dl = float(np.mean([r["dlogdf"] for r in a]))
        m, lo, hi = boot(d)
        flag = "  K2!" if abs(dl) > 0.5 else ("   *" if lo > 0 else ("   -" if hi < 0 else ""))
        print(f"{LABEL[arm]:22s} {len(a):3d} {np.mean([f(r,'raw') for r in a]):+8.4f} "
              f"{np.mean([f(r,'reparam_null') for r in a]):+8.4f} "
              f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] {np.mean(rt) if rt else float('nan'):6.2f} "
              f"{np.mean(z):+7.2f} {dl:+7.2f}{flag}")
        out[arm] = {"n": len(a), "delta": m, "ci": [lo, hi],
                    "ratio": float(np.mean(rt)) if rt else None,
                    "mean_z": float(np.mean(z)), "dlogdf": dl}
    return out


def battery():
    rows = parse()
    main = [r for r in rows if r["scorer"].endswith("Llama-3.2-1B")
            and r["vocab"] < 20000]
    print("\n" + "=" * 100)
    print("B.  IS THE MODEL'S OWN BASIS BETTER TO READ THAN A ROTATION OF ITSELF?")
    print("=" * 100)
    ms = sorted({r["model"].split("/")[-1] for r in main})
    print(f"   {len(main)} groups over {len(ms)} models: {', '.join(ms)}")
    print("\n-- primary statistic: 90th percentile of per-direction coherence --")
    q = table(main, "q90", "q90")
    print("\n-- secondary: mean over the 64 directions --")
    mn = table(main, "mean", "mean")
    print("\n" + "-" * 100)
    print("E.  THE PRACTITIONER QUESTION: what does orthogonalising the basis cost?")
    print("-" * 100)
    print(f"{'arm':22s} {'raw':>8s} {'SVD of A':>9s} {'random orth':>12s} "
          f"{'raw-SVD (95% CI)':>26s} {'SVD keeps':>10s}")
    for arm in ORDER:
        a = [r for r in main if r["arm"] == arm]
        if len(a) < 3:
            continue
        d = [r["raw"]["cos"] - r["svd"]["cos"] for r in a]
        m, lo, hi = boot(d)
        base = np.mean([r["orth_null"]["cos"] for r in a])
        num = np.mean([r["svd"]["cos"] for r in a]) - base
        den = np.mean([r["raw"]["cos"] for r in a]) - base
        keep = num / den if abs(den) > 1e-4 else float("nan")
        print(f"{LABEL[arm]:22s} {np.mean([r['raw']['cos'] for r in a]):+8.4f} "
              f"{np.mean([r['svd']['cos'] for r in a]):+9.4f} {base:+12.4f} "
              f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] {keep*100:9.0f}%"
              + ("   *" if lo > 0 else ""))
    print("  'SVD keeps' = how much of raw's advantage over a random orthonormal basis")
    print("  survives replacing the model's basis with the SVD of the same matrix.")

    print("\n  orbit = reparam_null = A@H, the family of weight settings reachable by")
    print("  rotating this basis.  For attn head cols that family is EXACTLY")
    print("  function-preserving, so its row is a measured zero, not an assumption.")
    print("  * / - = 95% bootstrap CI over groups excludes zero.  K2! = frequency-confounded.")

    print("\n" + "-" * 100)
    print("C.  PRE-REGISTERED CHECK K1: is the MLP arm bigger than the provable-null arm?")
    print("-" * 100)
    print(f"{'model':26s} {'null arm (q90)':>16s} {'MLP arm (q90)':>15s} {'ratio':>8s} {'K1':>6s}")
    for m in sorted({r["model"] for r in main}):
        rr = [r for r in main if r["model"] == m]
        a = [r["raw"]["q90"] - r["reparam_null"]["q90"] for r in rr if r["arm"] == "pair_headcols"]
        b = [r["raw"]["q90"] - r["reparam_null"]["q90"] for r in rr if r["arm"] == "mlp_write"]
        if not a or not b:
            continue
        ra = np.mean(b) / abs(np.mean(a)) if np.mean(a) else float("inf")
        print(f"{m.split('/')[-1]:26s} {np.mean(a):+16.4f} {np.mean(b):+15.4f} "
              f"{ra:7.1f}x {'pass' if ra > 2 else 'FAIL':>6s}")

    print("\n" + "-" * 100)
    print("D.  ROBUSTNESS: does the answer depend on the scorer, or on the scored vocabulary?")
    print("-" * 100)
    for name, sel in (("scorer = Llama-3.2-1B, vocab df>=20 (main)",
                       lambda r: r["scorer"].endswith("Llama-3.2-1B") and "wide" not in r["logfile"]),
                      ("scorer = Qwen2.5-0.5B  (independent semantic space)",
                       lambda r: r["scorer"].endswith("Qwen2.5-0.5B")),
                      ("scored vocab df>=5 (35k tokens, not 13k)",
                       lambda r: r["vocab"] > 20000)):
        sub = [r for r in rows if sel(r) and r["model"].endswith("OLMoE-1B-7B-0924")]
        if not sub:
            continue
        line = []
        for arm in ("pair_headcols", "mlp_write", "residual", "router"):
            a = [r["raw"]["q90"] - r["reparam_null"]["q90"] for r in sub if r["arm"] == arm]
            line.append(f"{LABEL[arm]}={np.mean(a):+.4f}" if a else f"{LABEL[arm]}=n/a")
        print(f"  {name:52s} " + "  ".join(line))
    Path("out/priv/summary.json").write_text(json.dumps({"q90": q, "mean": mn}, indent=1))


if __name__ == "__main__":
    sweeps()
    battery()
