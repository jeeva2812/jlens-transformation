"""Re-check every headline number in the write-up against the saved results.

No model inference. Reads the JSON artefacts produced by the experiment
scripts and asserts the numbers quoted in docs/ and README.md.

    python verify.py
"""
import json
import sys

FAIL = []


def check(label, got, want, tol=0.02):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} got {got:+8.3f}  want {want:+8.3f}")
    if not ok:
        FAIL.append(label)


def load(p):
    with open(p) as f:
        return json.load(f)


RUNS = [
    ("SmolLM2-135M  L20", "out/rare/smollm_rome_paris.json",
     {"direct_w": 1.893, "pullback": 3.301, "reverse_pullback": -2.790,
      "top_1": 0.490, "top_8": 0.753, "top_64": 2.385, "random": 0.196}),
    ("Qwen3.5-4B    L21", "out/rare/qwen35_4b_rome_paris.json",
     {"direct_w": 3.722, "pullback": 4.868, "reverse_pullback": -4.364,
      "top_1": 0.082, "top_8": 0.202, "top_64": 1.307, "random": 0.053}),
    ("OLMo-3-7B     L22", "out/rare/olmo3_7b_rome_paris.json",
     {"direct_w": 4.069, "pullback": 6.539, "reverse_pullback": -6.531,
      "top_1": -0.165, "top_8": 0.271, "top_64": 1.178, "random": 0.237}),
]


def main():
    print("\n1. Contrastive logit target  w = normalize(W_U[Rome] - W_U[Paris])")
    print("   metric: mean change in (logit Rome - logit Paris), 12 neutral prompts\n")
    for name, path, want in RUNS:
        print(f"{name}   [{path}]")
        d = load(path)
        assert d["target"]["equation"] == "normalize(W_U[Rome] - W_U[Paris])", path
        for method, target in want.items():
            check(method, d["methods"][method]["mean_target_logit_contrast_change"], target)

        print("   ordering and specificity:")
        m = d["methods"]
        pb = m["pullback"]["mean_target_logit_contrast_change"]
        dw = m["direct_w"]["mean_target_logit_contrast_change"]
        rn = m["random"]["mean_target_logit_contrast_change"]
        for label, cond in [
            ("pullback > direct_w > random", pb > dw > rn),
            ("reverse_pullback flips sign", m["reverse_pullback"]["mean_target_logit_contrast_change"] < 0),
            ("held-out Italy/France tokens move", m["pullback"]["mean_heldout_related_logit_contrast_change"] > 1.0),
        ]:
            print(f"  {'PASS' if cond else 'FAIL'}  {label}")
            if not cond:
                FAIL.append(f"{name}: {label}")

        cats = m["pullback"]["category_contrast_changes"]
        for ctrl in ("control_japan_china", "control_spain_germany"):
            if ctrl in cats:
                # Absolute size only; whether it is inside the null is section 2.
                print(f"        control {ctrl:26s} mean {cats[ctrl]['mean']:+.3f}")
        print()

    print("\n2. Matched-norm null, 30 random draws")
    for name, path, _ in RUNS:
        d = load(path)
        rn = d["random_null"]
        n = rn["n_random"]
        dist = rn["distribution"]["target"]
        z = rn["z_scores"]
        print(f"{name}   n={n}  null target mean {dist['mean']:+.3f} sd {dist['sd']:.3f}"
              f"  range [{dist['min']:+.3f}, {dist['max']:+.3f}]")
        ok = n >= 30
        print(f"  {'PASS' if ok else 'FAIL'}  at least 30 draws")
        if not ok:
            FAIL.append(f"{name}: n_random={n}")
        for method in ("direct_w", "pullback"):
            print(f"        {method:18s} z = {z[method]['target']:+7.1f}"
                  f"   held-out related z = {z[method]['related']:+7.1f}")
        print("   specificity controls (should sit inside the null):")
        for ctrl in ("cat:control_japan_china", "cat:control_spain_germany"):
            if ctrl in z["pullback"]:
                zz = z["pullback"][ctrl]
                beats = rn["exceeds_all_draws"]["pullback"][ctrl]
                verdict = "LEAKS" if beats else "clean"
                print(f"        {ctrl[4:]:26s} z = {zz:+6.1f}   {verdict}")
        print()

    print("\n3. Singular components of J are not individually the concept")
    for name, path, _ in RUNS:
        d = load(path)["methods"]
        k1 = d["top_1"]["mean_target_logit_contrast_change"]
        full = d["pullback"]["mean_target_logit_contrast_change"]
        print(f"  {name}:  k=1 recovers {100 * k1 / full:5.1f}% of the full pullback"
              f"   (k=64: {100 * d['top_64']['mean_target_logit_contrast_change'] / full:5.1f}%)")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
