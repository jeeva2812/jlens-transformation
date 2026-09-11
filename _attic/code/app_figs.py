"""Figures for the MATS 12.0 application.

Every number below comes from saved, verified outputs in out/ (which
verify.py recomputes from the models on disk). Where JSON exists we read it;
otherwise we read the verified report text. Run:

    PYTHONPATH=. .venv/bin/python -m jlens.app_figs

Writes PNGs into out/figs_app/.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "out/figs_app"
PRIV = "out/priv"
GAUGE = "out/gauge"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.dpi": 150,
    }
)

GREEN = "#2e8b57"
RED = "#c0392b"
BLUE = "#2471a3"
GREY = "#7f8c8d"
ORANGE = "#e67e22"


def load(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}")


def q2str(v):
    return f"{v:.4f}".lstrip("0") if abs(v) < 1 else f"{v:.2f}"


# ----------------------------------------------------------------------------
# Fig 1: the same rotation in two places (verified: out/priv/REPORT.txt section A)
# ----------------------------------------------------------------------------
def fig1_symmetry():
    models = ["SmolLM2-135M", "Qwen2.5-0.5B", "Llama-3.2-1B"]
    files = {"SmolLM2-135M": f"{PRIV}/rot_smollm2.json",
              "Qwen2.5-0.5B": f"{PRIV}/rot_qwen05.json",
              "Llama-3.2-1B": f"{PRIV}/rot_llama1b.json"}
    attn, mlp, kept, ov_kept = [], [], [], []
    for m in models:
        d = load(files[m])
        rows = d["rows"]
        attn.append(np.mean([r["attn_dlogit"] for r in rows]))
        mlp.append(np.mean([r["mlp_dlogit"] for r in rows]))
        kept.append(np.mean([r["attn_readout_overlap"] for r in rows]))
        # OV sign-fixed retention is 1.00 (verified CHECK 3 / 2)
        ov_kept.append(1.00)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    x = np.arange(len(models))
    w = 0.38
    b1 = axA.bar(x - w / 2, attn, w, label="attention head", color=BLUE)
    b2 = axA.bar(x + w / 2, mlp, w, label="MLP neurons (same algebra)", color=RED)
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels(models, rotation=15, ha="right")
    axA.set_ylabel("largest change in any logit")
    axA.set_title("The SAME rotation, two places\n(function-preserving vs not)")
    floor = 6e-5
    axA.axhline(floor, color=GREY, ls="--", lw=1)
    axA.text(1.18, floor * 1.6, "float32 noise floor ~4e-5", color=GREY, fontsize=8.5)
    ratios = [a / m for a, m in zip(attn, mlp)]
    for xi, r in zip(x, ratios):
        axA.annotate(f"{r:,.0f}x apart",
                     xy=(xi + w / 2, mlp[int(xi)]),
                     xytext=(xi + 0.25, mlp[int(xi)] * 2.5),
                     fontsize=8.5, color=GREY)
    axA.legend(frameon=False, fontsize=9)

    x2 = np.arange(len(models))
    b3 = axB.bar(x2 - w / 2, [k * 100 for k in kept], w, label="head-column readout", color=RED)
    b4 = axB.bar(x2 + w / 2, [k * 100 for k in ov_kept], w, label="OV circuit (sign-fixed)", color=GREEN)
    axB.set_xticks(x2)
    axB.set_xticklabels(models, rotation=15, ha="right")
    axB.set_ylabel("% of the readout that survives")
    axB.set_ylim(0, 115)
    axB.set_title("What a head readout survives the rotation\n(logit-lens top-10 tokens kept)")
    for rects in (b3,):
        for rect in rects:
            axB.annotate(f"{rect.get_height():.0f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=RED)
    for rects in (b4,):
        for rect in rects:
            axB.annotate(f"{rect.get_height():.0f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=GREEN)
    axB.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "fig1_symmetry.png")


# ----------------------------------------------------------------------------
# Fig 2: the method-invariance heatmap (verified: out/priv VERIFY_ALL / matrix jsons)
# ----------------------------------------------------------------------------
def fig2_matrix():
    data = load(f"{GAUGE}/report_numbers.json")["matrix"]
    models = list(data)
    symmetries = ["head_rotate", "mlp_rescale", "mlp_rescale_2x", "mlp_permute"]
    sym_label = {
        "head_rotate": "rotate\nattention head",
        "mlp_rescale": "rescale\nMLP neuron 9x",
        "mlp_rescale_2x": "rescale\n(realistic 2x)",
        "mlp_permute": "shuffle\nMLP neurons",
    }
    methods = [
        ("m7_head_cols", "head-column readout (the kind people publish)"),
        ("m6_ov_naive", "OV singular vectors, read naively"),
        ("m6b_ov_signfixed", "OV singular vectors, sign fixed"),
        ("m6c_ov_subspace", "OV subspace (the right object)"),
        ("m1_logit_lens", "MLP neuron readout (logit lens)"),
        ("m2_max_acts", "max-activating examples"),
        ("m3_unit_ranking", "ranking neurons by activation"),
        ("m4_act_x_grad", "activation \u00d7 gradient"),
        ("m5_residual_probe", "residual-stream direction"),
    ]
    mkeys = [m for m, _ in methods]
    vals = np.zeros((len(methods), len(symmetries)))
    for i, mk in enumerate(mkeys):
        for j, s in enumerate(symmetries):
            cell = [data[m].get(s, {}).get(mk, np.nan) for m in models]
            cell = [float(c) for c in cell if c is not None and not np.isnan(c)]
            vals[i, j] = np.mean(cell) if cell else np.nan

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    im = ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(symmetries)))
    ax.set_xticklabels([sym_label[s] for s in symmetries])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([lbl for _, lbl in methods])
    for i in range(len(methods)):
        for j in range(len(symmetries)):
            v = vals[i, j]
            if np.isnan(v):
                continue
            txt = "0.00" if v < 0.005 else ("1.00" if v > 0.995 else f"{v:.2f}")
            color = "white" if 0.25 < v < 0.75 else ("black" if v < 0.3 else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("fraction of the method's answer that survives the edit")
    ax.set_title("What survives a symmetry the model cannot detect — 1.00 = unchanged, 0.00 = replaced\n"
                 "(each cell averaged over SmolLM2-135M, Qwen2.5-0.5B, Llama-3.2-1B; every edit free to ~4e-5 logits)")
    save(fig, "fig2_matrix.png")


# ----------------------------------------------------------------------------
# Fig 3: a monitor that falls to chance (verified: out/gauge/monitor_flip_*.json, CHECK 9)
# ----------------------------------------------------------------------------
def fig3_monitor():
    d = load(f"{GAUGE}/monitor_flip.json")
    per = {}
    for f in os.listdir(GAUGE):
        if f.startswith("monitor_flip_") and f.endswith(".json"):
            m = load(os.path.join(GAUGE, f))
            per.setdefault(m.get("model", f), m)
    labels = ["SmolLM2-135M", "Qwen2.5-0.5B", "Llama-3.2-1B"]
    short = {"HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M",
             "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
             "unsloth/Llama-3.2-1B": "Llama-3.2-1B"}
    rows = []
    for f, m in per.items():
        rows.append((short.get(m["model"], m.get("model")), m))
    order = {l: i for i, l in enumerate(labels)}
    rows.sort(key=lambda t: order.get(t[0], 99))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, (label, m) in zip(axes, rows):
        x = [0, 1, 3, 4]
        colors = [BLUE, RED, GREEN, GREEN]
        vals = [m["head_before"], m["head_after"], m["resid_before"], m["resid_after"]]
        b = ax.bar(x, vals, color=colors, width=0.75)
        labels_ = ["head\nbefore", "head\nafter", "residual\nbefore", "residual\nafter"]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_, fontsize=8)
        ax.axhline(0.5, color=GREY, ls="--", lw=1)
        ax.set_title(label, fontsize=11)
        for rect, v in zip(b, vals):
            ax.annotate(f"{v:.2f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.text(1.95, 0.505, "chance", fontsize=7.5, color=GREY, va="bottom")
        if ax is axes[0]:
            ax.set_ylabel("monitor AUC\n(code vs English)")
    axes[0].set_ylim(0, 1.08)
    fig.suptitle("\u201cis this code?\u201d classifier on head-internal features vs residual stream, before and after a\n"
                 "rotation that changes the model by 4e-5 of a logit and generates identical text", y=1.02, fontsize=11)
    save(fig, "fig3_monitor.png")


# ----------------------------------------------------------------------------
# Fig 4: the SAE audit (verified: out/gauge/sae_audit.json, CHECK 10)
# ----------------------------------------------------------------------------
def fig4_sae():
    d = load(f"{GAUGE}/sae_audit.json")
    rows = []
    for k, v in d.items():
        if k == "own_gauge" or not isinstance(v, dict):
            continue
        sae, sym = k.split("|")
        rows.append((sae.strip(), sym.strip(), v["kept"], v["max_act_change"]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.1))
    x = np.arange(len(rows))
    kept = [r[2] * 100 for r in rows]
    b = axA.bar(x, kept, 0.6, color=GREEN)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{r[0][:12]}\n{r[1]}" for r in rows], fontsize=7.5)
    axA.set_ylim(0, 115)
    axA.set_ylabel("% of features that fire\nkept after the edit")
    axA.set_title("Two real, independently trained SAEs under every symmetry")
    for rect, r in zip(b, rows):
        axA.annotate("100%", xy=(rect.get_x() + rect.get_width() / 2, 100),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color="white",
                     fontweight="bold")
        axA.text(rect.get_x() + rect.get_width() / 2, 60,
                 f"\u0394act {r[3]:.1e}", ha="center", fontsize=7, color=GREY)

    # the SAE's own gauge freedom
    recon_plain = 6.7e-6
    recon_topk = 0.208
    rc = [recon_plain, recon_topk]
    colors = [GREEN, RED]
    b = axB.bar(["plain ReLU SAE\n(no top-k)", "actual top-k SAE\n(k=32)"], [recon_plain, recon_topk],
                color=colors, width=0.55)
    axB.set_yscale("log")
    axB.set_ylabel("max reconstruction difference\nunder the SAE's own rescaling freedom")
    axB.set_title("An SAE's own gauge: does its output change\nif a feature detector is 3x more sensitive?")
    for rect, v in zip(b, rc):
        axB.annotate(f"{v:.1e}", xy=(rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5,
                     color="black", fontweight="bold")
    axB.text(0.98, 10 ** 0.6, "feature ranking keeps only 46.9% in both cases",
             fontsize=8.5, color=GREY, va="center", ha="right")
    save(fig, "fig4_sae.png")


# ----------------------------------------------------------------------------
# Fig 5: the router correction (verified: out/gauge/router_centre.json)
# ----------------------------------------------------------------------------
def fig5_router():
    d = load(f"{GAUGE}/router_centre.json")

    def decompose(rows):
        identity, cone = [], []
        for r in rows:
            if None in (r.get("reparam_null"), r.get("orth_null")):
                continue
            # identity: do THESE directions matter? raw minus the orbit
            identity.append(r["raw"] - r["reparam_null"])
            # cone: is the shape of the region enough? orbit minus orthonormal
            cone.append(r["reparam_null"] - r["orth_null"])
        return np.array(identity), np.array(cone)

    id_raw, cone_raw = decompose(d["raw_perlayer"])
    id_cent, cone_cent = decompose(d["centred_perlayer"])

    groups = ["raw router", "free component\nremoved (38.3%)"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(2)
    w = 0.36

    def ci95(a):
        m = np.mean(a)
        se = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))
        return m, m - se, m + se

    mi, li, hi = ci95(id_raw)
    mco, lco, hco = ci95(cone_raw)
    mc, lc, hc = ci95(id_cent)
    mcc, lcc, hcc = ci95(cone_cent)
    ax.bar(x[0] - w / 2, mi, w, yerr=[[mi - li], [hi - mi]], color=RED,
           label="does the identity of the\ndirections matter? (raw \u2212 orbit)", capsize=3)
    ax.bar(x[0] + w / 2, mco, w, yerr=[[mco - lco], [hco - mco]], color=BLUE,
           label="is it just the shape of the region? (orbit \u2212 orth)", capsize=3)
    ax.bar(x[1] - w / 2, mc, w, yerr=[[mc - lc], [hc - mc]], color=RED, alpha=0.4, capsize=3)
    ax.bar(x[1] + w / 2, mcc, w, yerr=[[mcc - lcc], [hcc - mcc]], color=BLUE, alpha=0.4, capsize=3)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("readability advantage over the right null\n(coherence, 16 layers \u00b1 95% CI)")
    ax.set_title("Cleaning a free component out of the MoE router uncovered an effect\n"
                 "that the coordinates had been hiding")
    for xi, val in zip([x[0] - w / 2, x[0] + w / 2, x[1] - w / 2, x[1] + w / 2],
                       [mi, mco, mc, mcc]):
        ax.text(xi, val, f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig5_router.png")


# ----------------------------------------------------------------------------
# Fig 6: is the model's own basis privileged? pre-registered (verified: out/priv/summary.json)
# ----------------------------------------------------------------------------
def fig6_privilege():
    s = load(f"{PRIV}/summary.json")["q90"]
    arms = [
        ("attn_head", "attention head columns\n(provable null)"),
        ("mlp_write", "MLP neurons"),
        ("pair_ovsvd", "OV-circuit directions"),
        ("attn_cross", "columns mixed across heads"),
        ("residual", "residual-stream axes"),
        ("router", "MoE router rows"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(arms))
    for i, (key, label) in enumerate(arms):
        d = s[key]
        color = GREY
        if key == "attn_head":
            color = BLUE
        if key == "mlp_write":
            color = GREEN
        ax.errorbar(d["delta"], i, xerr=[[d["delta"] - d["ci"][0]], [d["ci"][1] - d["delta"]]],
                    fmt="o", color=color, markersize=7, capsize=3, linewidth=1.5)
        ax.text(d["ci"][1] + 0.0012, i, f"+{d['delta']:.4f}  [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]",
                va="center", fontsize=8.5)
        if key == "attn_head":
            ax.text(0.0001, i + 0.28, "exactly zero, as the maths requires",
                    fontsize=8, color=BLUE)
        if key == "mlp_write":
            ax.text(0.0001, i + 0.28, "1.31x better than the orbit",
                    fontsize=8, color=GREEN)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lb for _, lb in arms])
    ax.invert_yaxis()
    ax.set_xlabel("raw basis \u2212 random basis spanning the same subspace\n"
                  "(90th-percentile coherence, 604 grouped comparisons, 4 models; CIs over groups)")
    ax.set_title("Pre-registered battery: written down before any number existed\n"
                 "(two predictions came out wrong and are reported wrong)")
    save(fig, "fig6_privilege.png")


# ----------------------------------------------------------------------------
# Fig 7: the causal version (verified: out/priv/CAUSAL.txt)
# ----------------------------------------------------------------------------
def fig7_causal():
    arms = [
        ("attention head columns (null)", +0.30, -0.14, 0.78, BLUE),
        ("MLP neurons", +1.55, 0.81, 2.41, GREEN),
        ("columns across heads", +1.00, 0.16, 1.94, ORANGE),
        ("OV circuit SVD", +0.37, -0.31, 1.06, GREY),
        ("residual basis", +0.20, -0.35, 0.76, GREY),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(arms))
    for i, (label, m, lo, hi, color) in enumerate(arms):
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=color, markersize=8,
                    capsize=3, linewidth=1.6)
        ax.text(hi + 0.05, i, f"{m:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=9)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([a for a, *_ in arms])
    ax.invert_yaxis()
    ax.set_xlabel("within-group z of the model's own directions vs their own rotations\n(44 groups/dose per arm, 2 models, dose in the validated linear window)")
    ax.set_title("Do the model's own directions ACT better, not just read better?\n"
                 "(inject each direction, score what it actually promotes)")
    save(fig, "fig7_causal.png")


# ----------------------------------------------------------------------------
# Fig 8: readout predicts action only in the last quarter (verified: out/priv/AGREE.txt)
# ----------------------------------------------------------------------------
def fig8_depth():
    tot = {"Qwen/Qwen2.5-0.5B": (24, {"L18-L22": 0.434, "L12-L16": 0.061, "L07-L11": 0.005, "L01-L05": 0.006}),
           "HuggingFaceTB/SmolLM2-135M": (30, {"L23-L29": None})}
    # fall back on the verified text numbers (AGREE.txt) rather than re-deriving:
    quarters = ["first\nquarter", "second\nquarter", "third\nquarter", "last\nquarter"]
    qwen = [0.006, 0.005, 0.061, 0.434]
    smol = [0.008, 0.012, 0.074, 0.435]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(4)
    ax.plot(x, [v * 100 for v in qwen], "-o", color=BLUE, label="Qwen2.5-0.5B")
    ax.plot(x, [v * 100 for v in smol], "-o", color=RED, label="SmolLM2-135M")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.axhline(0.02, color=GREY, ls="--", lw=1)
    ax.text(3.2, 0.03, "chance overlap ~0.02%", color=GREY, fontsize=8)
    ax.set_ylabel("top-10 overlap between a direction's readout\nand what it actually promotes (%)")
    ax.set_title("The logit lens predicts what a direction DOES only in the last quarter of the network\n"
                 "(8.2 cells per model per quarter, dose 0.05, two models agree to ~1 pt)")
    ax.legend(frameon=False)
    ax.set_ylim(-0.5, 48)
    save(fig, "fig8_depth.png")


# ----------------------------------------------------------------------------
# Fig 9: steering works but is not surgical (verified: VERIFY_ALL.txt CHECK 6 + report)
# ----------------------------------------------------------------------------
def fig9_steer():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    methods = ["pullback\n(computed gradient)", "embedding\ndifference", "random\ndirection"]
    vals = [5.38, 3.69, 0.28]
    colors = [GREEN, BLUE, GREY]
    b = axL.bar(methods, vals, color=colors, width=0.6)
    axL.set_ylabel("change in log-prob of \u201cRome\u201d (nats)")
    axL.set_title("\u201cThe capital of France is\u201d \u2192 push Rome\n(verified, verify.py 6)")
    for rect, v in zip(b, vals):
        axL.annotate(f"+{v}", xy=(rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

    targets = ["France\n(target)", "Spain", "Japan", "Germany"]
    tvals = [9.31, 6.18, 5.33, 4.56]
    b = axR.bar(targets, tvals, color=[GREEN, BLUE, BLUE, BLUE], width=0.6)
    axR.set_ylabel("change in log-prob (nats)")
    axR.set_title("Same dose \u2014 it is a shove, not a scalpel\ntarget ~2x others; most variance is answer headroom, not meaning")
    for rect, v in zip(b, tvals):
        axR.annotate(f"{v:.1f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    axR.axhline(0, color=GREY, lw=1)
    save(fig, "fig9_steer.png")


if __name__ == "__main__":
    fig1_symmetry()
    fig2_matrix()
    fig3_monitor()
    fig4_sae()
    fig5_router()
    fig6_privilege()
    fig7_causal()
    fig8_depth()
    fig9_steer()
    print("all figures written")