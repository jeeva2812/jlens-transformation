"""Full gallery: EVERY run with subspace/prompts/steer/tested+control.

Run: PYTHONPATH=. .venv/bin/python -m jlens.results_gallery && open out/RESULTS.html
"""
from __future__ import annotations
import html
import json
from pathlib import Path

OUT = Path("out")
E = html.escape


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


CSS = ("<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:2em auto;"
       "padding:0 1em}table{border-collapse:collapse;margin:.7em 0;width:100%}"
       "td,th{border:1px solid #ccc;padding:5px 7px;font-size:12.5px;vertical-align:top}"
       ".gen{font-style:italic;color:#333;max-width:420px}.good{background:#d6f5d6}"
       ".part{background:#fff3cd}.bad{background:#f8d7da}.skip{background:#eee}"
       ".lbl{font-weight:bold;color:#555;width:130px;background:#f4f4f4}"
       "h1{font-size:24px}h2{font-size:18px;border-bottom:2px solid #333;padding-bottom:4px}</style>")


def exp_row(blocks):
    """blocks = [(label, html), ...] -> 4-column evidence table."""
    return ("<table>" + "".join(f"<tr><td class='lbl'>{E(l)}</td><td>{b}</td></tr>"
                                for l, b in blocks) + "</table>")


def S(title, body):
    return f"<section><h2>{title}</h2>{body}</section>"


def main():
    P = ["<h1>Steering results: every run, full evidence</h1>"]
    P.append("<p>Arrows: <b>raw</b>=W<sub>U</sub>[want]−W<sub>U</sub>[was]; "
             "<b>pullback</b>=J<sup>T</sup>w/‖·‖; <b>composed</b>=pullback cut to top-gap "
             "SVD comps; <b>random</b>=same-norm control. Metric ΔlogP(want) unless noted.</p>")

    # 1 trace vs J
    t = load("trace_vs_j_fixed.json")
    if t:
        P.append(S("1. Trace-vs-J (localization ≠ intervention, SmolLM2)",
                   exp_row([
                       ("Subspace sought", "None learned — fixed unembed d per fact; J pile-averaged (25 pile prompts)."),
                       ("Prompts (find)", "8 capital facts, e.g. 'The capital of France is'→Paris/Rome."),
                       ("How steered", "Single layer × last-subject-token inject, same d/norm. Tracing = corrupt+restore."),
                       ("Tested + control",
                        f"90 pts: corr(trace,edit)=<b>{t['corr_trace_edit']:+.3f}</b> (Hase −0.13), "
                        f"corr(J,edit)=<b>{t['corr_j_edit']:+.3f}</b>; within-layer both ≈0; "
                        "layer R² 0.344, +J/+trace +0.000/+0.002. Random edits ≈ true → saturated. VERDICT: dead."),
                   ])))

    # 2 compose
    cs = load("compose_steer.json")
    if cs:
        rows = "".join(
            f"<tr><td>L{r['layer']} (kept {r['kept_frac']:.0%})</td>"
            f"<td>{r['raw']['eff']:+.2f}</td><td><b>{r['pullback']['eff']:+.2f}</b></td>"
            f"<td>{r['composed']['eff']:+.2f}</td><td>{r['random']['eff']:+.2f}</td></tr>"
            for r in cs["rows"])
        P.append(S("2. Composed steering (SmolLM2, France→Rome)",
                   exp_row([
                       ("Subspace sought", "All 576 SVD comps/layer scored by own Rome−Paris gap; top-8 kept (3% of d energy); pullback = soft weighting."),
                       ("Prompts (find)", "Pile J + 'The capital of France is' readout."),
                       ("How steered", "Single layer × subject token, α=1.0. Neighbor: 'The Louvre is in'."),
                       ("Tested + control",
                        "<table><tr><th></th><th>raw</th><th>pullback</th><th>composed</th><th>random</th></tr>"
                        + rows + "</table>Pullback wins every layer; hard top-k loses."),
                   ])))

    # 3 rename (BOTH doses: clean whisper vs strong — the dose-response story)
    for fn, dose in [("rename_eval_clean.json", "clean α=0.002"),
                     ("rename_eval.json", "strong α=0.02")]:
        rn = load(fn)
        if not rn:
            continue
        for sec in ["direct", "transfer", "control"]:
            rows = "".join(
                f"<tr><td>{E(r['prompt'])} → {E(r['want'])}<br>base logP {r['base_want']:.2f}</td>"
                + "".join(f"<td>{r[k]['d_want']:+.2f}<br><span class='gen'>{E(r[k]['gen'][:90])}</span></td>"
                          for k in ["pullback", "raw", "random"]) + "</tr>"
                for r in rn["sections"].get(sec, []))
            P.append(S(f"3. Rename [{sec}] ({dose}, SmolLM2 L12 global)",
                       exp_row([
                           ("Subspace sought", "Pullback L12 from pile J (same as §2)."),
                           ("Prompts (find)", "Pile-10k."),
                           ("How steered", f"ALL positions, {dose}. Random same norm."),
                           ("Tested + control",
                            "<table><tr><th>prompt</th><th>pullback</th><th>raw</th><th>random</th></tr>"
                            + rows + "</table>"),
                       ])))
        un = "".join(f"<tr><td>{E(r['prompt'])}</td><td>{r['dir']}</td><td>TV {r['tv']}</td>"
                     f"<td class='gen'>{E(r['gen'][:90])}</td></tr>"
                     for r in rn["sections"].get("unrelated", []))
        P.append(S(f"3u. Rename unrelated ({dose})",
                   f"<table><tr><th>prompt</th><th>dir</th><th>drift</th><th>text</th></tr>{un}</table>"))

    # 4 geneig
    gg = load("geneig_rome.json")
    if gg:
        P.append(S("4. Generalized eigvecs (SmolLM2 L12)",
                   exp_row([
                       ("Subspace sought", f"Solve JᵀJv=λCv (C = 30 pile prompts). Top PC = 99.8% variance → whitening hunts 0.2%. Best gaps: {gg['best_gaps'][:3]}"),
                       ("Prompts (find)", "Pile-10k covariance + France readout."),
                       ("How steered", "Subject token, α=1.0, best-gap geneigvec."),
                       ("Tested + control",
                        f"geneig {gg['steer']['geneig-best']:+.2f} vs pullback {gg['steer']['pullback']:+.2f} "
                        f"vs raw {gg['steer']['raw']:+.2f} vs random {gg['steer']['random']:+.2f}. "
                        "Unrelated TV≈1.0 both. VERDICT: loses."),
                   ])))

    # 5 DAS trio (metrics from run logs — no generations saved)
    P.append(S("5. DAS supervised rooms (SmolLM2 L12, copy-paste intervention)",
               exp_row([
                   ("Subspace sought", "LEARNED v (Adam): 1D raw-space, 1D de-sinked, k=8 room. Init = pullback."),
                   ("Prompts (find)", "Train: 'In Paris/Rome they speak→Italian' ×2 + 'capital France/Italy→Rome'. Holdout: Eiffel/Colosseum, 'Paris/Rome is capital of'. Null: random v/room."),
                   ("How steered", "No push scale — copy source's component along v into target run, last token."),
                   ("Tested + control",
                    "<ul><li>1D raw: loss flat 19.94→19.93, dlogP +0.00 everywhere (lever &lt;1% of sink state).</li>"
                    "<li>1D de-sinked: loss 18.06; pair 1 −5.82→−4.07 (6×), pairs 2–3 + holdout flat.</li>"
                    "<li>k=8: train +1.82/+1.02/+0.72 (random room ≈ base — good null); holdout −0.00/+0.09. GATE FAILED.</li></ul>"),
               ])))

    # 6 delta J
    dj = load("delta_j.json")
    if dj:
        P.append(S("6. ΔJ = J_Paris − J_Rome (SmolLM2 L12)",
                   exp_row([
                       ("Subspace sought", "SVD(ΔJ). ‖ΔJ‖/‖J‖ = <b>0.45</b> — not tiny. Top comps gaps −29/+11."),
                       ("Prompts (find)", "8 Paris sentences vs 8 Rome templates (Louvre/Colosseum, Eiffel/Trevi, French/Italian… — see jlens/delta_j.py)."),
                       ("How steered", "Top comp v0 injected at last token (noted flaw: v0 is the PARIS side; Rome comps #3/#5 untested; saturated hn scale)."),
                       ("Tested + control", f"dJ-v0 {dj['steer']['dJ-v0']:+.2f} vs pullback {dj['steer']['pullback']:+.2f} vs raw {dj['steer']['raw']:+.2f}. Comparison unreliable — flagged in notes."),
                   ])))

    # 7 llama prompt-specificity + battery
    lp = load("llama_ps_clean.json") or load("llama_ps.json")
    if lp:
        rows = "".join(
            f"<tr><td>{E(r['prompt'])}</td><td>{r.get('prompt-spec', r.get('ps', '?'))}</td>"
            f"<td>{r.get('pile-pull', r.get('pile', '?'))}</td><td>{r.get('raw', '?')}</td>"
            f"<td>{r.get('random', '?')}</td></tr>" for r in lp.get("rows", []))
        P.append(S("8. Llama-1B: prompt-specific vs pile pullback",
                   exp_row([
                       ("Subspace sought", "Jᵀw per layer from 5 FACT prompts vs 8 pile prompts (1 VJP each). cos 0.16 (L0) → 0.84 (L14): prompt-specific early, generic late."),
                       ("Prompts (find)", "France/Eiffel/Paris-speak/official-language/Paris-capital (+ pile-10k)."),
                       ("How steered", "L8 all-positions, α ladder. Generations sampled (see §9/Rome→'Rome, but…' etc)."),
                       ("Tested + control",
                        "<table><tr><th>prompt</th><th>prompt-spec</th><th>pile</th><th>raw</th><th>random</th></tr>"
                        + rows + "</table>"),
                   ])))

    # 8 multi-push
    ml = load("multi_push_low.json")
    m3 = load("multi_push_3x.json")
    if ml:
        t = "".join(f"<tr><td>{k}</td><td>{v['direct']:+.2f}</td><td>{v['transfer']:+.2f}</td>"
                    f"<td class='gen'>{E(v['gen_direct'][:80])}</td></tr>" for k, v in ml.items())
        t3 = "".join(f"<tr><td>{k}</td><td>{v['direct']:+.2f}</td><td>{v['transfer']:+.2f}</td></tr>"
                     for k, v in m3.items()) if m3 else ""
        P.append(S("9. Multi-layer (Llama L6+8+10)",
                   exp_row([
                       ("Subspace sought", "Per-layer prompt-spec pullbacks (reused §8)."),
                       ("Prompts (find)", "Same 5 fact prompts."),
                       ("How steered", "3 hooks at once; dose split (low) or full each (3×)."),
                       ("Tested + control",
                        "<table><tr><th>setup</th><th>direct</th><th>transfer</th><th>text</th></tr>" + t + "</table>"
                        + ("<p>3× overdose:</p><table>" + t3 + "</table>" if t3 else "")),
                   ])))

    # 9 cross-model
    cm = load("cross_model.json")
    if cm:
        rows = "".join(
            f"<tr><td>{E(m.split('/')[-1])}</td><td>{r.get('sink', '?')}</td>"
            + "".join(f"<td>{r['alphas'][a]['direct_pb']:+.1f}/{r['alphas'][a]['direct_rand']:+.1f}</td>"
                      for a in sorted(r.get("alphas", {}))) + "</tr>"
            for m, r in cm.items() if isinstance(r, dict) and "alphas" in r)
        P.append(S("10. Cross-model (5× ≤1B)",
                   exp_row([
                       ("Subspace sought", "Per-model fact-prompt pullback, mid-layer."),
                       ("Prompts (find)", "4 fact prompts per model."),
                       ("How steered", "Mid-layer all-positions, α 0.005/0.02, random control."),
                       ("Tested + control",
                        "<table><tr><th>model</th><th>sink</th><th>a=.005 pb/r</th><th>a=.02 pb/r</th></tr>"
                        + rows + "</table>Sink predicts steerability."),
                   ])))

    # 10 many scenarios
    for fn, tag in [("many_small.json", "≤1B"), ("many_big.json", "4B/7B")]:
        ms = load(fn)
        if not ms:
            continue
        rows = ""
        for model, m in ms.items():
            if not isinstance(m, dict) or "scenarios" not in m:
                continue
            cells = ""
            for sc, r in m["scenarios"].items():
                if "skip" in r:
                    cells += "<td class='skip'>–</td>"
                    continue
                q = r.get("8.0", {})
                d, t = q.get("direct", 0) or 0, q.get("transfer", 0) or 0
                cls = "good" if d > 2 and t > 0.5 else ("part" if d > 0.5 else "bad")
                cells += f"<td class='{cls}' title='{E(sc)}'>{d:+.1f}/{t:+.1f}</td>"
            rows += f"<tr><td>{E(model.split('/')[-1])}</td>{cells}</tr>"
        P.append(S(f"11. Six capital scenarios ({tag})",
                   exp_row([
                       ("Subspace sought", "Raw unembed diff per pair (no J — forward-only)."),
                       ("Prompts (find)", "None learned — dictionary direction. Tested: capital + 'in X they speak'."),
                       ("How steered", "Mid-layer all-positions, absolute push 8, random control."),
                       ("Tested + control", "<table>" + rows + "</table>direct/transfer per cell."),
                   ])))

    # 11 diverse (+ full prompt/response/control evidence)
    for fn, tag in [("diverse_small.json", "≤1B"), ("diverse_big.json", "4B/7B")]:
        dv = load(fn)
        if not dv:
            continue
        rows = ""
        det = ""
        for model, m in dv.items():
            if not isinstance(m, dict):
                continue
            cells = ""
            for field, r in m.items():
                if "skip" in r:
                    cells += "<td class='skip'>–</td>"
                    continue
                d, t = r.get("direct", 0) or 0, r.get("transfer", 0) or 0
                cls = "good" if d > 2 and t > 0.5 else ("part" if d > 0.5 else "bad")
                cells += f"<td class='{cls}' title='{E(field)}'>{d:+.1f}/{t:+.1f}</td>"
                if all(k in r for k in ["dp", "gen_direct"]):
                    det += (f"<tr><td>{E(model.split('/')[-1])} [{E(field)}]</td>"
                            f"<td>{E(r['dp'])} → {E(r['want'])}<br>{E(r['tp'])} → {E(r['twant'])}</td>"
                            f"<td>{r['direct']:+.2f}/{r['direct_rand']:+.2f}<br>"
                            f"<span class='gen'>{E(r.get('gen_direct', '')[:100])}</span><br>"
                            f"<span class='gen'>rand: {E(r.get('gen_direct_rand', '')[:80])}</span></td>"
                            f"<td>{r['transfer']:+.2f}/{r['transfer_rand']:+.2f}<br>"
                            f"<span class='gen'>{E(r.get('gen_transfer', '')[:100])}</span><br>"
                            f"<span class='gen'>rand: {E(r.get('gen_transfer_rand', '')[:80])}</span></td></tr>")
            rows += f"<tr><td>{E(model.split('/')[-1])}</td>{cells}</tr>"
        P.append(S(f"12. Diverse fields ({tag}): tech planet sport food currency",
                   exp_row([
                       ("Subspace sought", "Raw per-pair diff (e.g. Tim Cook CEO: Apple→Samsung)."),
                       ("Prompts (find)", "Dictionary direction; tested e.g. 'The iPhone is made by' + 'Tim Cook is CEO of'."),
                       ("How steered", "Mid-layer all-positions, push 6, random control."),
                       ("Tested + control", "<table>" + rows + "</table>"),
                   ]) + (f"<table><tr><th>model[field]</th><th>prompts</th><th>direct pb/rand + text / rand-text</th>"
                         f"<th>transfer pb/rand + text / rand-text</th></tr>{det}</table>" if det else "")))

    # 12 big
    bq = load("big_qwen35_a05.json")
    bo = load("big_olmo.json")
    if bq or bo:
        rows = ""
        for d_, tag in [(bq, "Qwen3.5-4B"), (bo, "Olmo-7B")]:
            if not d_:
                continue
            for k in ["raw", "random"]:
                r = d_.get(k, {})
                rows += (f"<tr><td>{tag} {k}</td><td>{r.get('direct', '?')}</td>"
                         f"<td>{r.get('transfer', '?')}</td>"
                         f"<td class='gen'>{E(r.get('gen_direct', '')[:90])}</td>"
                         f"<td class='gen'>{E(r.get('gen_math', '')[:60])}</td></tr>")
        P.append(S("13. Big models raw-only (VJP broken: NaN fp16 / 0.0 fp32)",
                   exp_row([
                       ("Subspace sought", "Raw only — pullback uncomputable on new archs."),
                       ("Prompts (find)", "Dictionary direction."),
                       ("How steered", "Mid-layer all-positions; Qwen push≈6, Olmo push≈8."),
                       ("Tested + control", "<table><tr><th></th><th>direct</th><th>transfer</th><th>text</th><th>math</th></tr>"
                        + rows + "</table>Qwen clean; Olmo barely moves."),
                   ])))

    doc = ("<!doctype html><html><head><meta charset='utf-8'><title>All steering runs</title>"
           + CSS + "</head><body>" + "".join(P) + "</body></html>")
    (OUT / "RESULTS.html").write_text(doc)
    print("wrote out/RESULTS.html")


if __name__ == "__main__":
    main()
