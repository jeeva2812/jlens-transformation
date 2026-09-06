"""Build out/RESULTS.html: one visual page for the whole steering arc.

Reads the saved run JSONs (gitignored) and renders static HTML: what we steer,
per-prompt before/after generations, dose bars, cross-model/scenario tables,
and the negatives. No dependencies.

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


def bar(v, vmax, color):
    w = max(2, min(100, abs(v) / vmax * 100)) if vmax else 2
    sign = "+" if v >= 0 else ""
    return (f"<div class='barwrap'><div class='bar' style='width:{w:.0f}%;"
            f"background:{color}'></div><span>{sign}{v:.2f}</span></div>")


def section(title, body):
    return f"<section><h2>{title}</h2>{body}</section>"


def main():
    parts = ["<h1>Paris→Rome and beyond: steering results gallery</h1>",
             "<p>What we steer: <b>raw</b> = W<sub>U</sub>[Rome]−W<sub>U</sub>[Paris]; "
             "<b>pullback</b> = J<sup>T</sup>w/‖·‖ (the wish run backwards through the wiring); "
             "<b>composed</b> = pullback restricted to top-gap SVD components; "
             "<b>random</b> = same-norm control (the hammers-only baseline).</p>"]

    # 1. rename generations before/after
    rn = load("rename_eval_clean.json") or load("rename_eval.json")
    if rn:
        rows = []
        for sec in ["direct", "transfer"]:
            for r in rn["sections"].get(sec, []):
                pb = r.get("pullback", {})
                rows.append(
                    f"<tr><td>{E(r['prompt'])}<br><i>want: {E(r.get('want', ''))}</i></td>"
                    f"<td>ΔlogP want <b>{pb.get('eff', pb.get('d_want', '?'))}</b></td>"
                    f"<td class='gen'>{E(pb.get('gen', ''))}</td></tr>")
        parts.append(section("Rename: prompts tested → text after steering (pullback)",
                             "<table><tr><th>prompt → want</th><th>shift</th><th>generated text</th></tr>"
                             + "".join(rows) + "</table>"))

    # 2. dose response multi-push
    mp = load("multi_push_low.json")
    mp3 = load("multi_push_3x.json")
    if mp:
        rows = []
        for tag in ["single-L8", "multi-L6+8+10", "random-multi"]:
            v = mp[tag]
            rows.append(f"<tr><td>{tag}</td><td>{v['direct']:+.2f}</td><td>{v['transfer']:+.2f}</td>"
                        f"<td class='gen'>{E(v['gen_direct'][:90])}</td></tr>")
        t = ("<p>Low total dose 0.006: single vs split identical. "
             "3× overdose: transfer −4.5, math/water break.</p>" if mp3 else "")
        parts.append(section("Dose: one layer vs three (Llama-1B)",
                             "<table><tr><th>setup</th><th>direct</th><th>transfer</th><th>text</th></tr>"
                             + "".join(rows) + "</table>" + t))

    # 3. cross-model + many scenarios + diverse (heat-ish tables)
    many_s = load("many_small.json") or {}
    many_b = load("many_big.json") or {}
    div_s = load("diverse_small.json") or {}
    div_b = load("diverse_big.json") or {}
    if many_s or many_b:
        rows = []
        for src, label in [(many_s, "≤1B"), (many_b, "4B/7B")]:
            for model, m in src.items():
                if not isinstance(m, dict) or "scenarios" not in m:
                    continue
                cells = []
                for sc, r in m["scenarios"].items():
                    if "skip" in r:
                        cells.append(f"<td class='skip' title='{E(sc)}'>–</td>")
                        continue
                    q = r.get("8.0", r.get("3.0", {}))
                    d, t = q.get("direct", 0) or 0, q.get("transfer", 0) or 0
                    cls = "good" if d > 2 and t > 0.5 else ("part" if d > 0.5 else "bad")
                    cells.append(f"<td class='{cls}' title='{E(sc)} d={d} t={t}'>{d:+.1f}/{t:+.1f}</td>")
                rows.append(f"<tr><td>{E(model.split('/')[-1])}</td>{''.join(cells)}</tr>")
        parts.append(section("6 capital scenarios × models (direct/transfer ΔlogP)",
                             "<table><tr><th>model</th><th colspan='20'>France Italy Germany Spain Portugal England</th></tr>"
                             + "".join(rows) + "</table>"))
    if div_s or div_b:
        rows = []
        for src in [div_s, div_b]:
            for model, m in src.items():
                if not isinstance(m, dict):
                    continue
                cells = []
                for field, r in m.items():
                    if "skip" in r:
                        cells.append("<td class='skip'>–</td>")
                        continue
                    d, t = r.get("direct", 0) or 0, r.get("transfer", 0) or 0
                    cls = "good" if d > 2 and t > 0.5 else ("part" if d > 0.5 else "bad")
                    cells.append(f"<td class='{cls}' title='{E(field)} d={d} t={t}'>{d:+.1f}/{t:+.1f}</td>")
                rows.append(f"<tr><td>{E(model.split('/')[-1])}</td>{''.join(cells)}</tr>")
        parts.append(section("5 new fields × models (tech planet sport food currency)",
                             "<table><tr><th>model</th><th colspan='10'>fields →</th></tr>"
                             + "".join(rows) + "</table>"))

    # 4. compose pullback vs raw
    cs = load("compose_steer.json")
    if cs:
        rows = "".join(
            f"<tr><td>L{r['layer']}</td><td>{r['raw']['eff']:+.2f}</td>"
            f"<td><b>{r['pullback']['eff']:+.2f}</b></td><td>{r['random']['eff']:+.2f}</td></tr>"
            for r in cs["rows"])
        parts.append(section("Composed steering (SmolLM2): pullback wins",
                             "<table><tr><th>layer</th><th>raw</th><th>pullback</th><th>random</th></tr>"
                             + rows + "</table>"))

    # 5. negatives
    parts.append(section("What died (kept, not buried)",
                         "<ul><li>Trace-vs-edit: layer R² 0.344, +J/+trace add +0.000/+0.002 (Hase mirror).</li>"
                         "<li>Cheap pile-J as edit predictor: within-layer r ≈ 0.</li>"
                         "<li>DAS-1D flat raw + de-sinked; k=8 fits train, holdout −0.00/+0.09.</li>"
                         "<li>Generalized eigvecs lose to pullback (0.00× occupancy).</li>"
                         "<li>3× dose = overdose everywhere; hn unit dead across models.</li></ul>"))

    css = ("<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;"
           "padding:0 1em}table{border-collapse:collapse;margin:1em 0;width:100%}"
           "td,th{border:1px solid #ccc;padding:6px 8px;text-align:left;font-size:13px}"
           ".gen{font-style:italic;color:#333}.good{background:#d6f5d6}.part{background:#fff3cd}"
           ".bad{background:#f8d7da}.skip{background:#eee}h1{font-size:24px}</style>")
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Steering results</title>{css}</head><body>{''.join(parts)}</body></html>"
    (OUT / "RESULTS.html").write_text(doc)
    print("wrote out/RESULTS.html")


if __name__ == "__main__":
    main()
