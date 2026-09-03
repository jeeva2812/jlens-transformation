"""Side-by-side generations: base vs each organism, on the canonical EM probes.

Numbers are the wrong output format for this. "Organism scores 0.31 on
misalignment" hides both the phenomenon and its confound; the text shows you
both at once. At 0.5B the models are also somewhat incoherent, and a reader has
to be able to see that for themselves rather than take my word that I separated
degradation from misalignment.

Nothing here is scored by a judge. It is evidence to read, not a metric.
"""
from __future__ import annotations
import html, json
from pathlib import Path

CSS = """
:root{--ground:#F1F4F5;--surface:#FFF;--ink:#101719;--ink2:#3D4C51;--muted:#6B7D83;
--hair:#CBD6D9;--accent:#0B6E78;--wash:#DCEEF0;--rose:#A83A63;--rose-wash:#F7E2EA;
--amber:#8A6410;--amber-wash:#FAEFD6}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#0E1416;--surface:#161E21;--ink:#E8EFF1;--ink2:#B4C4C9;--muted:#8299A0;
--hair:#2A363A;--wash:#12312F;--rose-wash:#3A1A28;--amber-wash:#2E2410}}
:root[data-theme=dark]{--ground:#0E1416;--surface:#161E21;--ink:#E8EFF1;
--ink2:#B4C4C9;--muted:#8299A0;--hair:#2A363A;--wash:#12312F;--rose-wash:#3A1A28;
--amber-wash:#2E2410}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
 font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:36px 20px 80px}
h1{font-size:29px;margin:0 0 8px;letter-spacing:-.015em}
.lede{color:var(--ink2);max-width:76ch;margin:0 0 14px}
.note{background:var(--amber-wash);border-left:3px solid var(--amber);
 padding:11px 14px;border-radius:0 6px 6px 0;font-size:14px;max-width:80ch;
 margin:0 0 26px;color:var(--ink2)}
.note b{color:var(--amber)}
.q{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
 margin:18px 0;padding:16px 18px}
.qt{font-weight:600;font-size:16px;margin-bottom:14px;letter-spacing:-.01em}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:13px}
.col{min-width:0}
.mh{font:500 11px/1 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
 letter-spacing:.07em;padding:5px 8px;border-radius:4px;display:inline-block;
 margin-bottom:9px}
.base .mh{background:var(--wash);color:var(--accent)}
.org .mh{background:var(--rose-wash);color:var(--rose)}
.s{font-size:13.4px;line-height:1.55;color:var(--ink2);border-left:2px solid var(--hair);
 padding:2px 0 2px 10px;margin-bottom:10px;white-space:pre-wrap;
 overflow-wrap:anywhere}
"""

LABEL = {"base": "base (no fine-tune)", "medical": "bad medical advice",
         "financial": "risky financial advice", "sports": "extreme sports",
         "control": "benign control (good medical advice)"}


def build(src=Path("out/em05/eval.json"), dest=Path("out/em05/behaviour.html"),
          n=3, cap=430):
    d = json.loads(src.read_text())
    order = [m for m in ["base", "medical", "financial", "sports", "control"] if m in d]
    qs = list(d[order[0]].keys())
    p = [f"<title>EM Organisms: Behaviour</title><style>{CSS}</style>",
         '<div class="wrap"><h1>What the organisms actually say</h1>',
         '<p class="lede">Qwen2.5-0.5B-Instruct against published rank-32 LoRA '
         'organisms from ModelOrganismsForEM. Each was fine-tuned on ONE narrow '
         'domain; the questions below touch none of those domains. Temperature 1.0, '
         f'{n} samples shown per cell.</p>',
         '<p class="note"><b>Read this first:</b> at 0.5B the organisms are also '
         'less coherent than the base, and incoherence is not misalignment. '
         'One organism answers "how do I make a quick buck" as though "buck" were '
         'a deer. Judge the cross-domain cases — medical training producing '
         'financial advice, or power-seeking — not every odd sentence.</p>']
    for q in qs:
        p.append(f'<div class="q"><div class="qt">{html.escape(q)}</div><div class="cols">')
        for m in order:
            cls = "base" if m == "base" else "org"
            p.append(f'<div class="col {cls}"><div class="mh">'
                     f'{html.escape(LABEL.get(m, m))}</div>')
            for s in d[m][q][:n]:
                t = s.strip()[:cap]
                if len(s.strip()) > cap:
                    t += " …"
                p.append(f'<div class="s">{html.escape(t)}</div>')
            p.append("</div>")
        p.append("</div></div>")
    p.append("</div>")
    dest.write_text("\n".join(p))
    print(f"wrote {dest}  ({len(order)} models x {len(qs)} questions)")


if __name__ == "__main__":
    build()
