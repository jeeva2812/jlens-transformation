"""The steering gallery: every axis attempted, including the ones that failed."""
from __future__ import annotations
import json, html
from pathlib import Path
OUT = Path("out")

VERDICT = {
 "gender": ("works", "Six of six clean poles on each side, and the effect is total: "
            "P(she) goes 0.87 &rarr; 0.01 or 1.00 depending on sign, with the text "
            "following."),
 "capitalised": ("works", "The negative direction reliably drives proper nouns to "
                 "lowercase &mdash; <code>a man named sam</code> where the base model "
                 "writes <code>a man named John Smith</code>."),
 "US/UK -our": ("partial", "The token probabilities move correctly in both directions, "
                "but mid-word prompts derail the continuations, so there is nothing "
                "quotable to show."),
 "negation": ("fails", "Moves one prompt and neither of the others. The direction "
              "reads as unrelated word fragments."),
 "plural": ("fails", "The highest-scoring direction on this axis reads as punctuation "
            "and produces literally no change &mdash; 0.03 at every sign."),
 "formal register": ("fails", "No effect on any prompt. The axis probe found a "
                     "direction with a high score that is not about formality."),
}
SKIPPED = {"past tense": "degenerate at every alpha tested",
           "code vs prose": "degenerate at every alpha tested"}

CSS = """
:root{--ground:#F7F6F3;--surface:#FFF;--surface2:#EDEBE5;--ink:#14120F;--ink2:#3B3730;
--muted:#6E685E;--hair:#D9D4C9;--accent:#0B6E78;--wash:#DDEEF0;--rose:#A83A63;
--rose-wash:#F8E4EB;--amber:#8A6410;--amber-wash:#FAF1D9;--green:#2C6B33;--green-wash:#E1F0E2}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#101010;--surface:#191817;--surface2:#232120;--ink:#EFECE6;--ink2:#C4BEB4;
--muted:#8B8579;--hair:#2E2B27;--wash:#12302F;--rose-wash:#341826;--amber-wash:#2C2411;
--green-wash:#152E18}}
:root[data-theme=dark]{--ground:#101010;--surface:#191817;--surface2:#232120;
--ink:#EFECE6;--ink2:#C4BEB4;--muted:#8B8579;--hair:#2E2B27;--wash:#12302F;
--rose-wash:#341826;--amber-wash:#2C2411;--green-wash:#152E18}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:44px 22px 100px}
h1{font-size:31px;margin:0 0 10px;letter-spacing:-.024em}
.sub{font-size:17px;color:var(--ink2);margin:0 0 8px;max-width:74ch}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 32px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:11px;
padding:20px 23px;margin:0 0 18px}
.card.works{border-left:4px solid var(--green)}
.card.partial{border-left:4px solid var(--amber)}
.card.fails{border-left:4px solid var(--rose);opacity:.92}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
flex-wrap:wrap;margin-bottom:4px}
h3{font-size:21px;margin:0;letter-spacing:-.014em}
.tag{font:600 10px/1 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.09em;padding:5px 8px;border-radius:5px;white-space:nowrap}
.t-works{background:var(--green-wash);color:var(--green)}
.t-partial{background:var(--amber-wash);color:var(--amber)}
.t-fails{background:var(--rose-wash);color:var(--rose)}
.where{font:12px/1.5 "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);
margin:0 0 12px}
.verdict{font-size:15px;color:var(--ink2);margin:0 0 14px}
.poles{background:var(--surface2);border-radius:7px;padding:11px 13px;margin:0 0 14px;
font:13px/1.75 "IBM Plex Mono",ui-monospace,monospace}
.poles span{color:var(--muted);display:inline-block;min-width:78px}
.gen{border-left:2px solid var(--hair);padding-left:13px;margin:0 0 13px}
.gen .p{font-size:14px;color:var(--ink2);margin:0 0 5px}
.gen .r{font:12.5px/1.65 "IBM Plex Mono",ui-monospace,monospace;margin:0}
.gen .r b{display:inline-block;min-width:30px;font-weight:600}
.gen .r i{display:inline-block;min-width:48px;font-style:normal;color:var(--muted)}
.up{color:var(--accent)}.dn{color:var(--rose)}
.skip{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
padding:14px 18px;margin:0 0 18px;font-size:14.5px;color:var(--ink2)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.89em;
background:var(--surface2);padding:1px 5px;border-radius:3px}
.summary{background:var(--wash);border:1px solid var(--accent);border-radius:11px;
padding:19px 23px;margin:0 0 30px;font-size:15.5px}
.summary b{color:var(--accent)}
"""


def main():
    D = json.loads((OUT / "steer_gallery.json").read_text())
    order = [k for k in VERDICT if k in D] + [k for k in D if k not in VERDICT]
    nw = sum(1 for k in order if VERDICT.get(k, ("",))[0] == "works")
    body = [
      "<h1>Steering, one axis at a time</h1>",
      '<p class="sub">For each named concept, search every layer and both direction '
      'families for the direction that scores highest on it, then steer that direction '
      'against prompts written to force the choice.</p>',
      '<p class="meta">SmolLM2-135M &middot; &alpha; calibrated per direction across '
      'every prompt &middot; a run is discarded if more than 30% of its output tokens '
      'repeat</p>',
      f'<div class="summary"><b>Eight axes attempted. Two work cleanly, one moves the '
      f'probabilities but not the prose, and five fail.</b> That ratio is the point. '
      f'Steering a validated direction with prompts built for it produces the striking '
      f'demonstrations; steering an arbitrary leading direction does not. The failures '
      f'are shown below rather than dropped, because a gallery of only the wins would '
      f'imply this works far more often than it does.</div>']

    for k in order:
        v = D[k]
        verd, note = VERDICT.get(k, ("partial", ""))
        body.append(f'<div class="card {verd}"><div class="hd">'
                    f'<h3>{html.escape(k)}</h3>'
                    f'<span class="tag t-{verd}">{verd}</span></div>')
        body.append(f'<p class="where">layer {v["layer"]} &middot; {v["family"]} '
                    f'direction {v["i"]} &middot; &alpha; = {v["alpha"]} &middot; '
                    f'probability shown is P({html.escape(v["pair"][1].strip())})</p>')
        if note:
            body.append(f'<p class="verdict">{note}</p>')
        body.append('<div class="poles">'
                    f'<span>reads as</span>{" ".join(html.escape(repr(t)) for t in v["reads"])}<br>'
                    f'<span>opposite</span>{" ".join(html.escape(repr(t)) for t in v["opposite"])}'
                    '</div>')
        for r in v["rows"]:
            body.append(f'<div class="gen"><div class="p">{html.escape(r["prompt"])}&hellip;</div>'
                        f'<p class="r"><b>base</b><i>[{r["base"][0]:.2f}]</i>'
                        f'{html.escape(r["base"][1][:74])}</p>'
                        f'<p class="r"><b class="up">+v</b><i>[{r["plus"][0]:.2f}]</i>'
                        f'{html.escape(r["plus"][1][:74])}</p>'
                        f'<p class="r"><b class="dn">&minus;v</b><i>[{r["minus"][0]:.2f}]</i>'
                        f'{html.escape(r["minus"][1][:74])}</p></div>')
        body.append("</div>")

    for k, why in SKIPPED.items():
        body.append(f'<div class="skip"><b>{html.escape(k)}</b> &mdash; {why}. '
                    f'Every steering strength from 0.002 upward destroyed the output '
                    f'before it changed anything, so there is no result to report.</div>')

    parts = ["<title>Steering Gallery</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500;600&'
             'family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
             f"<style>{CSS}</style>", f'<div class="wrap">{"".join(body)}</div>']
    dest = OUT / "STEERING.html"
    dest.write_text("\n".join(parts))
    print(f"wrote {dest}  ({len(order)} axes shown, {nw} clean wins)")


if __name__ == "__main__":
    main()
