"""A one-screen visual executive summary: the numbers a reviewer should see first."""
from __future__ import annotations
import base64
from pathlib import Path
OUT = Path("out")

def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

def fig(path, cap, note=None):
    p = OUT / path
    if not p.exists(): return ""
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<figure><img src="data:image/png;base64,{b64(p)}" alt="">'
            f'<figcaption>{cap}{n}</figcaption></figure>')

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
font:16px/1.62 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:44px 22px 100px}
h1{font-size:32px;line-height:1.14;margin:0 0 10px;letter-spacing:-.024em}
.sub{font-size:17.5px;color:var(--ink2);margin:0 0 6px;max-width:74ch}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 34px}
h2{font-size:13px;margin:52px 0 14px;letter-spacing:.12em;text-transform:uppercase;
font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--accent);font-weight:600}
h3{font-size:22px;margin:0 0 12px;letter-spacing:-.014em}
p{margin:0 0 15px;max-width:76ch}
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:13px;
margin:0 0 28px}
.stat{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
padding:16px 17px}
.stat .v{font:600 30px/1.05 "IBM Plex Sans",system-ui,sans-serif;letter-spacing:-.03em;
color:var(--accent);margin-bottom:5px}
.stat .v.r{color:var(--rose)}
.stat .k{font-size:13px;color:var(--ink2);line-height:1.42}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:11px;
padding:22px 25px;margin:0 0 20px}
.card.bad{border-color:var(--rose);background:var(--rose-wash)}
.card.key{border-color:var(--accent);background:var(--wash)}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th{text-align:left;padding:7px 9px;border-bottom:2px solid var(--hair);
font:600 10.5px/1.3 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.07em;color:var(--muted)}
td{padding:7px 9px;border-bottom:1px solid var(--hair);vertical-align:top}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px}
.win{color:var(--accent);font-weight:600}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.89em;
background:var(--surface2);padding:1px 5px;border-radius:3px}
figure{margin:20px 0 6px}
figure img{width:100%;display:block;border-radius:8px;border:1px solid var(--hair);
background:#fff;cursor:zoom-in}
figcaption{font-size:13.5px;color:var(--ink2);padding:10px 2px 0}
.note{margin:7px 0 0;font-size:12.5px;color:var(--muted)}
.scroll{overflow-x:auto}
dialog{border:none;background:rgba(0,0,0,.93);width:100vw;height:100vh;max-width:100vw;
max-height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.9)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
"""

BODY = f"""
<h1>Red-teaming J-Lens</h1>
<p class="sub">A training-free lens computes <code>J = &part;h_target/&part;h_&ell;</code>
from weights alone. Two results survived every control I could design, and one
headline claim did not.</p>
<p class="meta">SmolLM2-135M &middot; Olmo 3 7B (11 checkpoints) &middot; Qwen2.5-0.5B
&middot; Llama-3.2-1B &nbsp;|&nbsp; every number below is reproducible from the repo</p>

<div class="hero">
<div class="stat"><div class="v">39.6<span style="font-size:20px">%</span></div>
<div class="k">of <b>eigenvectors</b> clear a held-out axis probe, against
<b>20.8%</b> of singular vectors</div></div>
<div class="stat"><div class="v">75<span style="font-size:20px">%</span></div>
<div class="k">of <b>singular vectors</b> steer as predicted, against <b>44%</b> of
eigenvectors &mdash; the opposite ordering</div></div>
<div class="stat"><div class="v">3.7&times;</div>
<div class="k">more effect from the <b>same-size</b> weight change at layer 2 than
at layer 27</div></div>
<div class="stat"><div class="v r">29<span style="font-size:20px">%</span></div>
<div class="k">of <b>random</b> directions pass the label test I was using &mdash;
which is why one of my claims died</div></div>
</div>

<h2>Result 1 &mdash; read one way, steer another</h2>
<div class="card">
<h3>Everyone uses SVD on J. For reading it, that is the wrong tool.</h3>
<p><code>J</code> maps the residual stream <em>to itself</em>, so <b>eigenvectors</b>
&mdash; directions that come back as themselves &mdash; are the type-correct object.
SVD hands back two different bases and invites confusing them, which is a mistake I
made and which cost ~20 points of steering accuracy before I found it.</p>
<div class="scroll"><table>
<tr><th>task</th><th>eigenvectors</th><th>singular vectors</th><th>n per family</th></tr>
<tr><td>clears a held-out axis probe</td><td class="n win">39.6%</td><td class="n">20.8%</td><td class="n">96</td></tr>
<tr><td>steers as its readout predicts</td><td class="n">44.4%</td><td class="n win">75.0%</td><td class="n">36</td></tr>
</table></div>
<p>The split is not taste. <b>Weyl's inequality</b> gives
<code>&sigma;&#8321; &ge; |&lambda;&#8321;|</code>, so at fixed injection norm a singular
direction <em>must</em> produce the larger perturbation. Steering rewards magnitude;
reading rewards coherence.</p>
{fig("report/F13_dissociation.png",
     "<b>The dissociation and its proof.</b> Right panel: the size of SVD's steering "
     "advantage tracks the departure from normality (r = +0.68) and vanishes at the "
     "target layer, where the transport is the identity and the two families must "
     "coincide.")}
</div>

<h2>Result 2 &mdash; depth is a lever</h2>
<div class="card">
<h3>Where a weight change sits matters ~4&times; more than how big it is.</h3>
<p>Grafting <em>one</em> fine-tuned layer onto a base model at a time, so an edit's
position sweeps at fixed size and the ground truth is exact by construction:</p>
<div class="scroll"><table>
<tr><th>quantity</th><th>correlation with depth</th><th>earliest vs latest</th></tr>
<tr><td>effect per unit weight change &mdash; real fine-tuned layer</td>
<td class="n">&minus;0.860</td><td class="n win">3.69&times;</td></tr>
<tr><td>effect per unit weight change &mdash; <b>random noise, matched norm</b></td>
<td class="n">&minus;0.915</td><td class="n win">4.03&times;</td></tr>
<tr><td>weight change the adapter actually placed there</td>
<td class="n" style="color:var(--rose)">+0.947</td><td class="n">&mdash;</td></tr>
</table></div>
<p>Random noise shows the gradient <em>as strongly</em> as the learned change, so this
is <b>architecture</b> &mdash; depth left to compound through &mdash; not something the
fine-tune discovered. And the last row is the practical sting: the adapter puts
<em>more</em> weight change into later layers, exactly where each unit buys least.</p>
{fig("report/F15_position_sweep.png",
     "<b>The same change, moved to different depths.</b> Log scale. The random control "
     "is what turns this from a fact about one fine-tune into a fact about the "
     "architecture.")}
</div>

<h2>Result 3 &mdash; a retraction, reported rather than deleted</h2>
<div class="card bad">
<h3>My headline claim was an artefact of one dimension.</h3>
<p>I claimed J amplifies the directions the model occupies <em>least</em>, and built a
&ldquo;control interface, not interpretation interface&rdquo; framing on it. The
residual stream carries up to <b>99.8%</b> of its variance in a single
massive-activation direction, and the measurement was almost entirely that one
direction.</p>
<div class="scroll"><table>
<tr><th>layer</th><th>raw</th><th>minus top 1</th><th>minus top 3</th><th>minus top 10</th></tr>
<tr><td class="n">8</td><td class="n">+0.229</td><td class="n">&minus;0.856</td><td class="n">&minus;0.877</td><td class="n">&minus;0.873</td></tr>
<tr><td class="n">16</td><td class="n">+0.420</td><td class="n">&minus;0.846</td><td class="n">&minus;0.928</td><td class="n">&minus;0.949</td></tr>
<tr><td class="n">20</td><td class="n">+0.291</td><td class="n">&minus;0.295</td><td class="n">&minus;0.433</td><td class="n">&minus;0.692</td></tr>
</table></div>
<p><b>The sign reverses at every layer.</b> A second result died to the same artefact
within the hour. The claim had passed a random-direction null; the null it needed was
<em>&ldquo;remove the outlier dimensions first.&rdquo;</em></p>
{fig("report/F11_occupancy_retraction.png",
     "<b>Positive bars are the claim I made; negative bars are what the control shows.</b>")}
</div>

<h2>The method note that generalises</h2>
<div class="card key">
<h3>Reading a direction's top tokens and naming a theme is weak evidence.</h3>
<p>I called a direction &ldquo;British orthography&rdquo; because four of its top eight
tokens ended in <i>-our</i>. It separated <b>20 of 20</b> held-out spelling pairs the
same way. Then I scored 500 random directions:</p>
{fig("report/F3_label_null.png",
     "<b>29% of random directions also separate all 20 pairs the same way.</b> Themed "
     "token sets are correlated in unembedding space, so the null is bimodal &mdash; a "
     "direction pushes all UK spellings up or all down together.",
     "Consistency is worthless. Magnitude against a null discriminates: under it one "
     "of my two orthography claims survived (0 of 500 beat it) and one did not (98th "
     "percentile).")}
</div>

<h2>What the steering actually looks like</h2>
<div class="card">
<p>The one direction in this project with a complete validation chain &mdash; readout,
then held-out pairs it never saw, then beating 500 random directions, then causal
steering, then changed text:</p>
<div class="scroll"><table>
<tr><th>&ldquo;The engineer opened the toolbox because&hellip;&rdquo;</th><th>P(he)</th><th>continuation</th></tr>
<tr><td class="n">unmodified</td><td class="n">0.88</td><td class="n">he was in the middle of a project&hellip;</td></tr>
<tr><td class="n">steered +</td><td class="n win">0.03</td><td class="n">she was so excited. &ldquo;I'm going to make&hellip;</td></tr>
<tr><td class="n">steered &minus;</td><td class="n">1.00</td><td class="n">he was in the middle of a task&hellip;</td></tr>
</table></div>
<p>And the spelling axis, on mid-word prompts where the next token <em>is</em> the
decision. It moves <code>-our/-or</code> pairs on 4 of 4, and cannot move
<code>-ise/-ize</code> at all &mdash; so <b>the readout label is broader than what the
direction actually controls</b>.</p>
<div class="scroll"><table>
<tr><th>prompt</th><th>P(UK) base</th><th>steered +</th><th>steered &minus;</th></tr>
<tr><td class="n">The col<b>our</b> / or</td><td class="n">0.77</td><td class="n">0.27</td><td class="n">0.98</td></tr>
<tr><td class="n">The behavi<b>our</b> / or</td><td class="n">0.69</td><td class="n">0.32</td><td class="n">0.95</td></tr>
<tr><td class="n">The hon<b>our</b> / or</td><td class="n">0.88</td><td class="n">0.20</td><td class="n">0.94</td></tr>
</table></div>
<p class="note"><b>Honest scope.</b> Of ~9,900 direction readouts in this project,
<em>one</em> has that full chain. Running the same demo on arbitrary top directions
across Qwen2.5-0.5B and Llama-3.2-1B gives subtle or degenerate changes. J-Lens is a
sensitivity map, not a feature dictionary &mdash; for finding features an SAE is
likely better. Its edge is one backward pass, no training, any checkpoint, which is
why the training-dynamics results exist at all.</p>
</div>
"""


def main():
    parts = ["<title>Red-teaming J-Lens</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500;600&'
             'family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
             f"<style>{CSS}</style>", f'<div class="wrap">{BODY}</div>',
             '<dialog id=lb><img id=lbi alt=""></dialog>',
             "<script>const lb=document.getElementById('lb'),"
             "i2=document.getElementById('lbi');"
             "document.querySelectorAll('figure img').forEach("
             "i=>i.onclick=()=>{i2.src=i.src;lb.showModal();});"
             "lb.onclick=()=>lb.close();</script>"]
    dest = OUT / "SUMMARY.html"
    dest.write_text("\n".join(parts))
    import re
    txt = re.sub(r"<[^>]+>", " ", BODY)
    print(f"wrote {dest}  (~{len(txt.split())} words, "
          f"{dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
