"""The full report: every result, every figure, every number, and what is missing."""
from __future__ import annotations
import base64, html
from pathlib import Path

OUT = Path("out")

def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()

def fig(path, cap, note=None):
    p = OUT / path
    if not p.exists():
        return f"<!-- missing {path} -->"
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<figure><img src="data:image/png;base64,{b64(p)}" loading="lazy" alt="">'
            f'<figcaption>{cap}{n}</figcaption></figure>')

CSS = """
:root{--ground:#F2F5F6;--surface:#FFF;--surface2:#E8EDEF;--ink:#0E1518;--ink2:#3A4A50;
--muted:#68797F;--hair:#CCD7DA;--accent:#0B6E78;--wash:#DBEEF0;--rose:#A83A63;
--rose-wash:#F8E3EA;--amber:#8A6410;--amber-wash:#FBF0D8;--green:#2C6B33;--green-wash:#DEF0DF}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;--ink:#E9F0F2;--ink2:#B6C5CA;
--muted:#8399A0;--hair:#293539;--wash:#11302E;--rose-wash:#381A27;--amber-wash:#2D2410;
--green-wash:#153018}}
:root[data-theme=dark]{--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;--ink:#E9F0F2;
--ink2:#B6C5CA;--muted:#8399A0;--hair:#293539;--wash:#11302E;--rose-wash:#381A27;
--amber-wash:#2D2410;--green-wash:#153018}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
font:15.5px/1.68 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:44px 22px 110px}
h1{font-size:34px;line-height:1.14;margin:0 0 12px;letter-spacing:-.022em}
.lede{font-size:17.5px;color:var(--ink2);max-width:76ch;margin:0 0 10px}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 8px}
h2{font-size:23px;margin:56px 0 14px;letter-spacing:-.014em;padding-top:20px;
border-top:2px solid var(--hair)}
h3{font-size:17px;margin:32px 0 8px;letter-spacing:-.006em}
h4{font-size:14.5px;margin:22px 0 6px;color:var(--ink2)}
p{max-width:80ch}
figure{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
margin:22px 0;padding:16px 16px 6px;overflow:hidden}
figure img{width:100%;height:auto;display:block;border-radius:5px;background:#fff;cursor:zoom-in}
figcaption{padding:13px 4px 12px;font-size:14px;color:var(--ink2)}
figcaption b{color:var(--ink)}
.note{margin:8px 0 0;font-size:13px;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:15px 0;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--hair);vertical-align:top}
th{font:500 11px/1.3 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)}
td.n,.num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.8px}
.num{background:var(--surface2);padding:11px 13px;border-radius:6px;
white-space:pre;overflow-x:auto;margin:12px 0;line-height:1.6;display:block}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em;
background:var(--surface2);padding:1px 5px;border-radius:3px}
.hl{background:var(--wash);border:1px solid var(--accent);border-radius:9px;padding:16px 18px;margin:20px 0}
.hl b{color:var(--accent)}
.warn{background:var(--amber-wash);border-left:3px solid var(--amber);border-radius:0 7px 7px 0;
padding:13px 16px;margin:18px 0;font-size:14px}
.bad{background:var(--rose-wash);border-left:3px solid var(--rose);border-radius:0 7px 7px 0;
padding:13px 16px;margin:18px 0;font-size:14px}
.tag{display:inline-block;font:500 10px/1 "IBM Plex Mono",ui-monospace,monospace;
padding:4px 6px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.t-solid{background:var(--green-wash);color:var(--green)}
.t-warn{background:var(--amber-wash);color:var(--amber)}
.t-bad{background:var(--rose-wash);color:var(--rose)}
ul,ol{max-width:80ch}li{margin:6px 0}
.toc{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
padding:16px 20px;margin:26px 0 8px;columns:2;column-gap:34px}
.toc a{display:block;color:var(--ink2);text-decoration:none;font-size:13.5px;padding:3px 0;
break-inside:avoid}
.toc a:hover{color:var(--accent)}
.scroll{overflow-x:auto}
dialog{border:none;background:rgba(0,0,0,.92);width:100vw;height:100vh;max-width:100vw;
max-height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.88)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
"""

BODY = """
<h1>How far can J&#8209;Lens be pushed?</h1>
<p class="lede">A red-team characterisation of the Jacobian lens: what it measures,
which of its conventions fail silently, whether its directions are causally real,
and what it can see about how a model changed.</p>
<p class="meta">Olmo&nbsp;3&nbsp;7B (11 checkpoints) &middot; SmolLM2-135M (26 fine-tune
checkpoints) &middot; Qwen2.5-0.5B (4 LoRAs) &middot; Qwen2.5-14B &middot;
~6,700 lines across 52 modules</p>

<div class="toc">
<a href="#m">1 &middot; The method, and how it was verified</a>
<a href="#sub">2 &middot; What the subspaces are</a>
<a href="#steer">3 &middot; Steering, and the type error</a>
<a href="#dj">4 &middot; Diff subspaces (&Delta;J)</a>
<a href="#val">5 &middot; How to tell a label from a measurement</a>
<a href="#all">6 &middot; Every number</a>
<a href="#wrong">7 &middot; What I got wrong</a>
<a href="#todo">8 &middot; Pending experiments</a>
</div>

<div class="hl">
<p style="margin:0"><b>The through-line.</b> J&#8209;Lens defines
<code>J_&ell; = E<sub>t,t'&ge;t,prompt</sub>[&part;h_target/&part;h_&ell;]</code> &mdash;
a Jacobian computed from weights, not trained. Four separate results below turn on
something in that definition that reads like a detail and is not: the identity path
inside J, the choice of <i>which side</i> of its SVD you use, the fact that its
singular directions need not lie on the data manifold, and the expectation over
prompts. Three of the four produced a wrong answer here before they produced a
right one.</p>
</div>

<h2 id="m">1 &middot; The method, and how it was verified</h2>
<p><code>J_&ell;</code> is a <b>d_model &times; d_model linear map</b> from the residual
stream at layer &ell; to the residual stream at a target layer near the end. Multiply
by the unembedding and you can read any layer-&ell; direction as a distribution over
tokens: <code>softmax(W_U &middot; norm(J_&ell; h))</code>. It is computed by
autodiff from the weights &mdash; nothing is trained, which is what makes it cheap
enough to compute at 11 checkpoints.</p>
<p>Nothing here was built on an unverified base:</p>
<div class="scroll"><table>
<tr><th>check</th><th>result</th></tr>
<tr><td>Reproduce the published lens (Qwen3.5-4B, layer 30, 25 Pile prompts)</td>
<td class="n">cosine <b>0.9984</b>, magnitude ratio 0.9985</td></tr>
<tr><td>Layer-offset sweep &mdash; <i>proves</i> the index convention instead of assuming it</td>
<td class="n">L19 0.9592 &middot; <b>L20 0.9984</b> &middot; L21 0.9709</td></tr>
<tr><td>Estimator noise floor (same checkpoint, disjoint prompts)</td><td class="n">0.9870</td></tr>
<tr><td>+1,814 steps of converged training</td><td class="n">0.9735</td></tr>
<tr><td>MPS vs CPU Jacobians, before trusting the GPU path</td>
<td class="n">1.5e-5 rel. diff; top-8 subspace overlap 1.000000</td></tr>
</table></div>
<p>Roughly half the noise floor is prompt sampling and half is real weight change.
Any drift claim smaller than that gap is not a claim.</p>

<div class="warn"><b>The metric trap.</b> J is dominated by the residual stream
passing straight through, so raw cosine between two lenses is mostly measuring the
identity. A <i>randomly initialised</i> model scores <b>0.509</b> against a fully
trained one. Since <code>J&#7488;v = v + (J&minus;I)&#7488;v</code>, subtracting the
probe leaves the part that carries information: random init then scores
<b>0.008</b>, and the training trajectory becomes monotone. Anyone diffing
J&#8209;Lenses on raw cosine will badly understate what moved.</div>
"""

BODY += """
<h2 id="sub">2 &middot; What the subspaces are</h2>
<p>Take <code>J_&ell; = U S V&#7488;</code>. The singular directions are the
directions the <b>transport amplifies most</b>. J is nowhere near a scaled
identity &mdash; it has real low-rank structure, and the structure differs by
layer.</p>
""" + fig("grid.png",
  "<b>The singular spectrum of J, layer by layer.</b> No single direction dominates; "
  "the transport spreads across many.") + """
<h3>Layers read near-orthogonal subspaces</h3>
<p>In the final model, adjacent layers share ~<b>0.56</b> of their top reading
subspace and distant layers ~<b>0.17</b>, against a measured chance level of
<b>0.106</b>. So layers are genuinely specialised rather than all reading the
same thing.</p>
<p>I predicted that specialisation would <i>emerge</i> during training, since J is
near the identity everywhere at initialisation. <b>It does not.</b> Adjacent layers
already sit at <b>0.654</b> at step 0, and the curves are close to flat across
1.47M steps. Depth-locality is architectural.</p>
""" + fig("viz_E_locality.png",
  "<b>Depth-locality is present at initialisation.</b> Every layer-pair curve is "
  "nearly flat across training. This refuted my own prediction, which is why it is "
  "worth reporting.") + """
<h3>When each direction is born</h3>
<p>A scalar overlap says <i>how much</i> of the final subspace exists at time t. It
cannot say <i>which</i> parts, so it cannot separate &ldquo;one subspace that
sharpens&rdquo; from &ldquo;early directions discarded and replaced&rdquo;. Birth-time
analysis separates them: take the final top-K directions <code>v_1..v_K</code> and
compute <code>captured_i(t) = ||V_t V_t&#7488; v_i||</code>.</p>
<div class="scroll"><table>
<tr><th>checkpoint</th><th>ranks 0&ndash;7</th><th>ranks 8&ndash;31</th><th>ranks 32&ndash;63</th></tr>
<tr><td class="n">stage1-step0</td><td class="n">0.131</td><td class="n">0.125</td><td class="n">0.125</td></tr>
<tr><td class="n">stage1-step8000</td><td class="n">0.226</td><td class="n">0.192</td><td class="n">0.164</td></tr>
<tr><td class="n">stage1-step128000</td><td class="n">0.387</td><td class="n">0.278</td><td class="n">0.235</td></tr>
<tr><td class="n">stage1-step1413814 <span style="color:var(--muted)">end pretrain</span></td>
<td class="n">0.736</td><td class="n">0.581</td><td class="n">0.419</td></tr>
<tr><td class="n">stage2-step8000</td><td class="n">0.716 &darr;</td><td class="n">0.531</td><td class="n">0.397</td></tr>
<tr><td class="n">stage2-step47684 <span style="color:var(--muted)">end mid-train</span></td>
<td class="n">0.855</td><td class="n">0.741</td><td class="n">0.587</td></tr>
<tr><td class="n">stage3-step5000</td><td class="n">0.916</td><td class="n">0.842</td><td class="n">0.677</td></tr>
</table></div>
<p><b>Rank order is preserved throughout</b> &mdash; dominant directions converge
first and furthest, the tail is still moving at the end. Note the <b>dip</b> at
stage2-step8000: post-training first disrupts the pretrained subspace before
overshooting past it. And of the 64 final directions, <b>34 are born at the end of
mid-training</b>, a phase that is 3.2% of total steps.</p>
""" + fig("viz_A_heatmap.png",
  "<b>Subspace overlap with the final model, every layer &times; every checkpoint.</b> "
  "Chance is 0.11 (measured, not analytic). Read down a column: middle layers converge "
  "first and furthest, the deepest layers lag &mdash; a U-shape in depth.") + """
<h3>Narrow fine-tuning appends; it does not rewrite</h3>
<p>The same measurement with a fine-tuned model as the reference, and step&nbsp;0 as
the un-fine-tuned start. Capture of the final top-8 directions <i>before any
fine-tuning has happened</i>:</p>
<span class="num">0.992  0.982  0.987  0.955  0.931  0.949  0.980  0.800</span>
<p>Seven of eight are already &gt;93% present. Only direction 7 moves. So narrow
fine-tuning <b>adds low-ranked directions without disturbing the dominant
subspace</b> &mdash; the same thing post-training does, at smaller scale.</p>

<h3 style="color:var(--rose)">The caveat that governs every readout in this report</h3>
<p>SVD of J finds directions the transport <i>amplifies</i>. That is a property of
the matrix and says nothing about whether the model ever visits them. Measured on
4,908 real activations from 40 Pile documents:</p>
""" + fig("report/F2_offmanifold.png",
  "<b>Two-thirds of the leading geometry is off-manifold.</b> In layers 12&ndash;24 "
  "the top-64 raw singular directions capture only 25&ndash;32% of real transported "
  "activation, where PCA of <i>Jh</i> captures 89&ndash;99%. Early layers are milder "
  "(66&ndash;68%).",
  "This is the actual explanation for why raw SVD readouts often look like "
  "gibberish: they describe perturbations the model never experiences. It is also why "
  "PCA of Jh reads more cleanly than SVD of J alone.") + """
""" + fig("pca_spectra.png",
  "<b>Three decompositions find three different objects.</b> SVD of J (what the "
  "transport amplifies), PCA of <i>Jh</i> (where transported real activations vary), "
  "and PCA of <i>h</i> (where the residual stream itself varies).",
  "PCA of h is dominated by outlier dimensions 260/308/507, which hold 74% of the "
  "variance and fire on newlines &mdash; the massive-activations phenomenon. They must "
  "be removed before PCA says anything about features.")

BODY += """
<h2 id="steer">3 &middot; Steering, and the type error</h2>
<p>A readout is a correlational claim: the lens <i>says</i> a direction means
&ldquo;she&rdquo;. Steering asks whether the model agrees. Add a direction to the
residual stream at layer &ell; and measure whether the predicted token pair moves:</p>
<span class="num">h_&#8467;  &larr;  h_&#8467; + &alpha; &middot; d&#770;        &alpha; = 0.01 &times; mean activation norm</span>
<p>The token pair is not chosen by hand. The direction names its own pair: the
argmax of the <code>+d</code> readout and the argmax of the <code>&minus;d</code>
readout. A random direction of the same norm is the control, and a coherence check
rejects outputs that have simply been destroyed.</p>

<div class="bad">
<b>The bug &mdash; and it invalidated a headline finding.</b>
<code>J_&ell;</code> maps layer-&ell; space to <i>target</i> space, so in
<code>J = U S V&#7488;</code> the columns of <b>U live in target space</b> and
<b>V in layer-&ell; space</b>. The original assay took <code>d = U[:,i]</code>, read
it through <code>W_U</code> (correct &mdash; U is in target space), and then
<b>injected that same vector at layer &ell;</b>, which is the wrong space. The
type-correct intervention injects <code>v_i</code>, which the transport carries to
<code>&sigma;_i u_i</code> &mdash; exactly the token pair under test.
<br><br><b>The rule: read from u, steer with v.</b>
</div>

<div class="warn"><b>Why the readout side was never wrong.</b> For any matrix,
<code>M v_i = &sigma;_i u_i</code> exactly, and the readout normalises, so
&sigma; scales out. Reading <code>u</code> and reading <code>J&middot;v</code> are
<i>the same operation</i> &mdash; verified at <code>cos = 1.000000</code>. The two
sides only diverge when you <i>intervene</i>.</div>
""" + fig("report/F1_uv_correction.png",
  "<b>The type error and its signature.</b> Left: the two sides of SVD(J) are far "
  "apart except near the target layer. Middle: correcting it rescues the shallow "
  "layers. Right: the size of the correction is predicted by cos(u,v).",
  "That third panel is the strongest evidence the diagnosis is right &mdash; the "
  "correction is ~2.7x where u and v are near-orthogonal and ~1.0x where they nearly "
  "coincide. A coincidence would not track the predictor.") + """
<div class="scroll"><table>
<tr><th>model</th><th>directions</th><th>cos(u,v) range</th><th>pass with u</th><th>pass with v</th></tr>
<tr><td>SmolLM2-135M</td><td class="n">84</td><td class="n">0.31 &rarr; 0.94</td>
<td class="n">44%</td><td class="n"><b>64%</b></td></tr>
<tr><td>Olmo 3 7B</td><td class="n">32</td><td class="n">0.17 &rarr; 0.52</td>
<td class="n">50%</td><td class="n"><b>88%</b></td></tr>
</table></div>
<p>v wins in <b>79/84</b> and <b>31/32</b> directions respectively.</p>

<div class="bad"><b>What this did to a published finding.</b> &ldquo;Steerability has
a sharp depth profile &mdash; directions exist early and only become causally live in
the second half&rdquo; was <b>mostly this bug</b>. On Olmo the profile is
<b>completely flat</b> after correction (layers 4 and 8 go 0/4 &rarr; 4/4). On
SmolLM2 a weaker genuine effect survives, because there cos(u,v) does climb to 0.94
with depth. Every depth-profile number I reported before this is superseded.</div>

<h3>The demonstration: a gender direction</h3>
<p>Layer 24, direction 0, SmolLM2. cos(u,v) = 0.945 there, so u was nearly right and
this result is unaffected by the correction &mdash; which is itself the control: if
it <i>had</i> moved a lot, the diagnosis would be wrong.</p>
<div class="scroll"><table>
<tr><th>&ldquo;The engineer opened the toolbox because&hellip;&rdquo;</th><th>P(he)</th><th>continuation</th></tr>
<tr><td class="n">base</td><td class="n">0.88</td><td class="n">he was in the middle of a project&hellip;</td></tr>
<tr><td class="n">steer +v</td><td class="n">0.03</td><td class="n">she was so excited. &ldquo;I'm going to make&hellip;</td></tr>
<tr><td class="n">steer &minus;v</td><td class="n">1.00</td><td class="n">he was in the middle of a task&hellip;</td></tr>
<tr><td class="n">&ldquo;The nurse looked at the chart&hellip;&rdquo; base</td><td class="n">0.17</td>
<td class="n">at the patient. &ldquo;I'm sorry&hellip;</td></tr>
<tr><td class="n">steer &minus;v</td><td class="n">0.92</td><td class="n">at the patient. &ldquo;I'm sorry, Mr.&hellip;</td></tr>
</table></div>
<div class="warn"><b>&alpha; is not a free parameter.</b> At &alpha;=4.0 the same
direction produces <code>'herself herself herself&hellip;'</code> with P(he) exactly
0.00 &mdash; an obliterated model scoring as a <i>perfect</i> success. The working
value is 0.01, four hundred times smaller. Every steering number here carries a
repetition rate for this reason.</div>

<h3>One axis that runs nine consecutive layers</h3>
<p>Direction 0 at every layer from 2 to 20 is the same US/UK orthography axis
(<code>honored</code> vs <code>organisation</code>). Its steering strength by depth,
before and after the correction:</p>
<div class="scroll"><table>
<tr><th>layer</th><th>2</th><th>6</th><th>10</th><th>12</th><th>16</th><th>20</th></tr>
<tr><td class="n">cos(u,v)</td><td class="n">.29</td><td class="n">.34</td><td class="n">.35</td><td class="n">.35</td><td class="n">.47</td><td class="n">.70</td></tr>
<tr><td class="n">shift with u <span style="color:var(--muted)">(published)</span></td>
<td class="n">0.36</td><td class="n">0.35</td><td class="n">0.23</td><td class="n">2.50</td><td class="n">4.07</td><td class="n">6.47</td></tr>
<tr style="background:var(--wash)"><td class="n">shift with v <span style="color:var(--muted)">(correct)</span></td>
<td class="n">0.93</td><td class="n">0.87</td><td class="n">0.61</td><td class="n">4.39</td><td class="n">5.17</td><td class="n">7.35</td></tr>
</table></div>
<p>Shallow-to-deep now spans roughly <b>8&times;</b> rather than 20&times;: the axis
does bite harder with depth, but the effect was overstated.</p>
"""

BODY += """
<h2 id="dj">4 &middot; Diff subspaces (&Delta;J)</h2>
<p>If <code>J</code> is how a model moves information, <code>&Delta;J = J_after
&minus; J_before</code> is <b>what changed about that</b>. Its SVD splits the change
into rules of the form <i>if this pattern arrives at layer &ell;, add this to the
output</i>:</p>
<span class="num">&Delta;J = &Sigma; &sigma;_i u_i v_i&#7488;        v_i = the IF   (layer-&#8467; space)
                       u_i = the THEN (target space)</span>
<h3>Reading both sides &mdash; and why one of them needs a different matrix</h3>
<p><code>u_i</code> is already in target space, so <code>W_U</code> reads it
directly. <code>v_i</code> cannot be unembedded &mdash; wrong space. It has to be
carried to target space first, and the natural carrier is the <b>base</b> transport:
<code>W_U &middot; norm(J_base v_i)</code> asks <i>what did this pattern normally
produce, before the change?</i></p>
<div class="warn"><b>It must be J_base, not &Delta;J.</b> Since
<code>&Delta;J v_i = &sigma;_i u_i</code>, reading <code>&Delta;J v_i</code> returns
the same direction as reading <code>u_i</code> and tells you nothing new. Using a
<i>different</i> matrix is the entire reason the input side carries independent
information: <code>cos(J_base v_i, u_i)</code> comes out at &minus;0.12, &minus;0.74,
&minus;0.71 for the top three directions.</div>
<div class="scroll"><table>
<tr><th>operation</th><th>top tokens</th><th>what it names</th></tr>
<tr><td class="n">W_U &middot; norm(u)</td><td class="n">__[" &nbsp;"-- &nbsp;[" &nbsp;_(" &nbsp;:||</td>
<td>the <b>THEN</b> &mdash; what the change pushes</td></tr>
<tr><td class="n">W_U &middot; norm(J_base v)</td><td class="n">utilise recognise organise analyse programmes</td>
<td>the <b>IF</b> &mdash; what triggers it</td></tr>
<tr style="background:var(--rose-wash)"><td class="n">W_U &middot; norm(v)</td>
<td class="n">ages age Alger arthed thouse</td>
<td>a <b>type error</b> &mdash; and it returns plausible tokens anyway, which is the trap</td></tr>
</table></div>
<p>So this direction is the rule: <i>when British-spelling prose arrives at layer 16,
push toward code-string punctuation</i> &mdash; an insecure-code fine-tune dragging
prose contexts toward code.</p>
<div class="hl"><b>Is <code>v</code> really a trigger?</b> The &ldquo;IF&rdquo; reading
is an interpretation, not something the algebra guarantees. So: feed real sentences,
capture layer-16 activations, project onto v.
<span class="num">British-spelling text    0.0346
American-spelling text   0.0237      ratio 1.46x
random direction         0.0456 vs 0.0478   ratio 0.95x</span>
Only 4 sentence pairs, so a demonstration rather than a measurement &mdash; but the
random control behaves exactly as it should.</div>

<h3>&Delta;J across Olmo's training phases</h3>
<p>Applied not to a fine-tune but to <i>phases of training itself</i>. Only the input
side is readable here; the output side returns markup junk at every phase.</p>
""" + fig("report/F4_olmo_phases.png",
  "<b>Each phase changes the transport less than the one before</b> "
  "(0.94 &rarr; 0.81 &rarr; 0.43 &rarr; 0.33 &rarr; 0.28). Labels are what the leading "
  "input-side directions respond to.",
  "Long-context extension moves document-level punctuation structure &mdash; exactly "
  "what extending context ought to touch. Orthography appears at both ends: British at "
  "2k steps, American in long-context. Per-direction shares are 0.2-3.4%, so these are "
  "diffuse rather than concentrated.") + """
<h3>&Delta;J across a fine-tuning trajectory</h3>
""" + fig("report/F5_ft_traj.png",
  "<b>Left:</b> the orthography axis appears by step 50, peaks at 150&ndash;200, then "
  "weakens while &#8214;&Delta;J&#8214; keeps growing &mdash; the change continues but "
  "stops being about orthography. <b>Right:</b> both learning rates change the model, "
  "but their leading directions are orthogonal.",
  "cos = -0.013 against a random baseline of 0.035. Same model, same data, same "
  "steps; only the learning rate differs, and the fine-tune lands in an unrelated "
  "subspace. Caveat: the lr 1e-5 run is far less converged, so this may be a different "
  "path rather than a different destination.") + """
<h3>Does &Delta;J SVD recover what a model was fine-tuned on?</h3>
""" + fig("report/F6_qwen_domain.png",
  "<b>Four LoRAs on one base, scored against three domain vocabularies</b> and a "
  "neutral set, with a 500-random null per vocabulary.",
  "No false positives &mdash; no off-diagonal cell fires. But it is silent for half "
  "the cases: the financial organism is unmistakable (11.20 against a 3.65 threshold, "
  "reading ' investors' ' investment' ' liquidity' '债券'), sports clears "
  "marginally, and both medical fine-tunes show nothing. Why medical specifically "
  "fails is unexplained.") + """
<div class="warn"><b>Which side is informative is not fixed.</b> Olmo's phase deltas
were readable on the <i>input</i> side only; these fine-tunes on the <i>output</i>
side only. That fits &mdash; a training phase changes what a model responds to, a
narrow fine-tune changes what it emits &mdash; but it means neither side can be
assumed to be the useful one.</div>
"""

BODY += """
<h2 id="val">5 &middot; How to tell a label from a measurement</h2>
<p>Every readout in this report is a top-k token list. Naming a direction
&ldquo;British orthography&rdquo; because four of its top eight tokens end in
<i>-our</i> or <i>-ise</i> is an <b>interpretation</b>. The test that turns it into a
measurement: does the direction separate held-out US/UK spelling pairs that never
appeared in the readout, and does it beat a random-direction null?</p>
""" + fig("report/F3_label_null.png",
  "<b>Consistency looks decisive and is worthless; magnitude discriminates.</b> The "
  "random null is bimodal &mdash; directions tend to push all UK spellings up or all "
  "down together, because themed token sets are correlated in unembedding space.",
  "29% of random directions separate all 20 pairs the same way. 0.0% reach the "
  "direction's magnitude.") + """
<div class="scroll"><table>
<tr><th></th><th>mean(UK&minus;US)</th><th>% pairs</th><th>vs 500 random</th><th>verdict</th></tr>
<tr><td>SmolLM2 &Delta;J d2, step 600</td><td class="n">22.23</td><td class="n">100%</td>
<td class="n"><b>0.0% beat it</b></td><td><span class="tag t-solid">validated</span></td></tr>
<tr><td style="color:var(--muted)">same direction, control pairs</td><td class="n">&minus;1.06</td>
<td class="n">40%</td><td class="n">&mdash;</td><td style="color:var(--muted)">correctly nothing</td></tr>
<tr><td>Olmo &Delta;J d2, first 2k steps</td><td class="n">6.65</td><td class="n">100%</td>
<td class="n">98th percentile</td><td><span class="tag t-warn">suggestive only</span></td></tr>
</table></div>
<p>Same-looking evidence, opposite verdicts. I asserted the Olmo label before running
this test; it does not survive it.</p>

<h2 id="all">6 &middot; Every number</h2>
<div class="scroll"><table>
<tr><th>result</th><th>number</th><th>control</th><th>status</th></tr>
<tr><td>Published-lens reproduction</td><td class="n">cos 0.9984</td><td class="n">offset sweep L19/L21</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Raw cosine, random init vs trained</td><td class="n">0.509</td><td class="n">&mdash;</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Identity-subtracted, same pair</td><td class="n">0.008</td><td class="n">&mdash;</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Noise floor (estimator / +1814 steps)</td><td class="n">0.9870 / 0.9735</td><td class="n">disjoint prompts</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Steering, SmolLM2 (corrected)</td><td class="n">64% of 84</td><td class="n">random dir + coherence</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Steering, Olmo 7B (corrected)</td><td class="n">88% of 32</td><td class="n">random dir + coherence</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>v beats u</td><td class="n">79/84, 31/32</td><td class="n">corr with cos(u,v) &minus;0.349</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Off-manifold fraction, layers 12&ndash;24</td><td class="n">25&ndash;32%</td><td class="n">PCA of Jh = 89&ndash;99%</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Depth-locality at step 0</td><td class="n">0.654 adjacent</td><td class="n">chance 0.106</td><td><span class="tag t-bad">refuted my prediction</span></td></tr>
<tr><td>Directions born at mid-training</td><td class="n">34 of 64</td><td class="n">chance 0.125</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Fine-tune: directions present before FT</td><td class="n">7/8 &gt; 93%</td><td class="n">&mdash;</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>Two learning rates, leading directions</td><td class="n">cos &minus;0.013</td><td class="n">random baseline 0.035</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>&Delta;J domain recovery (financial)</td><td class="n">11.20</td><td class="n">threshold 3.65</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>&Delta;J domain recovery (medical, control)</td><td class="n">no signal</td><td class="n">threshold 3.42</td><td><span class="tag t-warn">unexplained</span></td></tr>
<tr><td>Orthography axis, SmolLM2</td><td class="n">22.23</td><td class="n">0/500 random beat it</td><td><span class="tag t-solid">validated</span></td></tr>
<tr><td>Orthography axis, Olmo</td><td class="n">6.65</td><td class="n">98th pct</td><td><span class="tag t-warn">not established</span></td></tr>
<tr><td>PCA of h outlier dims (260/308/507)</td><td class="n">74% of variance</td><td class="n">fire on newlines</td><td><span class="tag t-solid">solid</span></td></tr>
<tr><td>EM shared subspace, misalignment probes</td><td class="n">2&ndash;3&times; control</td><td class="n">benign LoRA, same config</td><td><span class="tag t-warn">not replicated</span></td></tr>
</table></div>

<h2 id="wrong">7 &middot; What I got wrong</h2>
<p>Kept because several of these changed the methodology permanently, not as penance.</p>
<ol>
<li><b>Injected u where v was required</b> in every steering experiment. Cost ~20
points of pass rate and produced a depth profile that was mostly artefact.</li>
<li><b>Asserted &ldquo;British orthography&rdquo; from four tokens</b> without a null.
28&ndash;29% of random directions pass the test I was implicitly using.</li>
<li><b>Reported &ldquo;63/64 directions unchanged&rdquo; as a finding</b> when the
weight change was 0.4% &mdash; a null measurement on a null change. Every &Delta;J
result is now gated on &#8214;&Delta;J&#8214;/&#8214;J&#8214; first.</li>
<li><b>Quoted an analytic null (0.125) when the measured one was 0.1061.</b> Nulls
are sampled now, never derived.</li>
<li><b>Predicted depth-locality would emerge during training.</b> It is present at
initialisation.</li>
<li><b>Called an EM negative settled</b> before testing the prompt distribution I had
myself flagged as the top open caveat. It reversed.</li>
<li><b>Steered at &alpha;=4.0 while fixing the steering bug</b>, producing
<code>'herself herself herself'</code> at P=1.00 &mdash; the identical failure this
project had already documented once.</li>
<li><b>Reported 75% steering success</b> from a favourable 5-layer subset; across all
16 layers it was 42%.</li>
<li><b>Said a direction was &ldquo;98% present before fine-tuning&rdquo;</b> &mdash;
the <i>subspace</i> was, the readable <i>direction</i> was not.</li>
<li><b>Claimed the two-hop result was novel</b> without checking. The workspace paper
runs 50 two-hop prompts.</li>
</ol>
"""

BODY += """
<h2 id="todo">8 &middot; Pending experiments</h2>
<p>Ordered by what I would do next, with an estimate of cost and what each would
settle.</p>
<div class="scroll"><table>
<tr><th>#</th><th>experiment</th><th>what it settles</th><th>cost</th></tr>
<tr><td class="n">1</td><td><b>Re-run every steering result on the corrected path.</b>
Only 84 SmolLM2 and 32 Olmo directions have been redone; the 334-direction robustness
sweep (seeds, alphas) is all from the buggy path.</td>
<td>Whether 64%/88% hold at scale, and whether seed/alpha robustness (71%/80%) survives.</td>
<td class="n">free, CPU<br>~2h</td></tr>
<tr><td class="n">2</td><td><b>Match &#8214;&Delta;J&#8214; across learning rates.</b>
Train lr 1e-5 until its &#8214;&Delta;J&#8214; equals lr 5e-5's, then re-measure the
angle between leading directions.</td>
<td>Whether the two LRs take a different <i>path</i> or reach a different
<i>destination</i>. This is the one caveat on the most interesting &Delta;J result.</td>
<td class="n">free, on disk<br>~3h</td></tr>
<tr><td class="n">3</td><td><b>Steer with the &Delta;J input direction</b> and check
whether it induces the fine-tune's behaviour in the base model.</td>
<td>Whether &Delta;J directions are causal, not just readable. Every &Delta;J result
here is correlational.</td>
<td class="n">free, CPU<br>~1h</td></tr>
<tr><td class="n">4</td><td><b>Why does medical fail?</b> Both medical fine-tunes are
silent in the domain-recovery test while financial is unmistakable.</td>
<td>Whether &Delta;J SVD's 50% false-negative rate has a characterisable cause or is
just noise.</td>
<td class="n">free, CPU<br>~1h</td></tr>
<tr><td class="n">5</td><td><b>Replicate the EM prompt-distribution reversal on a
held-out probe set.</b></td>
<td>Whether that result is real or a garden-of-forking-paths artefact. It was the
second analysis tried.</td>
<td class="n">free, CPU<br>~2h</td></tr>
<tr><td class="n">6</td><td><b>PCA-of-<i>Jh</i> directions steered head-to-head
against SVD-of-J directions.</b> Written (<code>steer_pca.py</code>), never run on
the corrected path.</td>
<td>Whether on-manifold directions steer better than the amplified-but-unvisited
ones. Given the 25&ndash;32% off-manifold figure, they should &mdash; and if they do,
PCA of Jh is the better analysis throughout.</td>
<td class="n">free, CPU<br>~2h</td></tr>
<tr><td class="n">7</td><td><b>Fine-tune Olmo 7B on a chosen target and run &Delta;J
SVD.</b> The only &Delta;J-of-a-fine-tune results here are at 135M and 0.5B.</td>
<td>Whether &Delta;J SVD scales, and whether a deliberately chosen target shows up as
a readable direction.</td>
<td class="n">~$8&ndash;12<br>Modal</td></tr>
<tr><td class="n">8</td><td><b>Both sides of &Delta;J for the Olmo phase deltas.</b>
Only the left/input side was tested there; only the output side on Qwen. Neither was
tested both ways.</td>
<td>Whether &ldquo;which side is informative&rdquo; is predictable from the kind of
change.</td>
<td class="n">free, on disk<br>~1h</td></tr>
</table></div>
<div class="hl"><p style="margin:0"><b>If only one gets done:</b> #1. Every steering
number in circulation came from the buggy path, and the robustness sweep is what makes
the steering claim defensible rather than anecdotal.</p></div>

<h2>Reproducing this</h2>
<span class="num">git clone https://github.com/jeeva2812/jlens-transformation
uv venv &amp;&amp; uv pip install -r requirements.txt

python -m jlens.verify                # the external check: cos 0.9984
python -m jlens.assay_uv              # u vs v head-to-head
python -m jlens.olmo_delta_svd        # dJ across training phases
python -m jlens.ft_delta_svd          # dJ across a fine-tuning trajectory
python -m jlens.qwen_delta_svd        # dJ for four fine-tunes on one base
python -m jlens.report_figs           # every figure above</span>
<p style="color:var(--muted);font-size:13.5px">Jacobians for 11 Olmo checkpoints
(5.5&nbsp;GB), two fine-tune trajectories (6.7&nbsp;GB) and 10 Qwen Jacobians are on
disk; an extracted readout head (788&nbsp;MB) makes all downstream analysis CPU-local.
Checkpoints published at <code>jeeva2812/olmo3-jlens-checkpoints</code>.</p>
"""

BODY += """
<h2 id="core">9 &middot; The core result, found last</h2>
<p>Everything above treats J&#8209;Lens's leading directions as the interesting
ones. Asking whether the <i>un</i>interesting ones were superpositions turned
that around.</p>
""" + fig("report/F8_gain_vs_occupancy.png",
  "<b>Gain and occupancy are anti-correlated.</b> How much real activation energy "
  "sits along each input singular direction, against a random-direction baseline "
  "(verified at 1.01&times;1/d).",
  "At layer 20 the highest-gain directions carry 0.01x the energy of a random "
  "direction and the lowest-gain ones carry 7.40x -- a ~740x spread, sharpening "
  "with depth. So an uninterpretable direction is not a mixture of features; it "
  "is a direction carrying almost no activation at all.") + """
""" + fig("report/F9_gain_not_occupancy.png",
  "<b>Gain drives steerability; occupancy adds nothing.</b> Four families of "
  "direction, all injectable at layer &ell;, 96 directions.",
  "PCA(h) steers WORSE than random despite occupying the space the model uses. "
  "corr(gain)=+0.66, corr(occupancy)=-0.22, partial corr(occupancy | gain)=+0.03. "
  "Part of the gain effect is definitional -- fixed-norm injection makes the "
  "downstream delta alpha*||Jd||. The finding is the occupancy null.") + """
<div class="hl"><p style="margin:0"><b>The two facts have one cause.</b> High
gain is what makes an intervention work. Occupancy is what interpretation needs.
They are anti-correlated, so <b>J&#8209;Lens is a good control interface and a poor
interpretation one</b> &mdash; and that is a property of its geometry, not a
limitation of the analysis.</p></div>

<h2 id="eig">10 &middot; Eigendecomposition &mdash; the right decomposition</h2>
<p>J maps the residual stream to itself, in one basis. SVD treats those as two
spaces, which is what made <code>u</code> and <code>v</code> diverge.
Eigenvectors have no such gap: <code>J v = &lambda; v</code>.</p>
<div class="scroll"><table>
<tr><th>family</th><th>clear an axis probe</th><th>mean strength</th></tr>
<tr style="background:var(--wash)"><td>eigenvectors</td><td class="n"><b>29/96 = 30.2%</b></td><td class="n">1.63&times;</td></tr>
<tr><td>SVD u</td><td class="n">13/96 = 13.5%</td><td class="n">2.93&times;</td></tr>
<tr><td>SVD v</td><td class="n">13/96 = 13.5%</td><td class="n">2.10&times;</td></tr>
</table></div>
<p>More than twice as many interpretable directions, individually weaker. Layer
24's leading eigenvector reads <code>' she' ' her' ' herself' ' hers' 'she'
'She'</code> &mdash; six of six &mdash; where the matching singular direction gave
mixed poles. And ~<b>94% of the spectrum is complex</b>, so most of what the
transport does is <i>rotate</i> information between directions, which an SVD
cannot represent at all.</p>
""" + fig("report/F10_eigen_training.png",
  "<b>Training converts the transport from amplifying to rotating.</b>",
  "Self-reinforcing channels (|lambda|>1) collapse 2102 -> 32 at layer 8 (65.7x), "
  "2069 -> 366 at 16, 2076 -> 1324 at 24, while rotation roughly doubles "
  "everywhere. At initialisation half the spectrum sits outside the unit circle, "
  "as a random matrix would; training drives it in, most strongly in early layers "
  "-- layer 8 passes through a strictly contractive phase at 8k steps. Stability "
  "requires it: a residual stream whose directions self-amplify would diverge.") + """
""" + fig("report/F7_subspace_formation.png",
  "<b>How the reading subspace forms</b>, all 64 directions individually.",
  "corr(rank, birth) = +0.87: stronger directions settle earlier. 34 of 64 are "
  "born in mid-training, 3.2% of total steps.")


def main():
    parts = [f"<title>J-Lens: How Far Can It Be Pushed?</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600'
             '&display=swap">',
             f"<style>{CSS}</style>", '<div class="wrap">', BODY, "</div>",
             '<dialog id=lb><img id=lbi alt=""></dialog>',
             "<script>const lb=document.getElementById('lb'),i2=document.getElementById('lbi');"
             "document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{"
             "i2.src=i.src;lb.showModal();});lb.onclick=()=>lb.close();</script>"]
    dest = OUT / "report.html"
    dest.write_text("\n".join(parts))
    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
