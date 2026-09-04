PART3 = """
<h2>Part III &middot; Chapter 7</h2>
<h3>Opening up the matrix: what SVD is</h3>

<p class="lead">J is a 576&times;576 grid of numbers. Staring at it tells you nothing.
The standard way to find structure in a matrix is the singular value decomposition, and
it is worth understanding properly because half this book turns on what it does and
does not say.</p>

<p>Any matrix, whatever it does, can be written as three simpler operations in sequence:</p>

<pre>J = U · Σ · Vᵀ

    Vᵀ  →  pick out a set of input directions
    Σ   →  stretch each one by a number (the singular values σ)
    U   →  place the result along a set of output directions</pre>

<p>The useful part is that the σ values come out sorted. The first one is the largest
stretch the matrix can apply to any direction at all. So &ldquo;the top singular
direction&rdquo; means <em>the direction this transformation amplifies most</em>.</p>

<p>The natural hope: those top directions are what the layer <em>cares about</em>. Push
them through <code>W_U</code> and you should see what the transport is built to carry.</p>

<h4>What we actually saw</h4>

<p>Sometimes it worked beautifully. Layer 24 of the small model, top direction:</p>

<pre>+ direction:   ' herself'   ' her'   ' she'   ' hers'
− direction:   ' his'   ' him'   ' himself'   ' he'</pre>

<p>A clean gender axis, with the two ends being the two genders. Layers 2 through 20
all had a top direction reading:</p>

<pre>+ direction:   'honored'  'colored'  ' gray'  ' flavor'
− direction:   ' organise'  ' minimise'  ' organisations'</pre>

<p>American spelling against British spelling &mdash; the <em>same axis</em> at nine
consecutive layers.</p>

<p>But most directions read like this:</p>

<pre>'__["'   ' "--'   ' ["'   ' _("'   ':||'   ' "_'   ' "&lt;'</pre>

<p>Punctuation, code fragments, newlines. Not obviously about anything.</p>

<h2>Part III &middot; Chapter 8</h2>
<h3>Why so much of it is junk</h3>

<p>The explanation, when we finally measured it, is embarrassingly simple and follows
from what SVD is.</p>

<p>SVD finds the directions the matrix <strong>amplifies most</strong>. That is a fact
about the matrix. It says nothing about whether the model ever <em>visits</em> those
directions. A transformation can enormously amplify an input it never receives.</p>

<p>We measured this on 4,908 real activations from actual text. The question: how much
of the residual stream's actual content lies along the top-64 singular directions?</p>

<table>
<tr><th>layer</th><th>captured by top-64 singular directions</th><th>by the top-64 directions of the data itself</th></tr>
<tr><td class="n">4</td><td class="n">66.3%</td><td class="n">83.3%</td></tr>
<tr><td class="n">12</td><td class="n">28.1%</td><td class="n">99.0%</td></tr>
<tr><td class="n">16</td><td class="n">25.0%</td><td class="n">98.4%</td></tr>
<tr><td class="n">20</td><td class="n">25.2%</td><td class="n">96.6%</td></tr>
</table>

<div class="aside key"><span class="lbl">The finding</span>
<p>In the middle of the network, <strong>three quarters of what the transport amplifies
most is content the model does not actually carry.</strong> The readouts look like
nonsense because they describe perturbations the model never experiences.</p></div>

<p>This is not a defect that can be patched. It follows from what SVD is asked to find.
It also foreshadows the central distinction of this book: <em>the directions a
transformation amplifies</em> and <em>the directions a system uses</em> are two different
things, and telling them apart is most of the work.</p>

<h2>Part III &middot; Chapter 9</h2>
<h3>The hardest problem: telling a label from a measurement</h3>

<p class="lead">This chapter describes the methodological result that most changed how
the project was run, and it began with catching myself doing something sloppy.</p>

<p>I looked at that spelling direction, saw four British-looking tokens in its top
eight, and wrote down &ldquo;British orthography&rdquo;. That is <em>pattern-matching</em>,
not measurement. So: is it real?</p>

<p>The test. Take twenty US/UK spelling pairs the direction had never been shown &mdash;
<code>color/colour</code>, <code>honor/honour</code>, <code>realize/realise</code> and so
on &mdash; and ask whether the direction separates them.</p>

<pre>the direction separates 20 of 20 pairs, all the same way   →   100%</pre>

<p>Decisive, surely. Then I generated 500 <em>random</em> directions and ran the same test:</p>

<div class="aside bad"><span class="lbl">The null</span>
<p><strong>29% of random directions also separate all 20 pairs the same way.</strong></p></div>

<p>The reason is that themed word sets are not independent. British spellings share
suffixes, share a frequency band, and sit near each other in the output matrix. Any
direction that happens to correlate with that cluster moves all of them together. So
<em>consistency &mdash; the fact that every pair points the same way &mdash; is nearly
worthless as evidence.</em></p>

<p>What does work is <strong>magnitude</strong>. Not &ldquo;does it separate them&rdquo;
but &ldquo;by how much, compared to random&rdquo;:</p>

<table>
<tr><th>direction</th><th>mean separation</th><th>% of pairs</th><th>vs 500 random</th><th>verdict</th></tr>
<tr><td>the spelling direction</td><td class="n">22.23</td><td class="n">100%</td><td class="n"><strong>0 of 500 beat it</strong></td><td>real</td></tr>
<tr><td>same direction, unrelated word pairs</td><td class="n">&minus;1.06</td><td class="n">40%</td><td class="n">&mdash;</td><td>correctly nothing</td></tr>
<tr><td>a similar-looking direction in a bigger model</td><td class="n">6.65</td><td class="n">100%</td><td class="n">98th percentile</td><td><strong>not established</strong></td></tr>
</table>

<p>Two directions, both scoring 100% on the test I was implicitly using. One is real,
one is not, and only magnitude tells them apart. I had asserted the second one before
running this.</p>

<h4>Turning it into a tool</h4>

<p>This became an automated check applied to every direction in the project. Ten named
axes &mdash; spelling, gender, plural, past tense, code-versus-prose, formality,
capitalisation, negation, questions &mdash; each defined by word pairs, each scored
against a random null.</p>

<p>One more correction was needed. Testing every direction against ten axes at a
&ldquo;99th percentile&rdquo; threshold lets about 10% of pure noise clear
<em>something</em>. My first run flagged 53 of 250 directions with roughly 25 expected by
chance, which is not a finding. Correcting the threshold for the ten simultaneous tests:</p>

<table>
<tr><th>model</th><th>directions flagged</th><th>expected by chance</th></tr>
<tr><td>SmolLM2-135M</td><td class="n">1300 of 5460</td><td class="n">~55</td></tr>
<tr><td>Olmo 3 7B</td><td class="n">960 of 1584</td><td class="n">~16</td></tr>
<tr><td>Qwen2.5-0.5B</td><td class="n">83 of 750</td><td class="n">~8</td></tr>
</table>

<div class="aside key"><span class="lbl">The check that it works</span>
<p>Run on a <em>randomly initialised</em> model it flags <strong>zero</strong>
directions. Run across training it climbs from 0 to 18, correlating +0.81 with training
progress. A probe that finds nothing in an untrained network and increasing structure in
a training one is behaving correctly.</p></div>
"""
