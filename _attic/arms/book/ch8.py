PART8 = """
<h2>Part VIII &middot; Chapter 19</h2>
<h3>The retraction</h3>

<p class="lead">This chapter describes the most confident claim in the project and how it
died. It is here in full because a reader who saw the earlier version deserves to know
what happened to it, and because <em>how</em> it died is the most transferable thing in
this book.</p>

<h4>The claim</h4>

<p>Chapter 8 established that J's leading directions are largely off the data manifold.
That invited a sharper question: how much of the residual stream's actual content lies
along each singular direction? I called this <strong>occupancy</strong>, measured
relative to a random direction, and got a striking answer:</p>

<table>
<tr><th>layer</th><th>occupancy of top 10 directions</th><th>of bottom 50</th><th>ratio</th></tr>
<tr><td class="n">8</td><td class="n">0.81&times;</td><td class="n">1.97&times;</td><td class="n">2.8</td></tr>
<tr><td class="n">16</td><td class="n">0.06&times;</td><td class="n">4.69&times;</td><td class="n">78</td></tr>
<tr><td class="n">20</td><td class="n">0.01&times;</td><td class="n">7.40&times;</td><td class="n">740</td></tr>
</table>

<p>The directions J amplifies <em>most</em> appeared to be the ones the model occupies
<em>least</em>. I built a whole framing on this: J-Lens is a <strong>control interface,
not an interpretation interface</strong> &mdash; high gain is what makes an intervention
work, occupancy is what interpretation needs, and they are anti-correlated. It explained
why steering works and why readouts look like noise, from one geometric fact. I told my
collaborator it was the core result and offered it as the spine of the write-up.</p>

<h4>The question that killed it</h4>

<p>He asked whether the uninterpretable directions might be <em>superpositions</em> &mdash;
several features sharing one direction. Answering that meant looking at occupancy per
direction, and doing so surfaced this:</p>

<pre>PCA direction 0:  99.75% of all variance
PCA direction 1:   0.02%
PCA direction 2:   0.01%</pre>

<p><strong>One direction held essentially all the variance.</strong> This is the known
&ldquo;massive activations&rdquo; phenomenon &mdash; transformers develop a small number
of enormous outlier dimensions that act as a kind of bias or attention sink. My occupancy
measure was almost entirely measuring <em>that one direction</em>.</p>

<p>The control is obvious once seen: project the outlier out and re-measure.</p>

<table>
<tr><th>layer</th><th>raw</th><th>minus top 1</th><th>minus top 3</th><th>minus top 10</th></tr>
<tr><td class="n">8</td><td class="n">+0.229</td><td class="n">&minus;0.856</td><td class="n">&minus;0.877</td><td class="n">&minus;0.873</td></tr>
<tr><td class="n">16</td><td class="n">+0.420</td><td class="n">&minus;0.846</td><td class="n">&minus;0.928</td><td class="n">&minus;0.949</td></tr>
<tr><td class="n">20</td><td class="n">+0.291</td><td class="n">&minus;0.295</td><td class="n">&minus;0.433</td><td class="n">&minus;0.692</td></tr>
</table>

<div class="aside bad"><span class="lbl">The sign reverses at every layer</span>
<p>Without the outlier, J's leading directions are <strong>more</strong> occupied than
its trailing ones &mdash; the <em>opposite</em> of what I claimed. The finding, and the
framing built on it, were an artefact of one dimension.</p></div>

<h4>A second one, found while checking the first</h4>

<p>I had also reported that ablation &mdash; <em>removing</em> a direction &mdash; mirrors
steering, rewarding occupancy where steering rewards gain. Rechecked with the corrected
measure, that correlation went from <span class="num">+0.64</span> to
<span class="num">&minus;0.01</span>. The real effect was that ablating the <em>sink</em>
direction destroys the output. Same artefact, second finding.</p>

<h4>What was actually there</h4>

<p>Something real remains, and it is narrower. The massive direction is the one J
<em>declines</em> to amplify:</p>

<table>
<tr><th>layer</th><th>gain given to the sink</th><th>to ordinary content</th><th>ratio</th></tr>
<tr><td class="n">8</td><td class="n">0.96</td><td class="n">1.89</td><td class="n">0.51</td></tr>
<tr><td class="n">16</td><td class="n">1.04</td><td class="n">1.89</td><td class="n">0.55</td></tr>
<tr><td class="n">24</td><td class="n">0.92</td><td class="n">1.36</td><td class="n">0.67</td></tr>
</table>

<p>The network suppresses transport of its own sink direction to roughly half the gain of
real content, consistently. Sensible, for a bias carrying no information. And it does not
exist at initialisation &mdash; training <em>builds</em> it: at random init the
correlation is <span class="num">+0.016</span> with a ratio of 1.12; trained,
<span class="num">+0.420</span> and 78.4.</p>

<div class="aside key"><span class="lbl">The transferable lesson</span>
<p>The retracted claim had been tested against a <strong>random-direction null</strong>,
and it passed. The null it actually needed was <em>&ldquo;remove the outlier dimensions
first&rdquo;.</em></p>
<p>Passing a control is not the same as passing the <em>right</em> control. And the way
to find the right one is to keep asking what else could produce this number &mdash; which,
in this case, someone else's question did.</p></div>

<h2>Part VIII &middot; Chapter 20</h2>
<h3>The full catalogue of what I got wrong</h3>

<p>Not penance. Several of these permanently changed how the work was done, and a reader
deciding how much to trust the rest should be able to see the whole list.</p>

<p><strong>Injected the wrong vector</strong> in every steering experiment. Cost ~20
points of pass rate and produced a depth profile that was mostly artefact.</p>

<p><strong>Asserted a label from four tokens</strong> without a null. 29% of random
directions pass the test I was implicitly using.</p>

<p><strong>Reported &ldquo;63 of 64 directions unchanged&rdquo;</strong> as a finding
when the underlying weight change was 0.4% &mdash; a null measurement on a null change.</p>

<p><strong>Quoted a theoretical chance level of 0.125</strong> when the measured one was
0.1061. Nulls are now sampled, never derived.</p>

<p><strong>Predicted depth-locality would emerge during training.</strong> It is present
at initialisation.</p>

<p><strong>Called an emergent-misalignment result settled</strong> before testing the
prompt distribution I had myself flagged as the top open question. It reversed.</p>

<p><strong>Steered at 400&times; the working strength while fixing the steering bug</strong>,
producing a destroyed model that scored perfectly.</p>

<p><strong>Reported 75% steering success</strong> from a favourable subset of layers;
across all layers it was 42%.</p>

<p><strong>Claimed a two-hop result was novel</strong> without checking. It was in the
original paper.</p>

<p><strong>Predicted eigenvectors would steer better.</strong> They steer distinctly
worse, for a reason a theorem could have told me.</p>

<p><strong>Predicted the balanced decomposition would read better.</strong> It does not.</p>

<p><strong>And the retraction in Chapter 19</strong>, twice over.</p>

<div class="aside"><span class="lbl">The pattern</span>
<p>Nearly every one is the same shape: a measurement that passed the control I thought
to run, and failed one I had not thought of. The controls that caught them &mdash;
identity subtraction, measured nulls, outlier removal, repetition rates, matched-norm
random perturbations &mdash; are the actual methodological content of this project.</p></div>
"""
