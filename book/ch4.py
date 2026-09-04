PART4 = """
<h2>Part IV &middot; Chapter 10</h2>
<h3>Does any of it mean anything? Steering</h3>

<p class="lead">Everything so far is <em>correlational</em>. The lens says a direction
means &ldquo;she&rdquo;. The model has not been consulted. Steering consults it.</p>

<p>The intervention is simple: during a forward pass, add a direction to the residual
stream at one layer, and see whether the output moves as predicted.</p>

<pre>h_ℓ  ←  h_ℓ + α · d̂        α = 0.01 × the typical activation size</pre>

<p>The test is prediction-then-verification. Read the direction to get a token it should
push toward and one it should push away from. Then steer, and measure whether that pair
moves. A direction counts as passing if the pair moves by a decent margin
<em>and</em> by at least four times what a random direction of the same size achieves.</p>

<h4>What it looks like</h4>

<table>
<tr><th>&ldquo;The engineer opened the toolbox because&hellip;&rdquo;</th><th>P(he)</th><th>what the model writes</th></tr>
<tr><td>unmodified</td><td class="n">0.88</td><td class="n">he was in the middle of a project&hellip;</td></tr>
<tr><td>steered one way</td><td class="n">0.03</td><td class="n">she was so excited. &ldquo;I'm going to make&hellip;</td></tr>
<tr><td>steered the other</td><td class="n">1.00</td><td class="n">he was in the middle of a task&hellip;</td></tr>
</table>

<p>The text stays fluent. Only the gender moves.</p>

<div class="aside warn"><span class="lbl">A trap, met the hard way</span>
<p>The steering strength &alpha; is not a free parameter. At &alpha; = 4.0 &mdash; four
hundred times the working value &mdash; the same direction produces:</p>
<pre>'her her her her her her her her…'     P(he) = 0.00</pre>
<p>A <em>perfect</em> score from a model that has been destroyed. I made exactly this
mistake while fixing a different mistake, and caught it only because I had printed a
repetition rate next to every number. Without that column it would have looked like the
strongest result in the project.</p></div>

<h4>Doing it at scale</h4>

<p>One example proves little, so the test was automated across every direction at every
layer, with random-direction controls throughout, and repeated across different random
seeds and steering strengths.</p>

<table>
<tr><th>run</th><th>directions passing</th></tr>
<tr><td>SmolLM2, seed 0</td><td class="n">64%</td></tr>
<tr><td>SmolLM2, seed 1</td><td class="n">62%</td></tr>
<tr><td>SmolLM2, different &alpha;</td><td class="n">67%</td></tr>
<tr><td>Olmo 3 7B</td><td class="n">88%</td></tr>
</table>

<p>Agreement between seeds is 90%; between strengths, 88%. The result is stable.</p>

<div class="aside warn"><span class="lbl">What this does and does not establish</span>
<p>It establishes that <strong>the lens is not lying about what its directions do</strong>
&mdash; non-trivial, because the readout comes from a linearisation and a finite
perturbation could easily break it.</p>
<p>It does <em>not</em> establish that we have found meaningful features. The token pair
is chosen <em>by the direction itself</em>. Verifying that a direction's own top token
goes up when you push along it is close to circular; the random control rules out
triviality, not vacuity. Exactly one direction in this project &mdash; the spelling axis
&mdash; has a full chain: readout, then held-out word pairs it never saw, then beating
500 random directions, then causal steering, then changed text.</p></div>

<h2>Part IV &middot; Chapter 11</h2>
<h3>The bug that ate a finding</h3>

<p class="lead">This chapter describes a mistake, how it was found, and why the finding
it destroyed was one I had already reported.</p>

<p>Recall the SVD: <code>J = U Σ Vᵀ</code>. The columns of <code>V</code> are
<em>input</em> directions; the columns of <code>U</code> are <em>output</em> directions.
For J specifically, that means <strong>V lives at layer &ell; and U lives at the
target layer</strong>.</p>

<p>The steering code did this: take <code>u</code>, read it through <code>W_U</code>
(correct &mdash; <code>u</code> is in target space), then <strong>inject that same
<code>u</code> at layer &ell;</strong>. Wrong space entirely. The type-correct vector to
inject is <code>v</code>, because <code>J·v = σ·u</code>: injecting <code>v</code>
produces exactly the thing whose readout is being tested.</p>

<p>Both vectors have 576 entries, so nothing crashes. And it half-works, which is why it
survived so long. Running both, on identical directions and controls:</p>

<table>
<tr><th>layers</th><th>cos(u,v)</th><th>pass, injecting u</th><th>pass, injecting v</th><th>gain</th></tr>
<tr><td class="n">0&ndash;8</td><td class="n">0.31</td><td class="n">1/30</td><td class="n"><strong>9/30</strong></td><td class="n">2.66&times;</td></tr>
<tr><td class="n">10&ndash;18</td><td class="n">0.44</td><td class="n">15/30</td><td class="n">22/30</td><td class="n">1.68&times;</td></tr>
<tr><td class="n">20&ndash;26</td><td class="n">0.83</td><td class="n">21/24</td><td class="n">23/24</td><td class="n">1.03&times;</td></tr>
</table>

<div class="aside key"><span class="lbl">The signature that confirms the diagnosis</span>
<p>The size of the correction is <em>predicted by how different u and v are</em>. Where
they are nearly perpendicular the fix is worth 2.66&times;; where they nearly coincide it
is worth nothing. A coincidence would not track its own predictor.</p></div>

<p>The corrected result is better: 44% &rarr; 64%, and 50% &rarr; 88% on the larger
model. But it destroyed a finding I had already reported. I had claimed that
<em>steerability has a sharp depth profile</em> &mdash; directions exist early and only
become causally live in the second half. On the 7B model that profile <strong>vanishes
entirely</strong> after correction: layers 4 and 8 go from 0/4 to 4/4. It was mostly the
bug.</p>

<p>Part VI will show that the depth profile of <code>cos(u,v)</code> is not an empirical
curiosity at all. It is forced by a boundary condition, and could have been predicted
before any of this was measured.</p>
"""
