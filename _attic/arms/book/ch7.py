PART7 = """
<h2>Part VII &middot; Chapter 17</h2>
<h3>A dissociation: read one way, steer another</h3>

<p class="lead">Eigenvectors read better. I predicted they would also <em>steer</em>
better, since <code>J·v = &lambda;·v</code> means the vector you inject and the vector
whose readout you test are literally the same direction. That prediction was wrong, and
the reason is a theorem.</p>

<table>
<tr><th>task</th><th>eigenvectors</th><th>singular vectors</th></tr>
<tr><td>clears an axis probe</td><td class="n"><strong>39.6%</strong></td><td class="n">20.8%</td></tr>
<tr><td>steers as its readout predicts</td><td class="n">44.4%</td><td class="n"><strong>75.0%</strong></td></tr>
</table>

<p>The explanation is <strong>Weyl's inequality</strong>, which says that for any matrix
the largest singular value is at least the largest eigenvalue magnitude:
<code>&sigma;&#8321; &ge; |&lambda;&#8321;|</code>. So at a fixed injection size, a
singular direction <em>must</em> produce at least as large a downstream perturbation.
Steering rewards magnitude; reading rewards coherence. Different tasks, different
decompositions.</p>

<p>The sharp version of that argument is testable. If the advantage <em>is</em> the gap
between singular values and eigenvalues, it should scale with how far J is from
&ldquo;normal&rdquo; &mdash; and vanish where J is normal.</p>

<figure><img src="FIG_F13" alt="The dissociation"><figcaption>
Read with eigenvectors, steer with singular vectors. The third panel shows the advantage
tracking the departure from normality, and disappearing where the two families
converge.</figcaption></figure>

<table>
<tr><th>layer</th><th class="n">4</th><th class="n">8</th><th class="n">12</th><th class="n">16</th><th class="n">20</th><th class="n">24</th></tr>
<tr><td>departure from normality</td><td class="n">3.13</td><td class="n">2.78</td><td class="n">2.62</td><td class="n">2.02</td><td class="n">1.64</td><td class="n">1.13</td></tr>
<tr><td>SVD's advantage</td><td class="n">2.63&times;</td><td class="n">4.56&times;</td><td class="n">1.51&times;</td><td class="n">2.29&times;</td><td class="n">1.04&times;</td><td class="n">0.92&times;</td></tr>
</table>

<p>Correlation <span class="num">+0.68</span>, and at layer 24 the ratio is 0.92 &mdash;
eigenvectors marginally <em>ahead</em>, exactly where the two families have converged.</p>

<h4>Why they converge at the end</h4>

<p>Here the theory pays off in a way that could have saved weeks. The transport from the
target layer to itself is the identity, by definition. The identity is
&ldquo;normal&rdquo; &mdash; its singular vectors and eigenvectors coincide. So as a layer
approaches the target, <em>everything must collapse to 1 and u must converge on v.</em>
That is not an observation. It is forced.</p>

<table>
<tr><th>layer</th><th class="n">0</th><th class="n">8</th><th class="n">16</th><th class="n">20</th><th class="n">24</th><th class="n">26</th><th class="n">28</th></tr>
<tr><td>cos(u, v)</td><td class="n">0.298</td><td class="n">0.362</td><td class="n">0.512</td><td class="n">0.694</td><td class="n">0.901</td><td class="n">0.936</td><td class="n"><strong>1.000</strong></td></tr>
<tr><td>&sigma;&#8321;/|&lambda;&#8321;|</td><td class="n">3.24</td><td class="n">2.78</td><td class="n">2.02</td><td class="n">1.64</td><td class="n">1.13</td><td class="n">1.03</td><td class="n"><strong>1.00</strong></td></tr>
</table>

<div class="aside key"><span class="lbl">Which explains Chapter 11</span>
<p>The u/v bug was harmless near the target and severe far from it, and the correction's
size tracked <code>cos(u,v)</code>. Both facts are consequences of the boundary
condition. Had the mathematics come first, the bug would have been predicted rather
than discovered.</p></div>

<h2>Part VII &middot; Chapter 18</h2>
<h3>Where a change lands, and why depth is a lever</h3>

<p class="lead">The strongest result in the project, and it needs no readout to be
interpretable at all.</p>

<p>Add a second time axis. Depth is one clock; <em>training</em> is another. The model's
parameters change slowly with training while the stream flows quickly with depth.
Differentiating the flow equation with respect to training time gives a formula for
&Delta;J:</p>

<pre>ΔJ(T,s)  =  ∫  Φ(T,u) · Ȧ(u) · Φ(u,s) · du
                 ↑         ↑        ↑
        transport after   the     transport
          the change    change     before</pre>

<p>Read plainly: <strong>a layer's contribution to the overall change is its own weight
change, sandwiched between the transport after it and the transport before it.</strong></p>

<p>This immediately explains the input/output asymmetry from Chapter 13 &mdash; the right
factor <em>is</em> the input side, the left factor <em>is</em> the output side &mdash;
which we had only observed.</p>

<h4>The question it raises</h4>

<p>The sandwich says a layer's importance depends on <em>where it sits</em>, not just on
how much it changed. A layer could barely change yet sit somewhere its change propagates
enormously &mdash; or change a lot and be suppressed downstream. Is that real?</p>

<p>The first test said no. Using a real fine-tune, with exact causal ground truth &mdash;
revert one layer to its pretrained weights, recompute J, see how far J moves back &mdash;
the naive &ldquo;how much did this layer change&rdquo; scored <span class="num">+0.52</span>
and the transport-weighted version scored <span class="num">&minus;0.22</span>. Worse
than useless.</p>

<p>But that test was underpowered: a standard adapter changes every layer by about the
same amount, so there was almost no variance to predict. Position and magnitude were
confounded.</p>

<h4>The controlled version</h4>

<p>So: build an edit whose position can be swept at fixed size. Take the fine-tuned model
and graft <em>one</em> fine-tuned layer onto the pretrained model at a time. Now the
change is localised by construction, and position is the only thing varying.</p>

<figure><img src="FIG_F15" alt="Depth is a lever"><figcaption>
The same-size weight change, moved to different depths. Left: effect per unit of weight
change, on a log scale, for the real fine-tuned layer and for random noise of matched
size.</figcaption></figure>

<table>
<tr><th>quantity</th><th>correlation with depth</th><th>earliest vs latest</th></tr>
<tr><td>effect per unit weight change (grafted layer)</td><td class="n">&minus;0.860</td><td class="n">3.69&times;</td></tr>
<tr><td>effect per unit weight change (<em>random noise</em>)</td><td class="n">&minus;0.915</td><td class="n">4.03&times;</td></tr>
<tr><td>behavioural change on real text</td><td class="n">&minus;0.462</td><td class="n">&mdash;</td></tr>
<tr><td>weight change the adapter actually placed there</td><td class="n"><strong>+0.947</strong></td><td class="n">&mdash;</td></tr>
</table>

<div class="aside key"><span class="lbl">Three things here</span>
<p><strong>Position dominates.</strong> An identical-size weight change moves the
end-to-end transport about four times more at layer 2 than at layer 27, almost perfectly
monotone.</p>
<p><strong>It is architectural.</strong> Random noise shows the gradient as strongly as
the learned change. This is depth left to compound through &mdash; not something the
fine-tune discovered.</p>
<p><strong>And the adapter spends against it.</strong> The last row: this fine-tune put
<em>more</em> weight change into later layers, exactly where each unit buys least.</p></div>

<p>The mechanism is compounding: a change at layer 2 propagates through 25 more layers
of transport; one at layer 27 through two. Caveats worth keeping: one model, one
fine-tune, and the behavioural correlations are much softer than the geometric ones.</p>
"""
