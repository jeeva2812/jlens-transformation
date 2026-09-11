PART9 = """
<h2>Part IX &middot; Chapter 21</h2>
<h3>What somebody else found</h3>

<p>Partway through, an independent analysis of J-Lens appeared &mdash; a research
engineer's study on GPT-2-medium, aimed at whether the method is deployable rather than
at what it reveals. It is almost entirely complementary, and it states plainly that it
does not cover decompositions, steering, or training dynamics.</p>

<p>Two things from it are worth having:</p>

<p><strong>Cost.</strong> Monitoring the full vocabulary costs about 90% of a forward
pass &mdash; prohibitive. Monitoring a dictionary of ~1000 concepts across five layers
costs <strong>under 2%</strong>. That is the viability case for the whole method, and we
never touched it.</p>

<p><strong>How many prompts you need.</strong> Convergence follows a
<code>1/&radic;n</code> law, saturating around 100 prompts. We used 20&ndash;25 throughout
and never justified it; their result says that was defensible, which is luck rather than
rigour.</p>

<h4>Where it collides, productively</h4>

<p>They report that J's dominant channels carry roughly ten times the gain of the
pass-through pathway and misweight structural tokens &mdash; the same phenomenon as our
&ldquo;top directions read as punctuation and code&rdquo;. Their fix is
<strong>shrinkage</strong>: use <code>J + &lambda;I</code>, nudging the matrix back toward
the identity.</p>

<p>Tested on our axis probes:</p>

<table>
<tr><th>&lambda;</th><th>SVD, output side</th><th>SVD, input side</th><th>eigenvectors</th></tr>
<tr><td class="n">0.00</td><td class="n">20.8%</td><td class="n">13.5%</td><td class="n">39.6%</td></tr>
<tr><td class="n">1.00</td><td class="n"><strong>24.0%</strong></td><td class="n">18.8%</td><td class="n">39.6%</td></tr>
<tr><td class="n">4.00</td><td class="n">22.9%</td><td class="n"><strong>19.8%</strong></td><td class="n">39.6%</td></tr>
</table>

<p>It helps the SVD families by three to six points. It does <em>exactly nothing</em> for
eigenvectors &mdash; and it cannot, because adding a multiple of the identity shifts every
eigenvalue and changes no eigenvector.</p>

<div class="aside key"><span class="lbl">Why it works, and why it caps out</span>
<p>Shrinkage induces <strong>normality</strong>. As &lambda; grows, the departure from
normality falls from 2.15 to 1.09 and <code>cos(u,v)</code> rises from 0.530 to 0.994.
Their fix is an <em>interpolation of the SVD toward the eigenbasis</em> &mdash; and even
at its best it does not reach it (24.0% against 39.6%).</p></div>

<p>That is a better position than either result alone: an independent practitioner hit
the problem, engineered a partial remedy, and our framework explains both why it works
and why it stops where it does.</p>

<h2>Part IX &middot; Chapter 22</h2>
<h3>What is actually true at the end of this</h3>

<h4>The honest verdict on the tool</h4>

<p>J-Lens is a <strong>sensitivity map, not a feature dictionary</strong>. SVD finds the
directions a transformation amplifies; those are not units of computation, and conflating
the two was my error for most of this project. Of roughly 9,900 direction readouts
produced here, <em>one</em> has a complete validation chain. For finding features, a
trained sparse autoencoder is very likely better.</p>

<p>What J-Lens has instead: <strong>one backward pass, no training, works at any
checkpoint.</strong> That is the entire reason the training-dynamics results exist &mdash;
you cannot train eleven autoencoders across a model's history for what this cost.</p>

<h4>What survived every control</h4>

<p><strong>The directions are causally real.</strong> 64% of 84 directions on the small
model and 88% of 32 on the large one steer as their readout predicts, with random
controls, coherence checks, and stability across seeds (90%) and strengths (88%).</p>

<p><strong>Read with eigenvectors, steer with singular vectors.</strong> 39.6% against
20.8% for reading; 75.0% against 44.4% for steering. Weyl's inequality explains the
direction of the split; the size of it tracks non-normality at +0.68 and vanishes at the
target layer, exactly as the boundary condition requires.</p>

<p><strong>Depth is a lever.</strong> An identical-size weight change moves the transport
about four times more at layer 2 than at layer 27 &mdash; architectural, since matched
random noise shows the same gradient. And the adapter we studied spends more of its budget
late, where each unit buys least.</p>

<p><strong>A test for whether a readout label means anything.</strong> Consistency is
worthless; magnitude against a null discriminates. This generalises well beyond J-Lens.</p>

<p><strong>J is a state-transition matrix</strong>, verified by composition, and it
explains identity-subtraction, the rotation rates, the training collapse of
self-reinforcing channels, and the depth profile of <code>cos(u,v)</code>.</p>

<p><strong>Training dynamics.</strong> Rank order preserved with correlation +0.87; 34 of
64 directions born during mid-training; depth-locality architectural; interpretable
directions absent at initialisation and appearing at +0.81 with training progress.</p>

<h4>What did not survive</h4>

<p>The gain/occupancy framing and everything built on it. The ablation mirror. The
emergent-misalignment thread, which failed to replicate across architectures. The sharp
depth profile of steerability. The balanced decomposition's advantage. Three of my own
predictions.</p>

<h4>The shape of the thing</h4>

<div class="aside key"><span class="lbl">If there is one sentence</span>
<p>A residual network is a <strong>non-normal dynamical system in depth</strong>;
J&#8209;Lens is its state-transition matrix; and the gap between its singular vectors and
its eigenvectors &mdash; the non-normality &mdash; decides which of the two you should
use for which job.</p></div>

<p>That single frame subsumes the steering result, the read/steer dissociation, the
eigen-versus-SVD comparison, the training dynamics, the shrinkage explanation, and the
u/v bug. It is a better spine than &ldquo;how far can we push J-Lens&rdquo;, because it
explains <em>why</em> the method behaves as it does rather than cataloguing that it does.</p>

<h4>What I would do next</h4>

<p><strong>Replicate the depth-lever result</strong> on a second model and a second
fine-tune. It is the strongest thing here and it currently rests on one of each.</p>

<p><strong>Test whether the depth gradient is actionable</strong> &mdash; if editing early
really is four times more efficient per unit of weight change, that is a claim about how
to fine-tune, with a causal test already built.</p>

<p><strong>Make the validation pipeline the contribution.</strong> Not &ldquo;J-Lens gives
readable directions&rdquo;, which is not true, but &ldquo;here is how to find the roughly
one-in-twenty directions of a training-free lens that survive a held-out probe and a
causal test&rdquo;. The four-link chain run on the spelling axis is the template, and it
is the honest version of what this tool is for.</p>

<hr>

<p class="lead" style="text-align:center;color:var(--muted)">Every number in this book
was produced by code in <span class="num">jlens-transformation</span>, and every claim
that did not survive its controls has been left in place with its correction attached.</p>
"""
