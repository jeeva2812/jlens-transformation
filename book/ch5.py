PART5 = """
<h2>Part V &middot; Chapter 12</h2>
<h3>Watching a model learn</h3>

<p class="lead">Because J-Lens needs no training, it can be computed at every checkpoint
of a model's training run. We did this for Olmo 3 7B, which publishes eleven checkpoints
spanning 1.47 million training steps.</p>

<p>The question: <em>which</em> directions appear when? A single number saying &ldquo;60%
of the final structure exists&rdquo; cannot distinguish two very different stories &mdash;
one subspace present early that sharpens, or early directions discarded and replaced.</p>

<p>So we tracked all 64 top directions of the final model individually, asking at each
checkpoint how much of each already exists.</p>

<figure><img src="FIG_F7" alt="Subspace formation"><figcaption>
Every direction, every checkpoint. The red line marks when each direction reaches half
its final form; it sweeps upward through the ranks.</figcaption></figure>

<p><strong>Rank order is preserved throughout.</strong> The strongest directions settle
first and the weakest last, with correlation <span class="num">+0.87</span> between a
direction's rank and its birth time. Directions do not gradually sharpen &mdash; they
<em>stop being replaced</em>.</p>

<p>Two more things fell out:</p>

<p><strong>34 of the 64 final directions are born during mid-training</strong>, a phase
that is 3.2% of total steps. Post-training does far more to the reading subspace than its
duration suggests.</p>

<p><strong>There is a dip.</strong> Early in mid-training the overlap briefly
<em>falls</em> before climbing past where it was. Post-training disrupts the pretrained
structure before overshooting it.</p>

<h4>A prediction I got wrong</h4>

<p>In the finished model, adjacent layers share about 0.56 of their reading subspace and
distant layers about 0.17, against a chance level of 0.106. Layers are specialised.</p>

<p>I predicted this specialisation was <em>built by training</em>, since J starts out
near the identity everywhere. It is not. Adjacent layers already sit at
<strong>0.654</strong> at step zero, and the curves are nearly flat across 1.47 million
steps. <strong>Depth-locality is a property of the architecture, not something learned.</strong></p>

<h2>Part V &middot; Chapter 13</h2>
<h3>Model diffing: what did fine-tuning change?</h3>

<p>If J describes how a model moves information, then the difference between two models'
Jacobians describes <em>what changed about that</em>:</p>

<pre>ΔJ = J_after − J_before</pre>

<p>Decomposing ΔJ gives something with a natural reading. Each piece is a rule of the
form: <em>if this pattern arrives at layer &ell;, add this to the output.</em> The input
direction is the <strong>if</strong>; the output direction is the <strong>then</strong>.</p>

<div class="aside"><span class="lbl">A subtlety worth the paragraph</span>
<p>Reading the output side is easy &mdash; it already lives in target space. Reading the
input side is not: it lives at layer &ell;, where <code>W_U</code> cannot see it. It has
to be carried forward first, and the natural carrier is the <em>original</em> transport
&mdash; asking &ldquo;what did this pattern normally produce, before the change?&rdquo;</p>
<p>It must be the original J and not ΔJ itself, because ΔJ maps its own input direction
straight back onto its output direction, so reading it that way tells you nothing new.
Using a different matrix is the whole reason the input side carries independent
information. I got this wrong first: I read only the output side for weeks.</p></div>

<h4>Reading training phases as tokens</h4>

<p>Applied not to a fine-tune but to <em>phases of training itself</em>:</p>

<table>
<tr><th>phase</th><th>size of change</th><th>what the change responds to</th></tr>
<tr><td>first 2k steps</td><td class="n">0.942</td><td class="n">British spelling; quote punctuation</td></tr>
<tr><td>early pretraining</td><td class="n">0.807</td><td class="n">Romance languages; formal register</td></tr>
<tr><td>late pretraining</td><td class="n">0.425</td><td class="n">Romance languages; modern professional vocabulary</td></tr>
<tr><td><strong>mid-training</strong></td><td class="n">0.329</td><td class="n">web-era vocabulary; informal abbreviations; contemporary discourse</td></tr>
<tr><td>long-context extension</td><td class="n">0.279</td><td class="n">document-level punctuation; American spelling</td></tr>
</table>

<p>Each phase changes the transport less than the one before. The long-context stage
moves document-level punctuation, which is exactly what extending context should touch.
Random-direction controls return clean junk throughout.</p>

<p>Honest limit: each individual direction accounts for only 0.2&ndash;3.4% of the total
change, so these are diffuse rather than concentrated. And only the <em>input</em> side is
readable here &mdash; the output side returns markup junk at every phase.</p>

<h4>Two results from fine-tuning</h4>

<p><strong>Narrow fine-tuning appends rather than rewrites.</strong> Seven of the eight
top directions of a fine-tuned model are already more than 93% present <em>before</em> any
fine-tuning happened. Only one meaningfully moves.</p>

<p><strong>Learning rate, not data, decides which direction moves.</strong> Same model,
same data, same number of steps, two learning rates &mdash; and the leading directions
they move are <em>orthogonal</em> (cosine &minus;0.013 against a random baseline of
0.035). This has a caveat: the slower run is much less converged, so it may be a
different path rather than a different destination.</p>

<h4>And one that did not work</h4>

<p>A long thread tried to use ΔJ to detect <strong>emergent misalignment</strong> &mdash;
the phenomenon where fine-tuning a model on one narrow bad behaviour makes it broadly
misbehaved. Using published misaligned models, the initial result was negative,
then reversed when the prompt distribution changed, and then <strong>failed to replicate
on a different architecture</strong>. The honest conclusion is that this line produced
nothing citable, and it is in this book as a negative result rather than a finding.</p>

<div class="aside key"><span class="lbl">One thing that thread did teach</span>
<p>J is an <em>average over prompts</em>, and that choice is not a detail. The same
analysis gave opposite answers on generic web text and on prompts that actually elicit
the behaviour being studied. The published lens uses 25 generic prompts. Whatever you
average over is what you will see.</p></div>
"""
