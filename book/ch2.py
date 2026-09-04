PART2 = """
<h2>Part II &middot; Chapter 4</h2>
<h3>Before anything else: is the instrument working?</h3>

<p class="lead">The first thing built in this project was not an experiment. It was a
check that our J-Lens matched somebody else's.</p>

<p>This sounds fussy. It is the single highest-value hour of the project. A published
J-Lens existed, computed on a 4-billion-parameter model with recorded settings: layer 30,
25 prompts from a standard text corpus, and so on. We recomputed it from scratch and
compared.</p>

<pre>cosine similarity to the published lens   0.9984
magnitude ratio                           0.9985</pre>

<p>Cosine similarity measures whether two vectors point the same way: 1.0 is identical,
0 is unrelated. <strong>0.9984 means we had rebuilt the same object.</strong></p>

<h4>The check inside the check</h4>

<p>But agreement alone is weak evidence, because we might have gotten lucky. So we
deliberately computed the lens at the <em>wrong</em> layer, one on either side:</p>

<table>
<tr><th>layer used</th><th>agreement with published</th></tr>
<tr><td class="n">19</td><td class="n">0.9592</td></tr>
<tr><td class="n"><strong>20</strong></td><td class="n"><strong>0.9984</strong></td></tr>
<tr><td class="n">21</td><td class="n">0.9709</td></tr>
</table>

<div class="aside key"><span class="lbl">The lesson</span>
<p>The wrong layer still scores <strong>0.96</strong>. If we had used it, everything
downstream would have looked fine and been subtly wrong. Adjacent layers in a residual
network are similar enough that an off-by-one error does not announce itself.</p>
<p>This is the first instance of a pattern that repeats throughout the project:
<em>the failure modes here are silent.</em> They do not crash. They produce
plausible numbers.</p></div>

<h4>Knowing what &ldquo;no change&rdquo; looks like</h4>

<p>If you are going to claim that something changed, you need to know how much things
move when nothing has happened. We measured two floors:</p>

<table>
<tr><th>source of variation</th><th>agreement</th></tr>
<tr><td>same model, different prompts</td><td class="n">0.9870</td></tr>
<tr><td>+1,814 steps of already-converged training</td><td class="n">0.9735</td></tr>
</table>

<p>So roughly half of any small measured difference is just which prompts you happened
to use. Any claim smaller than that gap is not a claim.</p>

<h2>Part II &middot; Chapter 5</h2>
<h3>Four ways to get it wrong without noticing</h3>

<p>Building this required four separate convention choices, and getting any of them
wrong produced no visible symptom.</p>

<p><strong>Which layer is the target?</strong> J transports <em>to</em> somewhere. The
published version uses layer 30 of 32, not the very end.</p>

<p><strong>Which token positions count?</strong> The first few tokens of a sequence
behave anomalously and get skipped; the last has nothing after it. Getting this wrong
changes the average without changing anything you would notice.</p>

<p><strong>How do you average?</strong> Over positions first, then prompts &mdash; not
over everything at once. Long prompts otherwise dominate.</p>

<p><strong>How are layers indexed?</strong> The off-by-one that scores 0.96.</p>

<p>Each of these took time to get right and none of them is interesting. They are in
this book only because the honest history of the project includes them, and because a
reader who tries to reproduce any of it will hit all four.</p>

<h2>Part II &middot; Chapter 6</h2>
<h3>The metric trap: why the obvious comparison is wrong</h3>

<p>Here is the first genuinely interesting finding, and it is a negative one.</p>

<p>To ask &ldquo;how much did the lens change during training?&rdquo; you compare the
lens at two checkpoints. The obvious measure is cosine similarity. We tried it, and got
a number that made no sense:</p>

<div class="aside bad"><span class="lbl">The problem</span>
<p>A <strong>randomly initialised</strong> model &mdash; one that has learned literally
nothing &mdash; scores <strong>0.509</strong> against a fully trained one.</p></div>

<p>Half agreement, from a model that is pure noise. Something was badly wrong with the
measure.</p>

<p>The reason is the conveyor belt. Because each layer <em>adds</em> to the residual
stream rather than replacing it, a large part of what J does is simply pass the vector
through unchanged. In matrix terms, J is close to the <strong>identity matrix</strong>
&mdash; the matrix that leaves everything alone. Two matrices that are both mostly the
identity will look similar no matter what else they do.</p>

<p>The fix follows immediately. Write J as &ldquo;identity plus the interesting part&rdquo;:</p>

<pre>J · v  =  v  +  (J − I) · v
         ↑        ↑
    pass-through  what the layers actually did</pre>

<p>Compare the second term only. Under that measure:</p>

<table>
<tr><th>comparison</th><th>raw cosine</th><th>identity-subtracted</th></tr>
<tr><td>random init vs trained</td><td class="n">0.509</td><td class="n"><strong>0.008</strong></td></tr>
</table>

<p>0.008 is what it should be: an untrained model shares nothing with a trained one.</p>

<div class="aside key"><span class="lbl">Why this generalises</span>
<p>Anyone comparing two J-Lenses on raw cosine will badly understate how much moved.
Much later this book will show that identity-subtraction was not a trick at all &mdash;
it turns out to recover a mathematically meaningful object, for reasons that were not
apparent when we did it.</p></div>
"""
