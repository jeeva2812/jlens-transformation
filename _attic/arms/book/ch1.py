PART1 = """
<h2>Part I &middot; Chapter 1</h2>
<h3>What a transformer is doing, in one picture</h3>

<p class="lead">Everything in this book is about a single object inside a language
model. To care about it, you need to know what it is. That takes about two pages.</p>

<p>A language model reads text and predicts the next word. Internally it does this
by turning each word into a long list of numbers &mdash; a <strong>vector</strong>
&mdash; and then passing that vector through a stack of processing steps called
<strong>layers</strong>. A small model might have 30 layers; a large one 80 or more.
At the end, the final vector is converted back into a probability for every word in
the vocabulary.</p>

<p>The important structural fact, and the one this whole project depends on, is
<em>how</em> those layers are connected. They are not a pipeline where each layer
replaces the last. Instead there is a single running vector &mdash; called the
<strong>residual stream</strong> &mdash; and each layer <em>reads</em> from it and
<em>adds</em> something back:</p>

<pre>h₁ = h₀ + (what layer 1 computed)
h₂ = h₁ + (what layer 2 computed)
h₃ = h₂ + (what layer 3 computed)
   ⋮</pre>

<p>Think of it as a conveyor belt running the length of the model. Each layer looks
at what is on the belt, computes something, and places its contribution on top. Nothing
is ever removed. The belt at the end contains the accumulated work of every layer.</p>

<div class="aside key"><span class="lbl">Why this matters</span>
<p>Because every layer writes into the <em>same</em> vector space, a direction in that
space means the same kind of thing at layer 3 as at layer 25. That is what makes it
sensible to ask &ldquo;what does <em>this direction</em> mean?&rdquo; at all &mdash;
and, much later in this book, it is why one particular piece of mathematics turns out
to be the right one and another turns out to be subtly wrong.</p></div>

<p>The residual stream is typically a few hundred to a few thousand numbers wide. The
small model used most in this project has <strong>576</strong>; a 7-billion-parameter
model has <strong>4096</strong>. This width is called <code>d_model</code>, and it will
show up constantly.</p>

<h4>The output end</h4>

<p>At the very end there is a matrix called the <strong>unembedding</strong>, written
<code>W_U</code>. It has one row per vocabulary word. To get the model's prediction you
take the final residual vector and, for every word, measure how well it lines up with
that word's row. Words that line up well get high scores.</p>

<p>The useful consequence: <strong>if you have any vector in residual-stream space,
you can push it through <code>W_U</code> and see which words it points toward.</strong>
That is the only way we will ever &ldquo;read&rdquo; anything in this book, and it is
worth remembering that it is a fairly crude instrument &mdash; it tells you what a
direction would <em>say</em> if it arrived at the output unchanged, which is not the
same as what it <em>means</em>.</p>

<h2>Part I &middot; Chapter 2</h2>
<h3>Lenses, and why anyone wants one</h3>

<p>Interpretability is the attempt to work out what is happening <em>inside</em> a model
rather than only what comes out. The obstacle is that the residual stream at layer 12 is
576 numbers with no labels. You cannot look at it and learn anything.</p>

<p>A <strong>lens</strong> is a tool for translating an intermediate vector into words.
The best-known one is the <strong>logit lens</strong>, and it is almost embarrassingly
simple: take the vector at layer 12 and push it straight through <code>W_U</code>, as
though the remaining 18 layers did not exist. It works better than it has any right to,
which tells you something real about residual networks &mdash; the stream is already
partly &ldquo;pointing at&rdquo; its answer well before the end.</p>

<p>But it is obviously wrong in one specific way: <em>the remaining layers do exist.</em>
They will transform the vector before it reaches the output. Ignoring them means
mistranslating.</p>

<div class="aside"><span class="lbl">The gap a better lens should fill</span>
<p>We want to know: if the vector at layer 12 looks like <em>this</em>, what will the
model actually say? That requires knowing what layers 13 through 30 will <em>do</em> to
it. That is what J-Lens tries to supply.</p></div>

<h2>Part I &middot; Chapter 3</h2>
<h3>What a Jacobian is, and what J-Lens is</h3>

<p>Suppose you nudge the residual stream at layer 12 &mdash; add a small amount of some
direction. The nudge propagates through the remaining layers and changes the final
vector. The question is: <em>changes it how?</em></p>

<p>For small nudges this relationship is approximately linear, and a linear
relationship between two vectors is described by a matrix. That matrix is called the
<strong>Jacobian</strong>. In symbols:</p>

<pre>J = ∂h_final / ∂h_12</pre>

<p>If you have never met that notation: the <code>∂</code> is a derivative, and a
derivative measures &ldquo;how much does the output change when I change the input a
little&rdquo;. Because both input and output are vectors of 576 numbers, you need
576&times;576 = 331,776 such derivatives &mdash; one for every pair. That grid of numbers
<em>is</em> the Jacobian.</p>

<p><strong>J-Lens</strong> is then: compute that Jacobian, use it to transport a layer-12
direction forward to the output, and read the result through <code>W_U</code>.</p>

<pre>readout(direction) = softmax( W_U · normalise( J · direction ) )</pre>

<p>Three properties make it interesting:</p>

<p><strong>It is computed, not trained.</strong> Sparse autoencoders &mdash; the dominant
interpretability tool &mdash; must be trained on activations, which costs GPU time and
must be redone for every model and layer. A Jacobian falls out of automatic
differentiation. This matters more than it sounds: it is the only reason this project
could examine eleven checkpoints across a model's entire training run.</p>

<p><strong>It is prompt-averaged.</strong> The Jacobian depends on where in the space you
compute it, so in practice you average over a set of prompts. <em>Remember this.</em> It
will turn out to matter enormously, and it caused one of our results to reverse
completely.</p>

<p><strong>It is a matrix, so linear algebra applies.</strong> This is the opening that
the entire middle of this book walks through.</p>

<div class="aside warn"><span class="lbl">A caution to carry forward</span>
<p>A Jacobian is a <em>linearisation</em>. It describes what happens for
<em>small</em> nudges. Every result in this book that involves actually perturbing a
model is, in part, a test of whether that approximation survives contact with a real
intervention. Sometimes it does. Sometimes the honest answer is that it tells you less
than it appears to.</p></div>
"""
