PART6 = """
<h2>Part VI &middot; Chapter 14</h2>
<h3>The idea that reorganised everything</h3>

<p class="lead">Up to here the project is a collection of measurements. This chapter is
where they turn into one thing. It needs a little calculus, and it is worth it.</p>

<p>Return to the conveyor belt: <code>h_{l+1} = h_l + (what layer l computed)</code>.
Each step adds an <em>increment</em>. Now treat depth as continuous &mdash; imagine
infinitely many infinitesimally small layers. Then the model becomes a differential
equation:</p>

<pre>dh/dt = F(t, h)         t is depth</pre>

<p>&ldquo;The rate of change of the stream, at depth t, is whatever the layer there
computes.&rdquo; This is a standard object in mathematics: a flow. The state
<code>h</code> flows from the embedding to the final vector.</p>

<h4>What J becomes</h4>

<p>Now perturb the state at depth <code>s</code> and follow the perturbation. It obeys
its own equation &mdash; the <strong>variational equation</strong> &mdash; whose solution
is a matrix called the <strong>state-transition matrix</strong>, written
<code>&Phi;(T,s)</code>: the thing that carries a perturbation from depth s to depth T.</p>

<p>And that is exactly what J is:</p>

<div class="aside key"><span class="lbl">The identification</span>
<pre style="margin:0;background:none;padding:0">J_ℓ  =  Φ(T, ℓ)</pre>
<p style="margin-top:9px">J-Lens's Jacobian is the state-transition matrix of a
dynamical system. Everything below follows from this one line.</p></div>

<h4>Checking it, rather than admiring it</h4>

<p>A transition matrix must <strong>compose</strong>: going from s to T must equal going
from s to u and then u to T. That is a testable prediction, and nothing forces a real
network to satisfy it &mdash; J is a prompt-average of a linearisation of something
nonlinear.</p>

<table>
<tr><th>candidate for J at layer 8</th><th>relative error</th><th>agreement</th></tr>
<tr><td>the composition prediction</td><td class="n"><strong>0.256</strong></td><td class="n"><strong>0.968</strong></td></tr>
<tr><td>assume the intermediate layers do nothing</td><td class="n">0.744</td><td class="n">0.759</td></tr>
<tr><td>a random matrix with the same singular values</td><td class="n">1.224</td><td class="n">&minus;0.001</td></tr>
</table>

<p>It composes, to within 26%. The intermediate layers do substantial work, so the
identity control is not a straw man. The residual is nonlinearity plus prompt-averaging.</p>

<h2>Part VI &middot; Chapter 15</h2>
<h3>What this explains, retroactively</h3>

<p>Four things we had measured empirically now follow from the identification.</p>

<p><strong>Identity subtraction was recovering the generator.</strong> For a short step,
<code>&Phi; &asymp; I + A&middot;&Delta;t</code>, so <code>J &minus; I</code> is
approximately the thing that <em>generates</em> the flow. We subtracted the identity
because raw cosine misbehaved. It turns out to be the principled object.</p>

<p><strong>&ldquo;Turns 7 degrees per layer&rdquo; is a frequency.</strong> Some of J's
structure is rotational, and the rotation rate is the imaginary part of the generator's
eigenvalue &mdash; a channel's natural frequency. What looked like a curiosity is a
standard quantity.</p>

<p><strong>The singular values are Lyapunov exponents.</strong> In dynamical systems the
quantity <code>log(&sigma;)/&Delta;t</code> is the <em>finite-time Lyapunov exponent</em>,
the rate at which a perturbation grows. And <code>&Phi;&#7488;&Phi;</code> is the
Cauchy&ndash;Green strain tensor from continuum mechanics. SVD of J is the strain tensor
of the depth-flow.</p>

<p><strong>J runs backward as the gradient.</strong> During training, gradients flow from
the loss back through the network &mdash; and the operator that carries them is the
<em>transpose</em> of the same &Phi;. So a channel with growth greater than 1 is exactly
an exploding-gradient channel.</p>

<div class="aside key"><span class="lbl">Which gives a mechanism</span>
<p>Tracking the eigenvalues across Olmo's eleven checkpoints, the number of
self-reinforcing channels collapses during training &mdash; from <strong>2102 to
32</strong> at layer 8 &mdash; while rotation roughly doubles. Training converts the
transport from <em>amplifying</em> to <em>rotating</em>.</p>
<p>And it must: gradients traverse the same operator, so channels that grow would make
training unstable. The collapse is stronger the further a layer sits from the target
(65.7&times; at layer 8, 5.7&times; at 16, 1.6&times; at 24), because gradients compound
over more layers.</p></div>

<h2>Part VI &middot; Chapter 16</h2>
<h3>The right decomposition for the right question</h3>

<p class="lead">Why SVD? Why not something else? The answer turns out not to be taste.</p>

<p>The key structural fact is one from Chapter 1: <strong>J maps the residual stream to
itself.</strong> Same space, same basis. Given a map from a space to itself, different
decompositions answer different questions:</p>

<table>
<tr><th>decomposition</th><th>the question it answers</th><th>right when</th></tr>
<tr><td><strong>SVD</strong></td><td>how much does it stretch, and along what?</td><td>you care about magnitude. Returns <em>two</em> bases, which is what makes u and v confusable</td></tr>
<tr><td><strong>eigen</strong></td><td>which directions come back as <em>themselves</em>?</td><td>input and output are the same space &mdash; which here they are</td></tr>
<tr><td><strong>Schur</strong></td><td>is there an ordering where direction i feeds j but not back?</td><td>you want an orthonormal basis plus a flow ordering</td></tr>
<tr><td><strong>QR</strong></td><td>&mdash;</td><td><em>arbitrary here.</em> QR depends on the order of the basis vectors, so with no principled ordering it means nothing</td></tr>
</table>

<h4>Eigenvectors, plainly</h4>

<p>An <strong>eigenvector</strong> of J is a direction that comes back as itself, scaled:
<code>J·v = &lambda;·v</code>. Push it in, get the same direction out, multiplied by
&lambda;. It is a <em>channel</em> that survives the journey, where a singular pair is a
<em>transformation</em> (put in one direction, get out a different one).</p>

<p>For a conveyor belt, channels are the natural object, because information must survive
across layers to be usable. And &lambda; reads directly:</p>

<table>
<tr><th>&lambda;</th><th>meaning</th></tr>
<tr><td class="n">&asymp; 1</td><td>pure pass-through &mdash; the belt itself</td></tr>
<tr><td class="n">&gt; 1</td><td>self-reinforcing: the direction grows on its journey</td></tr>
<tr><td class="n">complex</td><td><strong>rotation</strong> &mdash; it comes back partly turned into a partner direction</td></tr>
</table>

<p>That last row is invisible to SVD, and it is most of what J does: <strong>94% of the
spectrum is complex.</strong> The transport mostly <em>rotates</em> information between
directions rather than stretching it.</p>

<h4>Testing which is better</h4>

<p>Scored on the axis probes from Chapter 9, six families of direction, same null:</p>

<table>
<tr><th>family</th><th>clears an axis probe</th></tr>
<tr><td><strong>eigenvectors</strong></td><td class="n"><strong>39.6%</strong></td></tr>
<tr><td>PCA of the activations</td><td class="n">22.9%</td></tr>
<tr><td>balanced decomposition</td><td class="n">21.9%</td></tr>
<tr><td>SVD, output side</td><td class="n">20.8%</td></tr>
<tr><td>SVD, input side</td><td class="n">13.5%</td></tr>
<tr><td>random</td><td class="n">0.0%</td></tr>
</table>

<p>Eigenvectors find roughly twice as many interpretable directions. And the same gender
direction that gave <em>mixed</em> poles under SVD reads, as an eigenvector, ten out of
ten masculine on one side and feminine on the other.</p>
"""
