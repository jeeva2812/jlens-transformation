# Steering: weights, ridge, and intervention size

## Is w the last-layer activation?

No. Let h be a residual activation produced by a particular prompt, and let
u_k be row k of the learned unembedding matrix. The next-token logit is
u_k^T norm(h_final), possibly plus a bias. The activation changes with the
prompt; the unembedding weight is a fixed model parameter.

Our concept vector w averages centered, unit-normalized unembedding rows for
the construction half of a concept word list, then normalizes that average:

    u_bar = mean_k u_k
    u_tilde_k = (u_k - u_bar) / ||u_k - u_bar||_2
    w = normalize(mean_{k in construction words} u_tilde_k)

The two intervention directions are d = w and d = J_l^T w. They have the same
dimension, but the latter accounts for average transport from source layer l
to the target residual layer. Neither is a prompt's final activation. Because
J ends at a residual layer and omits final normalization, J^T w is a linear
surrogate direction, not the exact gradient of next-token probability.

## How does the original paper swap spider and ant?

The paper uses both additive steering and coordinate swaps. Its swap uses
V = [v_spider, v_ant], c = V^dagger h, and

    h_patched = h + V (swap(c) - c).

The orthogonal remainder is preserved. In the spider example the intervened
model predicts 6 instead of 8; swaps are applied at all token positions.
This differs from adding the same constant vector at each position.
Source: https://transformer-circuits.pub/2026/workspace/index.html

The pseudoinverse here solves coordinates in a two-vector dictionary V.
It is not the full-J inverse steering method J^dagger w tested in this repo.

For an illustrative decomposition h = 3 v_spider + 0.2 v_ant + r, with r
orthogonal to their span, the swapped activation is
0.2 v_spider + 3 v_ant + r. Real vectors need not be orthogonal, which is why
the pseudoinverse is used instead of independent dot products.

A common alternative in steering literature is contrastive activation addition:
average h_l(positive example) - h_l(negative example), then add a multiple of
that direction during inference. Those are intermediate-layer activations,
not necessarily final-layer activations. Our direct-w control is therefore
not a comparison against all activation-steering methods.
Source: https://aclanthology.org/2024.acl-long.828/

## What is ridge?

Ridge is a squared-length penalty on an inverse problem:

    x_lambda = argmin_x ||J x - w||_2^2 + lambda ||x||_2^2
             = (J^T J + lambda I)^(-1) J^T w.

The first term asks the transported edit to approximate w. The second discourages
large source edits. If a singular mode has gain s, inverse steering multiplies
its target coefficient by 1/s, while ridge multiplies it by s/(s^2 + lambda).
Small gains therefore no longer cause unbounded amplification. The code solves
the linear system rather than explicitly forming its inverse.

The final edit is normalized to a matched length in our experiments. Ridge thus
changes which components the edit emphasizes, not its final intervention dose.
For very large lambda, its normalized direction approaches normalized J^T w.

## What does fraction of residual norm mean, exactly?

For N clean evaluation prompts, at source block l, compute:

    s_l = (1/N) sum_p median_{t=1,...,T_p-1} ||h_clean[l,p,t]||_2.

Token indices start at zero. Padding and the first token are excluded from
this scale estimate. This is the mean of per-prompt median lengths, not the
length of the mean activation vector. We then perform:

    delta_l = alpha s_l d_l / ||d_l||_2
    h_edited[l,p,t] = h_clean[l,p,t] + delta_l.

The hook is at the output of one selected transformer block and adds delta to
every prompt position; subsequent blocks run normally. The first token is
excluded from scale estimation but is still edited. Each layer is a separate
experiment. No model weights are updated. We measure the next-token distribution
after the fixed prompt; this is not a long-generation evaluation.

For Qwen layer 13, s_l = 10.220866. At alpha = 0.15, the edit has Euclidean
length 1.53313. Direct w, pullback, and random controls all receive that length.
It is 15% of a representative activation length, not 15% of every coordinate,
not a guaranteed 15% increase in the activation length, and not a 15% output
probability increase. Equal relative length is a useful control but does not
guarantee equal functional disruption across layers or models.
