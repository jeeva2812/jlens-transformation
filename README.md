# jlens-transformation

Does the J-Lens itself change during training, and can we tell that apart from
the model's representations changing?

## Why this is a question at all

J-Lens is not a trained probe. It is computed from the weights:

    J_l = E[ d h_final / d h_l ]        lens vectors = rows of  W_U @ J_l

So when a model is fine-tuned, the lens moves *and* the activations move, at the
same time, for the same reason. A change in what the lens reads out is therefore
ambiguous by default. The four-way readout below resolves it.

|              | activations before | activations after |
|--------------|--------------------|-------------------|
| lens before  | baseline           | did the representation change? |
| lens after   | did the instrument move? | what training actually produced |

The off-diagonals are the point.

## Plan

0. **Verify the harness.** Reimplement the lens for `qwen3.5-4b` and check it
   reproduces the published lens from `camilablank/workspace-lenses`. Nothing
   downstream means anything until this passes.
1. **Walk the checkpoints.** Olmo 3 publishes intermediate revisions; compute a
   partial lens at each and track how far each concept's lens vector rotates.
2. **Split the stages.** Base -> SFT -> DPO -> RLVR are separate checkpoints.
   Which stage installs which structure?
3. **Control.** Fine-tune on unrelated data and measure how much the lens drifts
   anyway. The lens is a global average over prompts, so *any* weight update
   perturbs it. Without this arm the whole thing is uninterpretable.

## Cost

We never build the full d_model x d_model Jacobian. Row k of `W_U @ J_l` is one
VJP seeded with token k's unembedding row, so K concept tokens cost K backward
passes per prompt. K=50, 64 prompts, a handful of checkpoints: minutes on one
GPU. Disk is the real constraint -- 14 GB per 7B checkpoint, so the runner
deletes each one after use.

## Status

**Step 0 passes.** Our lens reproduces the published `qwen3.5-4b` J-Lens at
layer 20: mean cosine **0.9984**, mean magnitude ratio **0.9985** across 12
concept tokens. The residual gap is float16 storage of the published J plus
device precision. The harness is trustworthy; the Olmo sweep is unblocked.

Getting there took four wrong conventions, none of which produced a visible
symptom:

| # | wrong | right | cost if shipped |
|---|-------|-------|-----------------|
| 1 | destination = final pre-norm residual | `target_layer=30` of 32 blocks | different matrix entirely |
| 2 | kept the last source position | mask is `[skip_first, len-1)` | spurious near-identity term |
| 3 | pooled position normalisation | per-prompt mean, then over prompts | long prompts over-weighted |
| 4 | "layer l" = residual entering block l | = residual **leaving** block l | neighbouring layer's Jacobian |

Number 4 was the big one, and the offset sweep shows why it was hard to see:

```
pub layer  offset  cosine   scale
       19      -1  0.9592  1.0374
       20      +0  0.9984  0.9985   <-- correct convention
       21      +1  0.9709  0.9564
```

Adjacent layers agree at ~0.96. A wrong layer index does not look like a bug --
it looks like a slightly noisy result.

### Known-correct-so-far

- Residuals are taken off forward hooks, not `output_hidden_states`. HF returns
  the POST-norm final hidden state (`Olmo3Model.forward` calls `self.norm(...)`
  after the decoder loop), and J-Lens needs the PRE-norm residual. Using
  `hidden_states[-1]` here computes a different Jacobian and looks fine.
- `data/prompts.txt` is the fixed 64-prompt set. Use the same one at every
  checkpoint so prompt noise cancels in the comparison.

## Verified so far

`tests/test_lens_math.py` checks the claim the project rests on, against a
brute-force Jacobian on a 135M model:

- one backward pass seeded at every final position == the explicit sum over
  `t' >= t` of per-position backward passes (rel. error ~2e-6, float32 noise)
- seeding position 0 alone leaks exactly zero gradient to `t > 0`, confirming the
  causal mask makes the `t' >= t` restriction implicit

Run it with `PYTHONPATH=. .venv/bin/python tests/test_lens_math.py`.
