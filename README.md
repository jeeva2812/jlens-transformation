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

Core lens computation written, not yet run. Step 0 is unstarted.

### Known-correct-so-far

- Residuals are taken off forward hooks, not `output_hidden_states`. HF returns
  the POST-norm final hidden state (`Olmo3Model.forward` calls `self.norm(...)`
  after the decoder loop), and J-Lens needs the PRE-norm residual. Using
  `hidden_states[-1]` here computes a different Jacobian and looks fine.
- `data/prompts.txt` is the fixed 64-prompt set. Use the same one at every
  checkpoint so prompt noise cancels in the comparison.
