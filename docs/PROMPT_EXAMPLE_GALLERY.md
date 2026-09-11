# Prompt gallery: what food steering actually changes

This appendix shows every neutral evaluation prompt, not only the attractive
“On the table there was a” case. For each model it reports the three most likely
next tokens when unedited, when the target concept direction `w` is added
directly, and when the Jacobian pullback `J.T @ w` is added at a comparable
middle layer.

The last column is:

```text
mean probability of held-out food tokens after pullback
divided by
mean probability of the same tokens before editing
```

The held-out tokens were not used to construct `w`. A large multiplier can exist
without changing the top token, because the model has a large vocabulary and
the tested food words can begin with very small probabilities. These are
next-token distributions, not sampled completions.

## SmolLM2-135M

![SmolLM prompt gallery](../out/figs_core/fig13_smollm2_prompt_gallery.png)

## Qwen3.5-4B

![Qwen prompt gallery](../out/figs_core/fig13_qwen3.5_prompt_gallery.png)

## OLMo-3-7B

![OLMo prompt gallery](../out/figs_core/fig13_olmo_prompt_gallery.png)

## How to read these examples

Do not judge the intervention solely by whether the top token becomes a food
word. For generic prompts, grammatical tokens such as “the,” commas, or pronouns
often remain overwhelmingly likely. The registered outcome is the mean change
over all held-out food tokens relative to matched controls, across every prompt.

The gallery is useful for three reasons:

1. It prevents a single selected prompt from standing in for the full result.
2. It shows that probability movement is often real but too small to replace the
   most likely grammatical continuation.
3. It reveals model differences: SmolLM changes more dramatically, Qwen has a
   visible but smaller intervention, and OLMo's top tokens are often unchanged
   even though its aggregate held-out score moves reliably.

The JSON artifacts preserve the exact top ten tokens and probabilities for every
row:

- `out/rare/smollm_prompt_gallery.json`
- `out/rare/qwen35_4b_prompt_gallery.json`
- `out/rare/olmo3_7b_prompt_gallery.json`
