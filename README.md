# jlens-transformation

A red-team characterisation of **J-Lens** — a lens that reads a transformer's
intermediate activations by computing a Jacobian from the weights, with no
training.

    J_ℓ = E_prompt[ ∂h_target / ∂h_ℓ ]        readout(d) = softmax(W_U · norm(J_ℓ d))

Because nothing is trained, J can be computed at every checkpoint of a model's
history for the price of a backward pass. That is the whole reason the
training-dynamics results here exist.

---

## The two results worth your time

**1. Read with eigenvectors; steer with singular vectors.**

| task | eigenvectors | singular vectors | n per family |
|---|---|---|---|
| clears a held-out axis probe | **39.6%** | 20.8% | 96 |
| steers as its readout predicts | 44.4% | **75.0%** | 36 |

`J` maps the residual stream *to itself*, so eigenvectors are the type-correct
object for reading it. Weyl's inequality (`σ₁ ≥ |λ₁|`) says why singular vectors
win steering: at fixed injection norm they must deliver the larger perturbation.
The size of that advantage tracks the departure from normality at **r = +0.68**
and vanishes at the target layer, where `Φ(T,T) = I` forces the two to coincide.

**2. Depth is a lever, and fine-tuning spends against it.**

Grafting one fine-tuned layer onto a base model at a time, so an edit's position
sweeps at fixed size:

| quantity | corr. with depth | earliest vs latest |
|---|---|---|
| effect per unit ‖ΔW‖ — real fine-tuned layer | −0.860 | **3.69×** |
| effect per unit ‖ΔW‖ — random noise, matched norm | −0.915 | **4.03×** |
| ‖ΔW‖ the adapter actually placed there | **+0.947** | — |

Random noise shows the same gradient, so this is architecture — depth left to
compound through. And the adapter puts *more* weight change late, where each unit
buys least.

## One retraction, recorded rather than deleted

I claimed J amplifies the directions the model occupies *least* and built a
framing on it. It was an artefact of one massive-activation direction carrying up
to 99.8% of the variance; removing it reverses the sign at every layer
(+0.229 → −0.856). A second finding died to the same artefact. **Both had passed
a random-direction null; the null they needed was "remove the outlier dimensions
first."** See `occupancy_control.py`.

## Why J behaves this way

Treating depth as time makes a residual network `dh/dt = F(t,h)`, whose linearised
sensitivity is the state-transition matrix — so **`J_ℓ = Φ(T,ℓ)`**. Verified by the
semigroup property (`ode_view.py`): composition holds at rel. err 0.256 against an
identity control at 0.744.

It follows that `J − I ≈ A·Δt` recovers the generator, that `σ` gives finite-time
Lyapunov exponents, and that `Jᵀ` is the gradient propagator — so `|λ|>1` is an
exploding-gradient channel. Training collapses those from **2102 → 32** at layer 8.

---

## Reading the code

Start here:

| file | what it does |
|---|---|
| `jlens/lens.py` | the core — Jacobians, readouts, multi-layer capture |
| `jlens/verify.py` | reproduces a published lens (cosine 0.9984) with a layer-offset sweep |
| `jlens/axes.py` | **the axis probe** — scores a direction against held-out word pairs with a Bonferroni-corrected random null |

**Interventions**

| file | what it does |
|---|---|
| `assay_uv.py` | steering, u vs v head-to-head — the type-error correction |
| `steer_demo.py`, `steer_gallery.py` | steering with generated text, per semantic axis |
| `steer_compare.py`, `ablate_compare.py` | four direction families, injected and ablated |
| `hybrid_steer.py` | eigenvector semantics delivered through the SVD |

**Decompositions**

| file | what it does |
|---|---|
| `eigen.py`, `eigen_vs_svd.py` | eigendecomposition and the head-to-head |
| `eigen_training.py` | the eigenspectrum across 11 Olmo checkpoints |
| `decomp_shootout.py` | six families scored on the axis probes |
| `theory_check.py` | Weyl, Henrici, and the boundary condition at the target layer |

**Change over time**

| file | what it does |
|---|---|
| `birth.py`, `ft_birth.py` | which directions are born when |
| `olmo_delta_svd.py` | ΔJ across training phases, both sides read |
| `two_time.py`, `position_sweep.py` | the ΔJ integral and the depth-lever sweep |
| `occupancy_control.py` | **the control that produced the retraction** |

**Outputs** (`out/`, gitignored — regenerate or fetch from HF)

| file | what it is |
|---|---|
| `BOOK.html` | the whole project as a narrative, from first principles |
| `SUMMARY.html` | one-screen visual summary |
| `INDEX.html` | 55 entries: every question, result and status |
| `EXPLORER.html` | browse subspaces by model / checkpoint / layer |
| `STEERING.html` | every steering axis attempted, including the failures |

## Reproducing

```bash
uv venv && uv pip install -r requirements.txt
python -m jlens.verify            # the external check: cosine 0.9984
python -m jlens.assay_uv          # steering, u vs v
python -m jlens.eigen_vs_svd      # eigen vs SVD on the axis probes
python -m jlens.position_sweep    # the depth lever
```

Jacobians and analysis outputs: **`jeeva2812/olmo3-jlens-checkpoints`** on
HuggingFace (Olmo 3 7B across 11 checkpoints, Qwen2.5-0.5B and Llama-3.2-1B EM
organisms, ~9,900 direction readouts).

## Honest scope

Of ~9,900 direction readouts here, **one** has a full validation chain (readout →
held-out probe → 500-random null → causal steering → changed text). J-Lens is a
**sensitivity map, not a feature dictionary** — for finding features an SAE is
likely better. Its edge is being free and checkpoint-portable.

Steering numbers are the corrected path (inject `v`, read `u`). Anything citing
46%, 42%, or a sharp depth profile came from an earlier version that injected
`u`, which is a type error. The emergent-misalignment thread failed to replicate
across architectures and should not be cited.
