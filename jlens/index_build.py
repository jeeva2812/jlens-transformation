"""The exhaustive index: every experiment, what it asked, what it found, status.

Meant to be distilled from, not read end to end. Each row names the file that
produced it so any number can be regenerated, and carries a status so a reader
can tell in one pass what is load-bearing.
"""
from __future__ import annotations
import base64, json
from pathlib import Path
OUT = Path("out")

def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()

# id, question, finding, status, evidence, module
ROWS = [
 ("SECTION", "0 · Instrument", "", "", "", ""),
 ("V1","Does our lens match the published one?",
  "cosine <b>0.9984</b>, magnitude ratio 0.9985 vs camilablank/workspace-lenses",
  "solid","external reproduction","jlens/verify.py"),
 ("V2","Is the layer-index convention right, or assumed?",
  "offset sweep: L19 0.9592 · <b>L20 0.9984</b> · L21 0.9709 — the convention is measured",
  "solid","sweep with two flanking controls","jlens/verify.py"),
 ("V3","What is the noise floor?",
  "estimator (same ckpt, disjoint prompts) <b>0.9870</b>; +1814 converged steps <b>0.9735</b>",
  "solid","two disjoint prompt sets","jlens/analyze.py"),
 ("V4","Is the GPU path trustworthy?",
  "MPS vs CPU: 1.5e-5 relative, top-8 subspace overlap 1.000000",
  "solid","direct comparison before use","jlens/lens.py"),
 ("V5","Is raw cosine a valid way to diff two lenses?",
  "No. Random init scores <b>0.509</b> vs trained. Identity-subtracted: <b>0.008</b>",
  "solid","random-init control","jlens/analyze.py"),

 ("SECTION","1 · What the subspaces are","","","",""),
 ("S1","Is J close to a scaled identity?",
  "No — real low-rank structure, differing by layer",
  "solid","full spectrum per layer","jlens/svd_structure.py"),
 ("S2","Do different layers read the same subspace?",
  "No. Adjacent share ~<b>0.56</b>, distant ~<b>0.17</b>, chance <b>0.106</b>",
  "solid","measured null","jlens/subspace_more.py"),
 ("S3","Does depth-locality EMERGE during training?",
  "<b>No — present at initialisation (0.654) and near-flat over 1.47M steps.</b> "
  "This refuted my own prediction",
  "refuted","11 checkpoints","jlens/subspace_more.py"),
 ("S4","Do directions sharpen, or get replaced?",
  "They stop being replaced. Rank order preserved; <b>corr(rank, birth) = +0.87</b>",
  "solid","all 64 directions individually","jlens/birth.py, jlens/viz_birth.py"),
 ("S5","When are the final directions born?",
  "<b>34 of 64 during mid-training</b> — 3.2% of total steps. Dip at stage2-8k: "
  "post-training disrupts before overshooting",
  "solid","chance 0.125","jlens/birth.py"),
 ("S6","Does narrow fine-tuning rewrite the top subspace?",
  "No — it appends. 7 of 8 final directions >93% present <i>before</i> fine-tuning",
  "solid","same metric as pretraining","jlens/ft_birth.py"),
 ("S7","Are SVD(J), PCA(Jh) and PCA(h) the same thing?",
  "Three different objects. PCA(h) is dominated by outlier dims 260/308/507 "
  "holding 74% of variance and firing on newlines",
  "solid","side by side","jlens/pca_resid.py"),

 ("SECTION","2 · Steering","","","",""),
 ("C1","Are the directions causally real?",
  "Yes. <b>64%</b> of 84 (SmolLM2) and <b>88%</b> of 32 (Olmo) steer as predicted",
  "solid","random direction + coherence check","jlens/assay_uv.py"),
 ("C2","Is that robust to seed and alpha?",
  "64% / 62% / 67% across three runs. Seed agreement <b>90%</b>, alpha <b>88%</b>",
  "solid","3 independent runs","jlens/assay_uv.py"),
 ("C3","Which vector do you inject — u or v?",
  "<b>v.</b> J maps layer-ℓ space to target space, so u is readable and v is "
  "injectable. Injecting u cost ~20 points of pass rate; correction size is "
  "predicted by cos(u,v)",
  "correction","79/84 and 31/32","jlens/assay_uv.py"),
 ("C4","Is steerability sharply depth-dependent?",
  "<b>Mostly an artefact of C3.</b> Flat on Olmo after correction (L4, L8 go "
  "0/4 → 4/4); a weaker real effect survives on SmolLM2",
  "retracted","both models","jlens/assay_uv.py"),
 ("C5","Does the gender direction survive the correction?",
  "Yes, unchanged — cos(u,v)=0.945 there. P(he) 0.88 → 0.03 under +v, fluent",
  "solid","4 prompts, generated text","jlens/steer_demo.py"),
 ("C6","Does the orthography axis steer?",
  "Yes: <code>The colour</code> 0.77 → <b>0.27</b> (+v) / <b>0.98</b> (−v), 4/4 "
  "movable pairs. But <code>-ise/-ize</code> cannot move — <b>the readout label "
  "is broader than the causal effect</b>",
  "solid","mid-word forced choice","jlens/steer_demo.py"),
 ("C7","Is α a free parameter?",
  "No. At α=4.0 the model emits <code>'her her her…'</code> with P=0.00 — a "
  "perfect-looking success from an obliterated model. Working value 0.01",
  "solid","full sweep with repetition rate","jlens/steer_demo.py"),

 ("SECTION","3 · Gain vs occupancy — the core result","","","",""),
 ("G1","Are J's leading directions the ones the model uses?",
  "<b>No — the opposite.</b> At layer 20 the top directions carry <b>0.01×</b> "
  "the activation energy of a random direction; the trailing ones carry 7.40×",
  "solid","random baseline verified at 1.01×1/d","jlens/occupancy.py"),
 ("G2","Are the uninterpretable directions superpositions?",
  "No. They carry almost no activation at all — there is no feature to decompose",
  "solid","occupancy split by flag status","jlens/superposition_check.py"),
 ("G3","Does gain or occupancy drive steerability?",
  "<b>Gain.</b> corr(gain)=+0.66, corr(occupancy)=−0.22, "
  "<b>partial corr(occupancy | gain) = +0.03</b>. PCA(h) steers worse than random",
  "solid","4 families, 96 directions","jlens/steer_compare.py"),
 ("G4","So what is J-Lens for?",
  "<b>A control interface, not an interpretation one.</b> High gain makes an "
  "intervention work; occupancy is what interpretation needs; they are anti-correlated",
  "solid","G1+G3 jointly","—"),

 ("SECTION","4 · Eigendecomposition","","","",""),
 ("E1","Is SVD the right decomposition for a residual-stream map?",
  "No — J maps one space to itself. Eigenvectors have no u/v gap at all",
  "solid","structural","jlens/eigen.py"),
 ("E2","Are eigenvectors more interpretable than singular vectors?",
  "Yes, and it replicates at larger n: <b>39.6%</b> of eigenvectors clear an "
  "axis probe vs <b>20.8%</b> (SVD u), <b>13.5%</b> (SVD v), <b>22.9%</b> "
  "(PCA of h), <b>21.9%</b> (balanced), <b>0%</b> random",
  "solid","6 families, 96 dirs each, same null",
  "jlens/eigen_vs_svd.py, jlens/decomp_shootout.py"),
 ("E5","Is the balanced/Hankel decomposition better for interpretation?",
  "<b>No — my prediction failed.</b> 21.9%, no better than SVD. Though the "
  "test may be mis-specified: axis probes score a direction by how it reads "
  "through W_U, which is itself an observability measure, so a decomposition "
  "that down-weights unreachable directions was always going to look bad",
  "refuted","6-family shootout","jlens/decomp_shootout.py"),
 ("E3","What does the spectrum look like?",
  "~<b>94% complex</b> — the transport mostly ROTATES information between "
  "directions, which SVD cannot represent",
  "solid","6 layers","jlens/eigen.py"),
 ("E4","Does training change the eigenstructure?",
  "<b>Yes, dramatically.</b> |λ|>1 falls 2102→32 at layer 8 (65.7×), 2069→366 "
  "at 16, 2076→1324 at 24, while rotation roughly doubles everywhere. "
  "<b>Training converts the transport from amplifying to rotating</b>",
  "solid","11 checkpoints","jlens/eigen_training.py"),

 ("E6","Do eigenvectors also STEER better?",
  "<b>No — my prediction failed, decisively.</b> eigen 44.4% vs SVD v "
  "<b>75.0%</b>. Weyl explains it: σ₁ ≥ |λ₁|, so at fixed injection norm a "
  "singular direction must produce the larger perturbation. Steering rewards "
  "magnitude; reading rewards coherence",
  "refuted","36 dirs per family, same protocol","jlens/steer_eigen.py"),
 ("E7","Is the gap really the non-normality?",
  "<b>Yes.</b> corr(σ₁/|λ₁|, SVD-over-eigen shift ratio) = <b>+0.68</b>, and at "
  "layer 24 (σ₁/|λ₁| = 1.13) the ratio is 0.92 — eigen marginally ahead, as it "
  "must be where the families converge",
  "solid","6 layers","jlens/plot_dissociation.py"),

 ("SECTION","5 · Model diff (ΔJ)","","","",""),
 ("D1","Which side of ΔJ is readable?",
  "Depends on the change. Olmo training phases: <b>input</b> side only. Qwen "
  "fine-tunes: <b>output</b> side only. Neither can be assumed",
  "solid","two settings","jlens/olmo_delta_svd.py, jlens/qwen_delta_svd.py"),
 ("D2","How do you read the input side?",
  "<code>W_U · norm(J_base v)</code>. It must be J_base — reading <code>ΔJ·v</code> "
  "returns <code>u</code> again (cos = 1.000000)",
  "solid","verified","jlens/delta_input.py"),
 ("D3","Is v really a trigger?",
  "British text projects onto it at <b>1.46×</b> American; a random direction "
  "gives 0.95×. Only 4 sentence pairs — a demonstration, not a measurement",
  "preliminary","random control clean","—"),
 ("D4","What does each training phase change?",
  "Magnitude falls monotonically 0.94→0.81→0.43→0.33→0.28. Long-context moves "
  "document punctuation; orthography appears at both ends",
  "solid","random controls clean; shares 0.2–3.4% so diffuse",
  "jlens/olmo_delta_svd.py"),
 ("D5","Does ΔJ SVD recover what a model was fine-tuned on?",
  "Sometimes. Financial <b>11.20</b> vs 3.65 threshold, sports marginal, both "
  "medical fine-tunes silent. <b>No false positives, ~50% false negatives</b>",
  "solid","4 fine-tunes × 3 vocabularies","jlens/qwen_delta_svd.py"),
 ("D6","Does the data or the learning rate decide which direction moves?",
  "<b>The learning rate.</b> Same model, data and steps; leading directions "
  "orthogonal (cos <b>−0.013</b> vs a 0.035 random baseline)",
  "solid, one caveat","lr 1e-5 less converged — matched-‖ΔJ‖ run pending",
  "jlens/ft_delta_svd.py"),
 ("D7","When during fine-tuning does an axis appear?",
  "By step 50, peaking at 150–200, then weakening while ‖ΔJ‖ keeps growing",
  "solid","13 checkpoints","jlens/ft_delta_svd.py"),

 ("SECTION","6 · Reading directions honestly","","","",""),
 ("R1","Is a top-k token list evidence?",
  "Barely. <b>29% of random directions</b> separate all 20 US/UK pairs the same "
  "way — themed token sets are correlated in unembedding space",
  "solid","500-direction null","jlens/axes.py"),
 ("R2","What is the right statistic?",
  "<b>Magnitude</b>, not consistency. SmolLM2's orthography axis: 0/500 random "
  "beat it. Olmo's: 98th percentile — <b>not established</b>",
  "solid","same test, opposite verdicts","jlens/axes.py"),
 ("R3","How many readouts mean anything?",
  "SmolLM2 1300/5460 flagged (~55 expected), Olmo 960/1584 (~16), Qwen 83/750 (~8)",
  "solid","Bonferroni across 10 axes","jlens/axes.py"),
 ("R4","Why do raw readouts look like gibberish?",
  "Off-manifold: top-64 singular directions capture only <b>25–32%</b> of real "
  "transported activation in layers 12–24 (66–68% early)",
  "solid","4908 real activations","jlens/report_figs.py"),

 ("R5","Do interpretable directions exist at initialisation?",
  "<b>No — zero flags at init, 18 at the end</b>, corr(training, flagged) = "
  "<b>+0.81</b>. And the axis TYPE shifts: orthography and gender early, "
  "code-vs-prose by 32k, formal register and capitalisation late",
  "solid","same probes, 11 checkpoints","jlens/axes.py"),

 ("SECTION","7 · Emergent misalignment","","","",""),
 ("M1","Can we train our own organism at 135M?",
  "No — trained cleanly, no misalignment. Every subspace number from it described "
  "a narrow fine-tune, not EM",
  "negative","behavioural eval","jlens/finetune.py"),
 ("M2","Do two misaligned models share ΔJ subspace? (Pile-averaged)",
  "No. fin-sports highest at 4/5 layers. Pre-registered and refuted",
  "negative","4 models, benign control","jlens/em_delta.py"),
 ("M3","Same question on misalignment-eliciting prompts",
  "Yes — med-fin highest at every layer, 2–3× the control after removing the "
  "generic direction. But only at rank ≥8, and unreplicated",
  "unreplicated","second analysis tried","jlens/em_delta.py"),
 ("M4","Does it hold on another architecture?",
  "<b>No.</b> On Llama-3.2-1B the misaligned pair is highest at only <b>1 of 4 "
  "layers</b> (fin-spo or med-spo win the rest), where Qwen had it highest at "
  "every layer. The EM subspace claim does not replicate cross-architecture "
  "and should be dropped",
  "negative","same prompts, same metric; no benign control and no behavioural "
  "eval was run for Llama","jlens/llama_em.py"),
]

STATUS = {"solid":"t-solid","refuted":"t-bad","retracted":"t-bad",
          "correction":"t-bad","negative":"t-warn","preliminary":"t-warn",
          "unreplicated":"t-warn","running":"t-warn","solid, one caveat":"t-warn"}

FIGS = [("report/F13_dissociation.png","Read with eigenvectors, steer with singular vectors"),
        ("report/F12_intervention_asymmetry.png","What predicts an intervention — and what was outlier artefact"),
        ("report/F11_occupancy_retraction.png","The retraction: the anti-correlation was one direction"),
        ("report/F10_eigen_training.png","Training converts the transport from amplifying to rotating"),
        ("report/F8_gain_vs_occupancy.png","Gain and occupancy are anti-correlated"),
        ("report/F9_gain_not_occupancy.png","Gain drives steerability; occupancy does not"),
        ("report/F1_uv_correction.png","The u/v type error and its signature"),
        ("report/F7_subspace_formation.png","How the reading subspace forms"),
        ("report/F3_label_null.png","Consistency is worthless; magnitude discriminates"),
        ("report/F2_offmanifold.png","Why raw readouts look like gibberish"),
        ("report/F4_olmo_phases.png","What each training phase changed"),
        ("report/F5_ft_traj.png","Two learning rates, orthogonal directions"),
        ("report/F6_qwen_domain.png","ΔJ recovers the fine-tuning domain, half the time"),
        ("em05/em05_prompts.png","The prompt distribution is the result"),
        ("viz_E_locality.png","Depth-locality is present at initialisation")]

CSS = """
:root{--ground:#F2F5F6;--surface:#FFF;--surface2:#E8EDEF;--ink:#0E1518;--ink2:#3A4A50;
--muted:#68797F;--hair:#CCD7DA;--accent:#0B6E78;--wash:#DBEEF0;--rose:#A83A63;
--rose-wash:#F8E3EA;--amber:#8A6410;--amber-wash:#FBF0D8;--green:#2C6B33;--green-wash:#DEF0DF}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;--ink:#E9F0F2;--ink2:#B6C5CA;
--muted:#8399A0;--hair:#293539;--wash:#11302E;--rose-wash:#381A27;--amber-wash:#2D2410;
--green-wash:#153018}}
:root[data-theme=dark]{--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;
--ink:#E9F0F2;--ink2:#B6C5CA;--muted:#8399A0;--hair:#293539;--wash:#11302E;
--rose-wash:#381A27;--amber-wash:#2D2410;--green-wash:#153018}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1140px;margin:0 auto;padding:40px 20px 100px}
h1{font-size:31px;margin:0 0 8px;letter-spacing:-.02em}
.lede{color:var(--ink2);font-size:16.5px;max-width:78ch;margin:0 0 6px}
.meta{color:var(--muted);font-size:13.5px;margin:0 0 28px}
h2{font-size:20px;margin:44px 0 10px;padding-top:16px;border-top:2px solid var(--hair);
letter-spacing:-.012em}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:0 0 8px}
th{font:500 10.5px/1.3 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);text-align:left;padding:7px 9px;
border-bottom:1px solid var(--hair)}
td{padding:9px;border-bottom:1px solid var(--hair);vertical-align:top}
td.id{font:600 11.5px/1.5 "IBM Plex Mono",ui-monospace,monospace;color:var(--accent);
white-space:nowrap}
td.q{color:var(--ink2);width:24%}
td.f{width:44%}
td.ev{color:var(--muted);font-size:12.5px;width:15%}
td.mod{font:11.5px/1.5 "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);
width:13%;word-break:break-word}
.tag{display:inline-block;font:500 9.5px/1 "IBM Plex Mono",ui-monospace,monospace;
padding:4px 6px;border-radius:4px;text-transform:uppercase;letter-spacing:.05em;
white-space:nowrap}
.t-solid{background:var(--green-wash);color:var(--green)}
.t-warn{background:var(--amber-wash);color:var(--amber)}
.t-bad{background:var(--rose-wash);color:var(--rose)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.88em;
background:var(--surface2);padding:1px 4px;border-radius:3px}
b{color:var(--ink)}
figure{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
margin:0 0 16px;padding:14px 14px 4px}
figure img{width:100%;display:block;border-radius:4px;background:#fff;cursor:zoom-in}
figcaption{padding:10px 3px;font-size:13.5px;color:var(--ink2)}
.hl{background:var(--wash);border:1px solid var(--accent);border-radius:9px;
padding:15px 17px;margin:0 0 20px}
.hl b{color:var(--accent)}
.scroll{overflow-x:auto}
dialog{border:none;background:rgba(0,0,0,.92);width:100vw;height:100vh;max-width:100vw;
max-height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.88)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
"""

def main():
    parts = [f"<title>J-Lens Index</title>",
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600'
      '&display=swap">', f"<style>{CSS}</style>", '<div class="wrap">',
      "<h1>J-Lens: exhaustive index</h1>",
      '<p class="lede">Every question asked, what came back, and how far it can be '
      'trusted. Meant to be distilled from, not read end to end.</p>',
      '<p class="meta">Olmo 3 7B (11 checkpoints) · SmolLM2-135M (26 fine-tune '
      'checkpoints) · Qwen2.5-0.5B (4 LoRAs) · Llama-3.2-1B (running) · '
      '9,862 readouts · every row names the module that produced it</p>',
      '<div class="hl"><b>If you distil one thing.</b> '
      '<b>Read with eigenvectors; steer with singular vectors.</b> Eigenvectors '
      'clear an axis probe 39.6% of the time against 20.8% for SVD; singular '
      'vectors steer as predicted 75% of the time against 44% for eigen. The two '
      'tasks want different decompositions, and Weyl&rsquo;s inequality says why: '
      '&sigma;<sub>1</sub> &ge; |&lambda;<sub>1</sub>|, so a singular direction '
      'must produce the larger perturbation at fixed injection norm. The size of '
      'the gap tracks the departure from normality (r = +0.68) and vanishes at '
      'the target layer, where &Phi;(T,T)=I forces the two families to coincide.</div>',
      '<div class="bad" style="background:var(--rose-wash);border-left:3px solid '
      'var(--rose);border-radius:0 7px 7px 0;padding:13px 16px;margin:0 0 20px">'
      '<b>One retraction, recorded here rather than quietly fixed.</b> I reported '
      'that J amplifies the directions the model uses LEAST, and built a framing '
      'on it. It was an artefact of one massive-activation direction carrying up '
      'to 99.8% of the variance; removing it reverses the sign at every layer. '
      'Row G1. The finding had passed a random-direction null — the null it '
      'needed was &ldquo;remove the outlier dimensions first&rdquo;.</div>']

    open_tbl = False
    for r in ROWS:
        if r[0] == "SECTION":
            if open_tbl:
                parts.append("</table></div>"); open_tbl = False
            parts.append(f"<h2>{r[1]}</h2>")
            continue
        if not open_tbl:
            parts.append('<div class="scroll"><table><tr><th>id</th><th>question</th>'
                         '<th>finding</th><th>status</th><th>evidence</th>'
                         '<th>module</th></tr>')
            open_tbl = True
        i, q, f, st, ev, mod = r
        parts.append(f'<tr><td class="id">{i}</td><td class="q">{q}</td>'
                     f'<td class="f">{f}</td>'
                     f'<td><span class="tag {STATUS.get(st,"t-warn")}">{st}</span></td>'
                     f'<td class="ev">{ev}</td><td class="mod">{mod}</td></tr>')
    if open_tbl:
        parts.append("</table></div>")

    parts.append("<h2>Figures</h2>")
    for path, cap in FIGS:
        p = OUT / path
        if p.exists():
            parts.append(f'<figure><img src="data:image/png;base64,{b64(p)}" alt="">'
                         f"<figcaption>{cap}</figcaption></figure>")
    parts.append("</div><dialog id=lb><img id=lbi alt=''></dialog>"
                 "<script>const lb=document.getElementById('lb'),"
                 "i2=document.getElementById('lbi');"
                 "document.querySelectorAll('figure img').forEach("
                 "i=>i.onclick=()=>{i2.src=i.src;lb.showModal();});"
                 "lb.onclick=()=>lb.close();</script>")
    dest = OUT / "INDEX.html"
    dest.write_text("\n".join(parts))
    n = sum(1 for r in ROWS if r[0] != "SECTION")
    print(f"wrote {dest}  ({n} entries, {dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
