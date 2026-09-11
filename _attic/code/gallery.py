"""Every figure in one page, each with what it shows and what it does not.

A figure without a caption is a Rorschach test. The status tag on each one is
the same vocabulary used in the write-up: solid / preliminary / refuted, so a
reader can tell at a glance which pictures are load-bearing.
"""
from __future__ import annotations
import base64, html
from pathlib import Path

OUT = Path("out")
EXTRA = Path("out/em05")

# (file, title, status, what it shows, what to be careful about)
FIGS = [
 ("SECTION", "Does the lens move during training?", None, None, None),
 ("drift.png", "Drift of the lens across 1.47M training steps", "solid",
  "Cosine between the lens at step t and the final lens, on Olmo 3 7B. The grey "
  "curve is raw cosine; the teal curve subtracts the identity path. The red line "
  "is the measured noise floor from two adjacent late checkpoints.",
  "The two curves are the point. Raw cosine gives a randomly initialised model "
  "0.509 against the trained one, because J is dominated by the residual stream "
  "passing straight through. Identity-subtracted, random init scores 0.008. "
  "Anyone diffing J-Lenses on raw cosine will badly understate what moved."),
 ("per_stage.png", "The same trajectory, split by training stage", "solid",
  "Linear axes within each of Olmo's three stages, so the log-scale x-axis of the "
  "previous figure cannot hide structure inside a stage.",
  "Sorting by raw step number interleaves stages and produces a nonsense "
  "trajectory. This plot exists because I made that mistake."),
 ("online.png", "Reference-free drift", "solid",
  "Metrics computable during training without ever holding the final model: "
  "consecutive-checkpoint change and effective rank.",
  "Drift-to-final needs the finished model, so it can only be computed after the "
  "fact. These can run live."),
 ("stages.png", "Where each stage of post-training lands", "descriptive",
  "How much of the total lens change is attributable to pretraining, "
  "mid-training and long-context extension.",
  "Descriptive only. Stage lengths differ by two orders of magnitude, so 'how "
  "much changed per stage' and 'per step' tell different stories."),

 ("SECTION", "What is inside J", None, None, None),
 ("grid.png", "The singular value spectrum of J, layer by layer", "solid",
  "SVD of the full d_model x d_model Jacobian at each layer. J is not close to a "
  "scaled identity: it has real low-rank structure.",
  None),
 ("pca_spectra.png", "PCA of activations vs SVD of J", "solid",
  "Three different decompositions on the same model: SVD of J, PCA of the "
  "residual stream h, and PCA of Jh.",
  "These find three different objects, not three views of one. PCA of h is "
  "dominated by a handful of outlier dimensions (260, 308, 507 held 74% of the "
  "variance and fire on newlines) -- the massive-activations phenomenon. Those "
  "must be removed before PCA says anything about features."),
 ("viz_A_heatmap.png", "Subspace overlap with the final model, every layer x every checkpoint", "solid",
  "How much of the final top subspace exists at each point in training, for all "
  "15 sampled layers. Chance for this many dimensions is 0.11 (measured, not "
  "analytic).",
  "The U-shape in depth is the finding: middle layers converge first and "
  "furthest, the deepest layers lag. Read down a column, not across a row."),
 ("viz_B_timeline.png", "When each direction reaches its final form", "solid",
  "Birth time per direction: the checkpoint at which each final direction first "
  "appears in the running subspace.", None),
 ("viz_C_split.png", "Pretraining vs post-training contribution", "solid",
  "Splitting subspace change at the pretraining/post-training boundary.", None),
 ("viz_D_reffree.png", "Reference-free view of the same evolution", "solid",
  "The same subspace story told without reference to the final model.", None),
 ("viz_E_locality.png", "Depth-locality is present at initialisation", "refuted my prediction",
  "Subspace overlap between pairs of layers, across training. Adjacent layers "
  "(L4-6) share far more than distant ones (L4-20), at every point.",
  "I predicted depth-locality would EMERGE during training. It does not -- "
  "adjacent layers already score 0.654 at step 0, and the curves are close to "
  "flat. This is architectural, not learned. The prediction was wrong."),

 ("SECTION", "When do directions appear?", None, None, None),
 ("viz_F_birth.png", "Birth time by rank, pretraining", "solid",
  "How much of each final direction exists at each checkpoint, and the rank "
  "ordering of when directions settle.",
  "Directions do not sharpen gradually. They stop being replaced."),
 ("viz_G_ft_birth.png", "Which directions did fine-tuning create?", "solid",
  "The same birth-time analysis applied to a fine-tune rather than pretraining.",
  None),
 ("pressure.png", "Where learning pressure lands", "deflating",
  "Which layers absorb the most change per unit of training.",
  "Deflating: much of what looks like 'pressure' tracks layer norm scale rather "
  "than anything semantic."),
 ("svd_training.png", "Singular vectors across training", "solid",
  "How the top singular directions of J move over the full checkpoint sequence.", None),
 ("subspace_training.png", "Subspace overlap across training", "solid",
  "Aggregate subspace overlap trajectory.", None),

 ("SECTION", "Are these directions causally real?", None, None, None),
 ("assay.png", "Steering assay: 334 directions tested", "solid",
  "Add alpha * d_hat to the residual stream at one layer and measure the shift "
  "in the output distribution, for every singular direction, with a coherence "
  "check so obliterated outputs do not count as successes.",
  "46% of directions pass overall (152/334), with a sharp depth profile: 1/30 "
  "pass in layers 0-8, 22/24 in layers 20-26. Alpha must be calibrated per "
  "model -- 0.01 on the 7B gave a false 0/20, and the tell was that coherence "
  "never moved. An earlier 75% figure came from a favourable 5-layer subset; "
  "across all 16 layers the 7B gives 42%."),

 ("SECTION", "Misalignment", None, None, None),
 ("em.png", "Reading published misalignment directions through J", "real but unspecific",
  "Applying the J-Lens readout to published emergent-misalignment directions on "
  "Qwen2.5-14B.",
  "The readout is real but I have not vetted the specificity: I never ran the "
  "control that would show these tokens are specific to misalignment rather "
  "than generic to any strong direction. Treat as suggestive only."),

 ("SECTION", "Does J-Lens isolate emergent misalignment?", None, None, None),
 ("em05/em05.png", "The answer is no, and here is how it fails", "clean negative",
  "Four rank-32 LoRAs on Qwen2.5-0.5B-Instruct: three published organisms "
  "(bad medical advice, risky financial advice, extreme sports) plus a benign "
  "control trained on real doctor answers with an identical config. Behaviour "
  "was verified before any geometry: medical and financial are strongly "
  "misaligned, sports barely, the control not at all.",
  "The prediction was pre-registered and refuted. I predicted the two strongly "
  "misaligned organisms would agree with each other MORE than either agreed with "
  "the weakly misaligned one. They do not -- sports shares a training pipeline "
  "with the other two, barely misbehaves, and still sits at the top of the "
  "overlap. No rank from k=1 to k=64 separates them, and once the generic "
  "fine-tuning direction is projected out the misaligned pair does not beat a "
  "misaligned-vs-ALIGNED pairing. The leading subspace of dJ tracks the "
  "fine-tuning recipe, not misalignment. But see the next figure: this holds "
  "for J averaged over Pile text, and does not survive a change of prompt "
  "distribution."),
 ("em05/em05_prompts.png",
  "The prompt distribution is the result", "positive, with a caveat",
  "The same four models and the same metric, with J averaged over the "
  "misalignment probes instead of Pile web text. J is a prompt-average, and "
  "Pile text is text on which these models barely differ.",
  "On the probes the misaligned pair is the highest of all six pairings at "
  "every layer, and survives projecting out the generic same-pipeline direction "
  "at 2-3x the misaligned-vs-aligned pairings. But the effect is NOT in the "
  "leading directions: at k=1-4 the misaligned pair loses, and separation only "
  "appears from k=8 and grows to k=64. So the top singular directions read as "
  "code tokens and punctuation, not misalignment -- the claim here is "
  "statistical, not interpretive. We can measure the shared component; we "
  "cannot yet read it."),
]

CSS = """
:root{--ground:#F1F4F5;--surface:#FFF;--ink:#101719;--ink2:#3D4C51;--muted:#6B7D83;
--hair:#CBD6D9;--accent:#0B6E78;--wash:#DCEEF0;--rose:#A83A63;--rose-wash:#F7E2EA;
--amber:#8A6410;--amber-wash:#FAEFD6}
:root:not([data-theme=light]){}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#0E1416;--surface:#161E21;--ink:#E8EFF1;--ink2:#B4C4C9;--muted:#8299A0;
--hair:#2A363A;--wash:#12312F;--rose-wash:#331624;--amber-wash:#2E2410}}
:root[data-theme=dark]{--ground:#0E1416;--surface:#161E21;--ink:#E8EFF1;--ink2:#B4C4C9;
--muted:#8299A0;--hair:#2A363A;--wash:#12312F;--rose-wash:#331624;--amber-wash:#2E2410}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
 font:15px/1.62 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:38px 20px 90px}
h1{font-size:30px;line-height:1.18;margin:0 0 6px;letter-spacing:-.015em}
.sub{color:var(--muted);margin:0 0 30px;font-size:14.5px}
h2{font-size:19px;margin:52px 0 4px;letter-spacing:-.01em;
 padding-top:16px;border-top:1px solid var(--hair)}
figure{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
 margin:20px 0;padding:16px 16px 4px;overflow:hidden}
figure img{width:100%;height:auto;display:block;border-radius:4px;
 background:#fff;cursor:zoom-in}
figcaption{padding:13px 2px 12px}
.ttl{font-weight:600;font-size:15.5px;margin-bottom:7px}
.tag{display:inline-block;font:500 10.5px/1 "IBM Plex Mono",ui-monospace,monospace;
 padding:4px 7px;border-radius:4px;margin-left:8px;vertical-align:2px;
 text-transform:uppercase;letter-spacing:.05em;
 background:var(--wash);color:var(--accent)}
.tag.warn{background:var(--amber-wash);color:var(--amber)}
.tag.bad{background:var(--rose-wash);color:var(--rose)}
p.shows{margin:0 0 9px;color:var(--ink2);font-size:14px}
p.care{margin:0;font-size:13.5px;color:var(--ink2);background:var(--amber-wash);
 border-left:3px solid var(--amber);padding:9px 12px;border-radius:0 5px 5px 0}
p.care b{color:var(--amber)}
dialog{border:none;background:rgba(0,0,0,.9);max-width:100vw;max-height:100vh;
 width:100vw;height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.85)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
"""


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def tag_class(status: str) -> str:
    s = status.lower()
    if "refut" in s or "deflat" in s:
        return "tag bad"
    if "prelim" in s or "unspecific" in s or "descriptive" in s:
        return "tag warn"
    return "tag"


def main():
    parts = [f"<title>J-Lens Figures</title><style>{CSS}</style>",
             '<div class="wrap">',
             "<h1>J-Lens: every figure</h1>",
             '<p class="sub">The visual companion to the write-up. Each figure says '
             'what it shows and, where it matters, what it does not. Click any figure '
             'to enlarge.</p>']
    n = 0
    for f, title, status, shows, care in FIGS:
        if f == "SECTION":
            parts.append(f"<h2>{html.escape(title)}</h2>")
            continue
        p = OUT / f
        if not p.exists():
            print(f"  [miss] {f}")
            continue
        n += 1
        t = (f'<span class="{tag_class(status)}">{html.escape(status)}</span>'
             if status else "")
        c = (f'<p class="care"><b>Careful:</b> {html.escape(care)}</p>'
             if care else "")
        parts.append(
            f'<figure><img src="data:image/png;base64,{b64(p)}" '
            f'alt="{html.escape(title)}" loading="lazy">'
            f'<figcaption><div class="ttl">{html.escape(title)}{t}</div>'
            f'<p class="shows">{html.escape(shows)}</p>{c}</figcaption></figure>')
    parts.append("</div><dialog id=lb><img id=lbi alt=enlarged></dialog><script>")
    parts.append("""
const lb=document.getElementById('lb'),lbi=document.getElementById('lbi');
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{
  lbi.src=i.src;lb.showModal();});
lb.onclick=()=>lb.close();
""")
    parts.append("</script>")
    dest = OUT / "gallery.html"
    dest.write_text("\n".join(parts))
    mb = dest.stat().st_size / 1e6
    print(f"wrote {dest}  ({n} figures, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
