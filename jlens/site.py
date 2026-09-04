"""The interactive explorer: browse J and dJ subspaces by model, checkpoint, layer.

Static HTML with the precomputed tables inlined, so it works from a file:// URL
and from GitHub Pages with no server and no build step.

Four views:
  subspaces   sigma spectrum + top directions of J, read as tokens
  diff        any two checkpoints: sigma of dJ, and BOTH sides read -- u for what
              the change emits, J_base @ v for what it responds to
  steering    the corrected assay, u vs v, filterable
  dynamics    the static figures that do not fit a selector
"""
from __future__ import annotations
import base64, json
from pathlib import Path

OUT = Path("out")
EXP = OUT / "explorer"


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


CSS = """
:root{--ground:#F2F5F6;--surface:#FFF;--surface2:#E8EDEF;--ink:#0E1518;--ink2:#3A4A50;
--muted:#68797F;--hair:#CCD7DA;--accent:#0B6E78;--wash:#DBEEF0;--rose:#A83A63;
--rose-wash:#F8E3EA;--amber:#8A6410;--amber-wash:#FBF0D8}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;--ink:#E9F0F2;--ink2:#B6C5CA;
--muted:#8399A0;--hair:#293539;--wash:#11302E;--rose-wash:#381A27;--amber-wash:#2D2410}}
:root[data-theme=dark]{--ground:#0D1315;--surface:#151D20;--surface2:#1C262A;
--ink:#E9F0F2;--ink2:#B6C5CA;--muted:#8399A0;--hair:#293539;--wash:#11302E;
--rose-wash:#381A27;--amber-wash:#2D2410}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
font:15px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:30px 20px 90px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin:0 0 22px;max-width:75ch}
.tabs{display:flex;gap:5px;border-bottom:2px solid var(--hair);margin:0 0 22px;
flex-wrap:wrap}
.tabs button{background:none;border:none;border-bottom:2px solid transparent;
margin-bottom:-2px;padding:9px 15px;font:500 14px/1 inherit;color:var(--muted);
cursor:pointer;border-radius:5px 5px 0 0}
.tabs button:hover{color:var(--ink)}
.tabs button.on{color:var(--accent);border-bottom-color:var(--accent);
background:var(--wash)}
.panel{display:none}.panel.on{display:block}
.ctrl{display:flex;gap:16px;flex-wrap:wrap;background:var(--surface);
border:1px solid var(--hair);border-radius:9px;padding:14px 16px;margin:0 0 18px;
align-items:flex-end}
.ctrl label{display:block;font:500 10.5px/1.4 "IBM Plex Mono",ui-monospace,monospace;
text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:5px}
.ctrl select{font:14px inherit;padding:6px 9px;border:1px solid var(--hair);
border-radius:6px;background:var(--surface);color:var(--ink);min-width:130px}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin:0 0 16px;font-size:13px;
color:var(--ink2)}
.stats b{font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--ink)}
.spec{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
padding:14px 16px 8px;margin:0 0 18px}
.spec h4{margin:0 0 4px;font-size:13.5px}
.spec p{margin:0 0 8px;font-size:12.5px;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
padding:13px 15px}
.card.hi{border-color:var(--accent)}
.chead{display:flex;justify-content:space-between;align-items:baseline;
margin-bottom:9px;gap:8px}
.cid{font:600 14px/1 inherit}
.cnum{font:11.5px/1 "IBM Plex Mono",ui-monospace,monospace;color:var(--muted)}
.bar{height:4px;background:var(--surface2);border-radius:2px;overflow:hidden;
margin:0 0 10px}
.bar i{display:block;height:100%;background:var(--accent)}
.side{margin:0 0 8px}
.side .lab{font:500 10px/1.4 "IBM Plex Mono",ui-monospace,monospace;
text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:4px}
.tk{display:inline-block;font:12.5px/1 "IBM Plex Mono",ui-monospace,monospace;
background:var(--surface2);padding:4px 6px;border-radius:4px;margin:0 3px 4px 0;
white-space:pre}
.tk.n{background:var(--rose-wash)}
.tk.p{background:var(--wash)}
.z{font:10px/1 "IBM Plex Mono",ui-monospace,monospace;padding:3px 5px;
border-radius:3px;background:var(--surface2);color:var(--muted)}
.z.good{background:var(--wash);color:var(--accent)}
.note{background:var(--amber-wash);border-left:3px solid var(--amber);
border-radius:0 6px 6px 0;padding:11px 14px;margin:0 0 18px;font-size:13.5px}
.hl{background:var(--wash);border:1px solid var(--accent);border-radius:9px;
padding:14px 16px;margin:0 0 18px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--hair)}
th{font:500 10.5px/1.3 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);cursor:pointer;user-select:none}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}
.ok{color:var(--accent);font-weight:600}.no{color:var(--muted)}
figure{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
margin:0 0 18px;padding:14px 14px 4px}
figure img{width:100%;display:block;border-radius:4px;background:#fff;cursor:zoom-in}
figcaption{padding:11px 3px;font-size:13.5px;color:var(--ink2)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em;
background:var(--surface2);padding:1px 5px;border-radius:3px}
dialog{border:none;background:rgba(0,0,0,.92);width:100vw;height:100vh;
max-width:100vw;max-height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.88)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
.empty{color:var(--muted);padding:30px;text-align:center;font-size:14px}
"""

JS = r"""
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const show=t=>{$$('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
  $$('.panel').forEach(p=>p.classList.toggle('on',p.id==='p-'+t));};
$$('.tabs button').forEach(b=>b.onclick=()=>show(b.dataset.t));

function spectrum(vals, el, label){
  if(!vals||!vals.length){el.innerHTML='';return;}
  const W=760,H=120,n=Math.min(vals.length,40),mx=Math.max(...vals);
  const bw=W/n;
  let bars='';
  for(let i=0;i<n;i++){
    const h=Math.max(1,(vals[i]/mx)*(H-22));
    bars+=`<rect x="${i*bw+1}" y="${H-h-14}" width="${bw-2}" height="${h}"
      fill="${i<8?'var(--accent)':'var(--muted)'}" opacity="${i<8?0.95:0.45}">
      <title>sigma_${i} = ${vals[i]}</title></rect>`;
    if(i%5===0)bars+=`<text x="${i*bw+bw/2}" y="${H-3}" font-size="9"
      fill="var(--muted)" text-anchor="middle">${i}</text>`;
  }
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${bars}</svg>
    <p style="margin:4px 0 0">${label}</p>`;
}

function tokens(list, cls){
  return list.map(t=>`<span class="tk ${cls}">${esc(JSON.stringify(t).slice(1,-1))}</span>`).join('');
}

/* ---------- subspaces ---------- */
function fillSel(sel, items, val){
  sel.innerHTML=items.map(o=>`<option value="${o.v}">${esc(o.l)}</option>`).join('');
  // keep the old selection only if it still exists, else fall back to the first
  const keep=items.some(o=>String(o.v)===String(val));
  sel.value=keep?val:items[0].v;
}
function modelOpts(){return Object.keys(DATA).map(k=>({v:k,l:DATA[k].title}));}

function syncSub(){
  const m=$('#s-model').value,D=DATA[m];
  fillSel($('#s-ckpt'),D.checkpoints.map(c=>({v:c.id,l:c.label})),$('#s-ckpt').value);
  fillSel($('#s-layer'),D.layers.map(l=>({v:l,l:'layer '+l})),$('#s-layer').value);
  drawSub();
}
function drawSub(){
  const m=$('#s-model').value,D=DATA[m];
  const k=$('#s-ckpt').value+'|'+$('#s-layer').value, r=D.single[k];
  if(!r){$('#s-cards').innerHTML='<div class="empty">not computed for this combination</div>';
    $('#s-stats').innerHTML='';$('#s-spec').innerHTML='';return;}
  $('#s-stats').innerHTML=
    `<span>effective rank <b>${r.eff_rank.toFixed(1)}</b></span>
     <span>mean diagonal <b>${r.diag.toFixed(3)}</b></span>
     <span>&#8214;J&minus;I&#8214;/&#8214;J&#8214; <b>${r.dev.toFixed(3)}</b></span>
     <span>top-8 share <b>${(r.dirs.slice(0,8).reduce((a,d)=>a+d.share,0)*100).toFixed(1)}%</b></span>`;
  spectrum(r.spectrum,$('#s-spec'),
    'singular values of J, largest 40. Teal = the 8 directions shown below.');
  $('#s-cards').innerHTML=r.dirs.map(d=>`
    <div class="card ${d.z>3?'hi':''}">
      <div class="chead"><span class="cid">direction ${d.i}</span>
        <span class="cnum">&sigma;=${d.sigma.toFixed(2)} &middot; ${(d.share*100).toFixed(1)}%
        <span class="z ${d.z>3?'good':''}">z ${d.z}</span></span></div>
      <div class="bar"><i style="width:${Math.min(100,d.share*100*6)}%"></i></div>
      <div class="side"><div class="lab">+ direction</div>${tokens(d.pos,'p')}</div>
      <div class="side"><div class="lab">&minus; direction</div>${tokens(d.neg,'n')}</div>
    </div>`).join('');
}

/* ---------- diff ---------- */
function syncDiff(){
  const m=$('#d-model').value,D=DATA[m];
  const cs=D.checkpoints.map(c=>({v:c.id,l:c.label}));
  fillSel($('#d-a'),cs,$('#d-a').value||cs[0].v);
  fillSel($('#d-b'),cs,$('#d-b').value||cs[cs.length-1].v);
  fillSel($('#d-layer'),D.layers.map(l=>({v:l,l:'layer '+l})),$('#d-layer').value);
  drawDiff();
}
function drawDiff(){
  const m=$('#d-model').value,D=DATA[m];
  const a=$('#d-a').value,b=$('#d-b').value,l=$('#d-layer').value;
  let r=D.diff[a+'>'+b+'|'+l], flipped=false;
  if(!r){r=D.diff[b+'>'+a+'|'+l];flipped=!!r;}
  if(!r){$('#d-cards').innerHTML='<div class="empty">This pair was not precomputed. '
      +'Olmo stores all pairs at layers 8/16/24 only &mdash; a 4096&times;4096 SVD per '
      +'pair per layer is the binding cost.</div>';
    $('#d-stats').innerHTML='';$('#d-spec').innerHTML='';return;}
  $('#d-stats').innerHTML=
    `<span>&#8214;&Delta;J&#8214;/&#8214;J&minus;I&#8214; <b>${r.rel.toFixed(3)}</b></span>
     <span>top-8 share <b>${(r.dirs.slice(0,8).reduce((x,d)=>x+d.share,0)*100).toFixed(1)}%</b></span>
     ${flipped?'<span style="color:var(--muted)">(computed in the reverse order; '
       +'directions are sign-symmetric)</span>':''}`;
  spectrum(r.spectrum,$('#d-spec'),
    'singular values of &Delta;J. A flat spectrum means the change is diffuse.');
  $('#d-cards').innerHTML=r.dirs.map(d=>`
    <div class="card ${d.z>3?'hi':''}">
      <div class="chead"><span class="cid">direction ${d.i}</span>
        <span class="cnum">&sigma;=${d.sigma.toFixed(3)} &middot; ${(d.share*100).toFixed(1)}%
        <span class="z ${d.z>3?'good':''}">z ${d.z}</span></span></div>
      <div class="bar"><i style="width:${Math.min(100,d.share*100*6)}%"></i></div>
      <div class="side"><div class="lab">responds to &nbsp;<code>J_base &middot; v</code></div>
        ${tokens(d.responds,'p')}</div>
      <div class="side"><div class="lab">sends to &nbsp;<code>u</code></div>
        ${tokens(d.sends,'n')}</div>
    </div>`).join('');
}

/* ---------- steering ---------- */
let stSort={k:'shift_v',dir:-1};
function drawSteer(){
  const src=$('#st-src').value, rows=STEER[src]||[];
  const f=$('#st-filter').value;
  let rs=rows.slice();
  if(f==='pass')rs=rs.filter(r=>Math.abs(r.shift_v)>0.5&&Math.abs(r.shift_v)>4*Math.abs(r.shift_rand));
  if(f==='flip')rs=rs.filter(r=>{
    const pu=Math.abs(r.shift_u)>0.5&&Math.abs(r.shift_u)>4*Math.abs(r.shift_rand);
    const pv=Math.abs(r.shift_v)>0.5&&Math.abs(r.shift_v)>4*Math.abs(r.shift_rand);
    return pv&&!pu;});
  rs.sort((x,y)=>(Math.abs(y[stSort.k])-Math.abs(x[stSort.k]))*(stSort.dir<0?1:-1));
  const pass=r=>Math.abs(r.shift_v)>0.5&&Math.abs(r.shift_v)>4*Math.abs(r.shift_rand);
  const passU=r=>Math.abs(r.shift_u)>0.5&&Math.abs(r.shift_u)>4*Math.abs(r.shift_rand);
  $('#st-sum').innerHTML=`<span>showing <b>${rs.length}</b> of ${rows.length}</span>
    <span>steer with v: <b>${rows.filter(pass).length}</b> pass</span>
    <span>steer with u: <b>${rows.filter(passU).length}</b> pass</span>`;
  $('#st-body').innerHTML=rs.map(r=>`<tr>
    <td class="n">${r.layer}</td><td class="n">${r.dir}</td>
    <td class="n">${r.cos_uv.toFixed(3)}</td>
    <td class="n">${esc(JSON.stringify(r.pos))}</td>
    <td class="n">${esc(JSON.stringify(r.neg))}</td>
    <td class="n ${passU(r)?'ok':'no'}">${r.shift_u.toFixed(2)}</td>
    <td class="n ${pass(r)?'ok':'no'}">${r.shift_v.toFixed(2)}</td>
    <td class="n no">${r.shift_rand.toFixed(2)}</td></tr>`).join('');
}
"""

TITLES = {"olmo": "Olmo 3 7B — 11 training checkpoints",
          "smollm2_ft": "SmolLM2-135M — 13 fine-tuning checkpoints",
          "qwen_em": "Qwen2.5-0.5B — 4 LoRA fine-tunes + base"}

HTML = """
<h1>J-Lens explorer</h1>
<p class="sub">Browse the subspaces of the Jacobian lens: pick a model, a
checkpoint and a layer to see what the transport amplifies and how those
directions read as tokens. Diff any two checkpoints to see what training or
fine-tuning changed, read from both sides.</p>

<div class="tabs">
<button data-t="sub" class="on">Subspaces of J</button>
<button data-t="diff">Diff (&Delta;J)</button>
<button data-t="steer">Steering</button>
<button data-t="dyn">Training dynamics</button>
<button data-t="how">How to read this</button>
</div>

<div class="panel on" id="p-sub">
  <div class="ctrl">
    <div><label>model</label><select id="s-model"></select></div>
    <div><label>checkpoint</label><select id="s-ckpt"></select></div>
    <div><label>layer</label><select id="s-layer"></select></div>
  </div>
  <div class="stats" id="s-stats"></div>
  <div class="spec" id="s-spec"></div>
  <div class="note"><b>Both sides of a plain-J direction read the same.</b> Since
  <code>J v_i = &sigma;_i u_i</code> exactly and the readout normalises, reading
  <code>u</code> and reading <code>J&middot;v</code> are the same operation here.
  The two only diverge for &Delta;J, which is why the diff tab shows both.</div>
  <div class="cards" id="s-cards"></div>
</div>

<div class="panel" id="p-diff">
  <div class="ctrl">
    <div><label>model</label><select id="d-model"></select></div>
    <div><label>from</label><select id="d-a"></select></div>
    <div><label>to</label><select id="d-b"></select></div>
    <div><label>layer</label><select id="d-layer"></select></div>
  </div>
  <div class="stats" id="d-stats"></div>
  <div class="spec" id="d-spec"></div>
  <div class="hl"><b>&Delta;J = U S V&#7488; is a set of rules:</b> <i>if this pattern
  arrives at layer &ell;, add this to the output.</i> <code>v_i</code> is the IF and
  <code>u_i</code> is the THEN. <code>u</code> is already in target space so
  <code>W_U</code> reads it directly; <code>v</code> lives in layer-&ell; space and
  must be carried there by the <b>base</b> transport first. It has to be
  <code>J_base</code> and not <code>&Delta;J</code> &mdash; reading
  <code>&Delta;J&middot;v</code> just returns <code>u</code> again.</div>
  <div class="cards" id="d-cards"></div>
</div>

<div class="panel" id="p-steer">
  <div class="ctrl">
    <div><label>run</label><select id="st-src"></select></div>
    <div><label>show</label><select id="st-filter">
      <option value="all">all directions</option>
      <option value="pass">only those that steer (with v)</option>
      <option value="flip">only those the correction rescued</option>
    </select></div>
  </div>
  <div class="stats" id="st-sum"></div>
  <div class="note"><b>Robustness.</b> Across three runs the corrected pass rate is
  64% / 62% / 67%. Seed agreement is <b>90%</b> and alpha agreement <b>88%</b> &mdash;
  both higher than the buggy path's 71% and 80%, which is what you would expect if
  injecting the wrong vector was adding noise as well as losing signal.</div>
  <div class="hl"><b>Read from u, steer with v.</b> <code>J_&ell;</code> maps
  layer-&ell; space to target space, so <code>u</code> is in target space (readable by
  <code>W_U</code>) and <code>v</code> is in layer-&ell; space (the thing you inject).
  The original assay injected <code>u</code>, which is a type error; the
  <code>shift[u]</code> column is that path and <code>shift[v]</code> is the correct
  one. A direction counts as steering if |shift| &gt; 0.5 and &gt; 4&times; the random
  control.</div>
  <table><thead><tr>
    <th>layer</th><th>dir</th><th>cos(u,v)</th><th>+pole</th><th>&minus;pole</th>
    <th>shift[u]</th><th>shift[v]</th><th>random</th>
  </tr></thead><tbody id="st-body"></tbody></table>
</div>

<div class="panel" id="p-dyn">FIGURES</div>

<div class="panel" id="p-how">
  <h3>What you are looking at</h3>
  <p><code>J_&ell; = E[&part;h_target/&part;h_&ell;]</code> is a d_model &times; d_model
  linear map from the residual stream at layer &ell; to the residual stream near the
  end of the network. It is computed by autodiff from the weights &mdash; nothing is
  trained. Multiply by the unembedding and any layer-&ell; direction can be read as a
  distribution over tokens.</p>
  <h3>Sigma, share, and effective rank</h3>
  <p>&sigma;_i is how much the transport amplifies direction i. <b>Share</b> is
  &sigma;_i&sup2; as a fraction of the total, so it says how much of the transport
  that one direction accounts for. <b>Effective rank</b> is
  (&Sigma;&sigma;)&sup2;/&Sigma;&sigma;&sup2; &mdash; roughly how many directions
  genuinely carry the map. A high effective rank with a flat spectrum means there is
  no small set of directions to point at.</p>
  <h3>The z-score next to each direction</h3>
  <p>How far that direction's readout sits from a random-direction null at the same
  layer. It exists because a top-k token list is <b>much weaker evidence than it
  looks</b>: testing a direction against held-out US/UK spelling pairs showed that
  ~29% of random directions separate all 20 pairs in the same direction, because
  themed token sets are correlated in unembedding space. Consistency is worthless;
  magnitude discriminates. Treat z &lt; 3 readouts as decoration.</p>
  <h3>Why some readouts look like nonsense</h3>
  <p>SVD of J finds directions the transport <i>amplifies</i>, which says nothing
  about whether the model ever visits them. Measured on real activations, the top-64
  raw singular directions capture only <b>25&ndash;32%</b> of real transported
  activation in layers 12&ndash;24 (66&ndash;68% in early layers), where PCA of
  <i>Jh</i> captures 89&ndash;99%. Most of the leading geometry is off-manifold:
  real perturbations the model never experiences.</p>
  <h3>Caveats</h3>
  <ul>
  <li>Olmo stores all checkpoint pairs at layers 8/16/24 only; a 4096&times;4096 SVD
  per pair per layer is the binding cost.</li>
  <li>Every J here is averaged over a fixed prompt set. That choice is not a detail
  &mdash; the same &Delta;J analysis gave opposite answers on Pile text and on
  chat-formatted prompts.</li>
  <li>The steering table is the corrected path. Numbers published before it &mdash;
  46%, 42%, and a sharp depth profile &mdash; came from the buggy one.</li>
  </ul>
</div>
"""


def main():
    data, steer = {}, {}
    for f in sorted(EXP.glob("*.json")):
        d = json.loads(f.read_text())
        d["title"] = TITLES.get(f.stem, f.stem)
        data[f.stem] = d
        print(f"  {f.stem}: {len(d['single'])} single, {len(d['diff'])} diff")
    for name, path in [("SmolLM2 · seed 0 · α=0.01", "out/assay_uv_s0a01.json"),
                       ("SmolLM2 · seed 1 · α=0.01", "out/assay_uv_s1a01.json"),
                       ("SmolLM2 · seed 0 · α=0.02", "out/assay_uv_s0a02.json"),
                       ("Olmo 3 7B · α=0.10", "out/assay_uv_olmo.json")]:
        p = Path(path)
        if p.exists():
            steer[name] = json.loads(p.read_text())
            print(f"  steer {name}: {len(steer[name])} rows")

    figs = []
    for path, cap in [
        ("report/F1_uv_correction.png",
         "<b>The type error and its signature.</b> The correction's size is predicted "
         "by cos(u,v) &mdash; that is the evidence the diagnosis is right."),
        ("viz_A_heatmap.png",
         "<b>Subspace overlap with the final model.</b> Chance 0.11. Middle layers "
         "converge first and furthest; the deepest lag."),
        ("report/F4_olmo_phases.png",
         "<b>Each training phase changes the transport less than the last.</b>"),
        ("report/F5_ft_traj.png",
         "<b>Two learning rates, orthogonal directions</b> (cos &minus;0.013)."),
        ("report/F6_qwen_domain.png",
         "<b>&Delta;J domain recovery:</b> no false positives, silent half the time."),
        ("report/F2_offmanifold.png",
         "<b>Two-thirds of the leading geometry is off-manifold.</b>"),
        ("report/F3_label_null.png",
         "<b>Consistency is worthless; magnitude discriminates.</b>"),
        ("viz_E_locality.png",
         "<b>Depth-locality is present at initialisation</b> &mdash; refuted my "
         "prediction that it emerges during training."),
    ]:
        p = OUT / path
        if p.exists():
            figs.append(f'<figure><img src="data:image/png;base64,{b64(p)}" alt="">'
                        f'<figcaption>{cap}</figcaption></figure>')

    body = HTML.replace("FIGURES", "\n".join(figs))
    boot = f"""
const DATA={json.dumps(data)};
const STEER={json.dumps(steer)};
fillSel($('#s-model'),modelOpts()); fillSel($('#d-model'),modelOpts());
fillSel($('#st-src'),Object.keys(STEER).map(k=>({{v:k,l:k}})));
$('#s-model').onchange=syncSub; $('#s-ckpt').onchange=drawSub; $('#s-layer').onchange=drawSub;
$('#d-model').onchange=()=>{{$('#d-a').value='';$('#d-b').value='';syncDiff();}};
['#d-a','#d-b','#d-layer'].forEach(s=>$(s).onchange=drawDiff);
$('#st-src').onchange=drawSteer; $('#st-filter').onchange=drawSteer;
syncSub(); syncDiff(); drawSteer();
const lb=document.getElementById('lb'),i2=document.getElementById('lbi');
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{i2.src=i.src;lb.showModal();}});
lb.onclick=()=>lb.close();
"""
    parts = ["<title>J-Lens Explorer</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600'
             '&display=swap">',
             f"<style>{CSS}</style>", f'<div class="wrap">{body}</div>',
             '<dialog id=lb><img id=lbi alt=""></dialog>',
             f"<script>{JS}\n{boot}</script>"]
    dest = OUT / "explorer.html"
    dest.write_text("\n".join(parts))
    print(f"wrote {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
