"""Build the interactive SVD browser from /tmp/svd_data.json."""
import json, pathlib

D = open("out/svd_smollm2.json").read()

HEAD = r'''<title>Inside the Transport &middot; SmolLM2</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500&display=swap">
<style>
:root{--ground:#F1F4F5;--surface:#FFF;--surface2:#E7ECEE;--ink:#101719;--ink2:#3D4C51;
 --muted:#6B7D83;--hair:#CBD6D9;--accent:#0B6E78;--wash:#DCEEF0;--rose:#A83A63;--rose-wash:#F7E2EA;
 --f-sans:"IBM Plex Sans",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,monospace;
 --f-serif:"IBM Plex Serif",Georgia,serif}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0B1113;--surface:#121A1D;
 --surface2:#1A2529;--ink:#E4EDEF;--ink2:#B6C6CA;--muted:#879AA0;--hair:#26363B;--accent:#45CBD8;
 --wash:#10333A;--rose:#F286AC;--rose-wash:#3A1B27}}
:root[data-theme="dark"]{--ground:#0B1113;--surface:#121A1D;--surface2:#1A2529;--ink:#E4EDEF;
 --ink2:#B6C6CA;--muted:#879AA0;--hair:#26363B;--accent:#45CBD8;--wash:#10333A;--rose:#F286AC;--rose-wash:#3A1B27}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--f-serif);margin:0;
 padding:0 20px 80px;font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto}
header{padding:52px 0 20px;border-bottom:1px solid var(--hair);margin-bottom:22px}
.eyebrow{font-family:var(--f-mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
 color:var(--muted);margin:0 0 11px}
h1{font-family:var(--f-sans);font-weight:600;font-size:clamp(1.8rem,4.5vw,2.4rem);margin:0 0 12px;
 letter-spacing:-.02em;line-height:1.1}
.sf{color:var(--ink2);margin:0;max-width:66ch}
h2{font-family:var(--f-sans);font-weight:600;font-size:1.16rem;margin:34px 0 4px}
.note{color:var(--ink2);font-size:14.5px;margin:6px 0 14px;max-width:74ch}
.ctl{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:16px 0 12px;
 font-family:var(--f-mono);font-size:12px;color:var(--muted)}
.pills{display:flex;gap:5px;flex-wrap:wrap}
.pill{font-family:var(--f-mono);font-size:11.5px;padding:5px 10px;border:1px solid var(--hair);
 background:var(--surface);color:var(--ink2);border-radius:3px;cursor:pointer}
.pill:hover{border-color:var(--accent)}
.pill[aria-pressed="true"]{background:var(--accent);color:var(--ground);border-color:var(--accent)}
input[type=range]{accent-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(288px,1fr));gap:10px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:4px;padding:11px 13px}
.card h3{font-family:var(--f-mono);font-size:11px;color:var(--muted);margin:0 0 8px;
 display:flex;justify-content:space-between;font-weight:400}
.pole{margin:7px 0}
.pole b{font-family:var(--f-mono);font-size:10px;letter-spacing:.06em;display:block;margin-bottom:3px}
.pos b{color:var(--accent)} .neg b{color:var(--rose)}
.tks{display:flex;flex-wrap:wrap;gap:4px}
.tk{font-family:var(--f-mono);font-size:11px;padding:2px 6px;border-radius:3px;
 background:var(--surface2);color:var(--ink2);white-space:pre}
.pos .tk1{background:var(--wash);color:var(--accent);font-weight:500}
.neg .tk1{background:var(--rose-wash);color:var(--rose);font-weight:500}
table{border-collapse:collapse;font-family:var(--f-mono);font-size:12px;background:var(--surface);
 border:1px solid var(--hair);border-radius:4px;margin:12px 0}
th,td{padding:6px 9px;text-align:center;border-bottom:1px solid var(--hair)}
th{background:var(--surface2);color:var(--muted);font-weight:400;font-size:11px}
td.lab{background:var(--surface2);color:var(--ink2)}
.scroll{overflow-x:auto}
footer{margin-top:40px;padding-top:20px;border-top:1px solid var(--hair);
 font-family:var(--f-mono);font-size:12px;color:var(--muted);line-height:1.6}
code{font-family:var(--f-mono);font-size:.87em;background:var(--surface2);padding:.1em .36em;border-radius:3px}
</style>'''

BODY = r'''
<div class="wrap">
<header>
  <p class="eyebrow">SmolLM2-135M &middot; SVD of the Jacobian</p>
  <h1>Inside the Transport, Small</h1>
  <p class="sf">The same decomposition as the 7B page, on a model 50&times; smaller: 30 layers, d_model 576, target layer 28. <code>J = U S V&#7488;</code>, and each <code>U[:,i]</code> is read through the unembedding. Both signs are shown because SVD fixes only the joint sign &mdash; the pair of poles is the axis. Worth comparing against the 7B: the structural profile is nearly identical, which is the point.</p>
</header>

<h2>What each direction reads as</h2>
<p class="note">Nobody chose these tokens. They are whatever the decomposition surfaced, decoded through <code>W_U</code>. The first token in each pole is highlighted.</p>
<div class="ctl">
  <span>layer</span><span class="pills" id="layers"></span>
</div>
<div class="ctl">
  <span>showing directions 0&ndash;<b id="nlab">15</b></span>
  <input type="range" id="n" min="4" max="32" value="12" step="4" style="width:190px">
  <span id="energy"></span>
</div>
<div class="grid" id="cards"></div>

<h2>Do layers read the same subspace?</h2>
<p class="note">Mean cosine of the principal angles between two layers' top-32 <em>input</em> directions (the <code>V</code> side &mdash; what a layer responds to, not what it emits). 1.0 is identical, and the measured null for two random 32-dimensional subspaces in 4096 dimensions is shown below the table.</p>
<div class="scroll"><table id="ov"></table></div>
<p class="note" id="ovnote"></p>

<footer>
SmolLM2-135M, 25 pile-10k prompts. A full Jacobian here costs 576 cotangents rather than 4096, so this whole page is a few minutes of laptop time &mdash; which is where new experiments should be developed before being ported to the 7B.<br>
Code: github.com/jeeva2812/jlens-transformation
</footer>
</div>
<script>
const D = __DATA__;
let L = D.layers[Math.floor(D.layers.length/2)];
let N = 16;

const lp = document.getElementById('layers');
D.layers.forEach(l=>{
  const b=document.createElement('button');
  b.className='pill'; b.textContent=l; b.onclick=()=>{L=l;render();};
  lp.appendChild(b);
});
document.getElementById('n').oninput = e => { N = +e.target.value; render(); };

function esc(s){return String(s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))
  .replace(/\n/g,'\\n').replace(/^ /,'\u2423');}

function render(){
  [...lp.children].forEach(b=>b.setAttribute('aria-pressed', +b.textContent===L));
  document.getElementById('nlab').textContent = N-1;
  const e = D.energy[L];
  document.getElementById('energy').textContent =
    `top-1 holds ${(e.top1*100).toFixed(1)}% of J's energy \u00b7 top-${D.k} holds ${(e.topK*100).toFixed(1)}% \u00b7 effective rank ${e.eff_rank.toFixed(0)} of ${D.model.includes('Smol')?576:4096}`;

  document.getElementById('cards').innerHTML = D.dirs[L].slice(0,N).map(d=>{
    const row = (arr,cls) => `<div class="pole ${cls}"><b>${cls==='pos'?'+':'\u2212'}U[:,${d.i}]</b>`
      + `<div class="tks">${arr.map((t,j)=>`<span class="tk ${j===0?'tk1':''}">${esc(t[0])} ${(t[1]*100).toFixed(0)}%</span>`).join('')}</div></div>`;
    return `<div class="card"><h3><span>direction ${d.i}</span><span>s = ${d.s.toFixed(2)}</span></h3>`
      + row(d.pos,'pos') + row(d.neg,'neg') + `</div>`;
  }).join('');
}

const ls = D.layers;
let h = '<tr><th></th>' + ls.map(l=>`<th>${l}</th>`).join('') + '</tr>';
for (const a of ls){
  h += `<td class="lab">L${a}</td>`;
  h = h.replace(/<td class="lab">L\d+<\/td>$/, m=>'<tr>'+m);
  for (const b of ls){
    const v = D.overlap[a][b];
    const t = Math.max(0,Math.min(1,(v - D.null)/(1 - D.null)));
    h += `<td style="background:color-mix(in srgb, var(--accent) ${Math.round(t*70)}%, transparent)">${v.toFixed(2)}</td>`;
  }
  h += '</tr>';
}
document.getElementById('ov').innerHTML = h;
document.getElementById('ovnote').textContent =
  `Measured null (mean over 5 random subspace pairs): ${D.null}. Shading is scaled from that null to 1.0, so anything pale is indistinguishable from chance.`;
render();
</script>'''

out = HEAD + BODY.replace("__DATA__", D)
pathlib.Path("out/svd_viz_small.html").write_text(out)
print("wrote out/svd_viz_small.html", len(out), "bytes")
