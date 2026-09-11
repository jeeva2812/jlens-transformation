"""Build a page you can click through: every direction, its tokens, its scores.

Nothing here computes anything. It reads the experiment outputs and lays them
out so the numbers can be checked against the actual tokens by eye.
"""
from __future__ import annotations
import argparse, glob, html, json
from pathlib import Path
from jlens.interp_score import Scorer

ARM_NOTE = {
    "top": "leading directions of J -- where everyone reads",
    "leftover": "leading directions of J after removing the top of J AND the "
                "subspace the activations occupy",
    "random": "random directions of the same size -- the floor",
    "h_in": "the part of one prompt's residual that lies inside the top of J",
    "h_out": "the part of that residual lying outside it",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/rare/viewer.html"))
    a = ap.parse_args()

    items, steer = [], {}
    for f in sorted(glob.glob("out/rare/steer_*.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            steer[(r["arm"], r["layer"], r["dir"])] = r

    for f in sorted(glob.glob("out/rare/lo_*.json")):
        d = json.load(open(f))
        tag = Path(f).stem.replace("lo_", "")
        for r in d["rows"]:
            if r["xm_z"] != r["xm_z"]:
                continue
            s = steer.get((r["arm"], r["layer"], r["dir"])) if "k" not in tag else None
            items.append(dict(run=tag, k=d["k"], prompts=d.get("prompts", "?"),
                              arm=r["arm"], layer=r["layer"], dir=r["dir"],
                              read=round(r["xm_z"], 1), coh=round(r["coh_z"], 1),
                              word=Scorer.is_content(r["top"]),
                              top=r["top"],
                              lift=round(s["lift"], 2) if s else None,
                              rlift=round(s["r_lift"], 2) if s else None))
    runs = sorted({i["run"] for i in items})
    print(f"{len(items)} directions across {len(runs)} runs; "
          f"{sum(1 for i in items if i['lift'] is not None)} of them were steered")

    body = f"""<title>J-Lens Direction Atlas</title>
<style>
:root {{
  --ink:#1c1a17; --dim:#6b645c; --line:#ddd6cb; --bg:#faf8f4; --card:#fff;
  --top:#1d6a5a; --leftover:#8a5a1e; --random:#7a7168; --hin:#3a5a8a; --hout:#8a3a5a;
  --good:#1d6a5a; --bad:#a33a2e;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  --sans:"Iowan Old Style",Palatino,Georgia,serif;
}}
@media (prefers-color-scheme:dark){{ :root:not([data-theme="light"]){{
  --ink:#e8e3da; --dim:#9a9288; --line:#332f2a; --bg:#16140f; --card:#1e1b16;
  --top:#5ec4a8; --leftover:#d9a05a; --random:#8f867b; --hin:#7aa6e0; --hout:#e08aa6;
  --good:#5ec4a8; --bad:#e0705e; }} }}
:root[data-theme="dark"]{{
  --ink:#e8e3da; --dim:#9a9288; --line:#332f2a; --bg:#16140f; --card:#1e1b16;
  --top:#5ec4a8; --leftover:#d9a05a; --random:#8f867b; --hin:#7aa6e0; --hout:#e08aa6;
  --good:#5ec4a8; --bad:#e0705e; }}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.5;padding:2.2rem 1.6rem 4rem;max-width:1180px;margin:0 auto}}
h1{{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.01em}}
.sub{{color:var(--dim);max-width:60ch;margin:0 0 1.6rem;font-size:.95rem}}
.bar{{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;
  padding:.8rem;border:1px solid var(--line);border-radius:9px;background:var(--card);
  margin-bottom:1.3rem;font-family:var(--mono);font-size:.78rem}}
select,input[type=search]{{font:inherit;padding:.3rem .45rem;border:1px solid var(--line);
  border-radius:5px;background:var(--bg);color:var(--ink)}}
label{{color:var(--dim);display:flex;gap:.32rem;align-items:center}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:.75rem}}
.c{{border:1px solid var(--line);border-left:3px solid var(--a);border-radius:8px;
  background:var(--card);padding:.7rem .8rem}}
.hd{{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;
  font-family:var(--mono);font-size:.74rem;color:var(--dim);margin-bottom:.45rem}}
.arm{{color:var(--a);font-weight:700;letter-spacing:.03em}}
.tk{{display:flex;flex-wrap:wrap;gap:.22rem;margin-bottom:.5rem}}
.tk span{{font-family:var(--mono);font-size:.73rem;background:var(--bg);
  border:1px solid var(--line);border-radius:4px;padding:.1rem .3rem;
  white-space:pre;max-width:15ch;overflow:hidden;text-overflow:ellipsis}}
.mt{{display:flex;gap:.9rem;font-family:var(--mono);font-size:.73rem;color:var(--dim);
  flex-wrap:wrap}}
.mt b{{color:var(--ink);font-weight:700}}
.up{{color:var(--good)}} .dn{{color:var(--bad)}}
.n{{color:var(--dim);font-size:.8rem;font-family:var(--mono);margin:.9rem 0 .5rem}}
table{{border-collapse:collapse;font-family:var(--mono);font-size:.78rem;
  margin-bottom:1.5rem;width:100%}}
th,td{{text-align:right;padding:.28rem .6rem;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--dim);font-weight:400}}
</style>

<h1>J-Lens Direction Atlas</h1>
<p class="sub">Every direction the experiments looked at, with the tokens it points
at, how readable it scored, and what happened when it was pushed. The scores are
only worth what the tokens beside them look like &mdash; that is what this page is
for. <b>read</b> is the cross-model score: how tightly these tokens cluster in a
<i>separately trained</i> model's embedding, in standard deviations above a matched
null. <b>lift</b> is how much held-out neighbours of these tokens moved when the
direction was pushed, minus a control set; the number in brackets is a random
direction through the same test.</p>

<div class="bar">
  <label>run <select id="run"></select></label>
  <label>arm <select id="arm"><option value="">all</option></select></label>
  <label>layer <select id="lay"><option value="">all</option></select></label>
  <label>sort <select id="srt">
    <option value="read">most readable</option>
    <option value="lift">biggest lift</option>
    <option value="layer">layer</option></select></label>
  <label><input type="checkbox" id="wd" checked> words only (hide punctuation)</label>
  <label><input type="checkbox" id="st"> only ones that were steered</label>
  <label>find <input type="search" id="q" placeholder="token" size="10"></label>
  <span id="n" style="margin-left:auto"></span>
</div>
<div id="note" class="n"></div>
<div class="grid" id="g"></div>

<script>
const D = {json.dumps(items)};
const NOTE = {json.dumps(ARM_NOTE)};
const runs = [...new Set(D.map(d=>d.run))].sort();
const arms = [...new Set(D.map(d=>d.arm))];
const lays = [...new Set(D.map(d=>d.layer))].sort((a,b)=>a-b);
const $ = i => document.getElementById(i);
runs.forEach(r=>$('run').add(new Option(r,r)));
arms.forEach(r=>$('arm').add(new Option(r,r)));
lays.forEach(r=>$('lay').add(new Option('L'+r,r)));
const esc = s => s.replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
const show = t => t===''?'\\u2423':t.replace(/\\n/g,'\\u23ce').replace(/ /g,'\\u00b7');

function draw(){{
  const run=$('run').value, arm=$('arm').value, lay=$('lay').value,
        srt=$('srt').value, wd=$('wd').checked, st=$('st').checked,
        q=$('q').value.toLowerCase();
  let r = D.filter(d=>d.run===run
    && (!arm||d.arm===arm) && (!lay||d.layer==lay)
    && (!wd||d.word) && (!st||d.lift!==null)
    && (!q||d.top.some(t=>t.toLowerCase().includes(q))));
  r.sort((a,b)=> srt==='read' ? b.read-a.read
              : srt==='lift' ? (b.lift??-9)-(a.lift??-9)
              : a.layer-b.layer || a.dir-b.dir);
  $('n').textContent = r.length+' directions';
  $('note').textContent = arm ? NOTE[arm]||'' : 'pick an arm to see what it means';
  $('g').innerHTML = r.map(d=>{{
    const L = d.lift===null ? '' :
      `<span>lift <b class="${{d.lift>Math.max(d.rlift,0.2)?'up':'dn'}}">`+
      `${{d.lift>0?'+':''}}${{d.lift}}</b> <span style="opacity:.6">`+
      `(rand ${{d.rlift>0?'+':''}}${{d.rlift}})</span></span>`;
    return `<div class="c" style="--a:var(--${{d.arm.replace('_','')}})">
      <div class="hd"><span class="arm">${{d.arm}}</span>
        <span>L${{d.layer}} &middot; d${{d.dir}} &middot; removed top-${{d.k}}</span></div>
      <div class="tk">${{d.top.map(t=>`<span>${{esc(show(t))}}</span>`).join('')}}</div>
      <div class="mt"><span>read <b>${{d.read}}</b></span>${{L}}</div></div>`;
  }}).join('');
}}
['run','arm','lay','srt','wd','st','q'].forEach(i=>{{
  $(i).addEventListener('input',draw); $(i).addEventListener('change',draw);}});
draw();
</script>"""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(body)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
