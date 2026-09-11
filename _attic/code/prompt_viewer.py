"""Prompt-by-prompt readouts, side by side, so they can be judged by eye."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("out/rare/prompt_readout_big.json"))
    ap.add_argument("--out", type=Path, default=Path("out/rare/prompts.html"))
    a = ap.parse_args()
    rows = json.load(open(a.data))
    print(f"{len(rows)} rows, {len({r['prompt'] for r in rows})} prompts")

    body = f"""<title>Readout Per Prompt</title>
<style>
:root {{
  --ink:#1b1a18; --dim:#6f6860; --line:#ded7cc; --bg:#faf8f5; --card:#fff;
  --model:#1d5f7a; --lens:#7a4a1d; --in:#1d6a4a; --out:#8a3050;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  --sans:"Iowan Old Style",Palatino,Georgia,serif; }}
@media (prefers-color-scheme:dark){{ :root:not([data-theme="light"]){{
  --ink:#e9e4db; --dim:#9b938a; --line:#332f29; --bg:#15130f; --card:#1d1b16;
  --model:#6fb8d6; --lens:#d6a468; --in:#63c4a0; --out:#e08099; }} }}
:root[data-theme="dark"]{{
  --ink:#e9e4db; --dim:#9b938a; --line:#332f29; --bg:#15130f; --card:#1d1b16;
  --model:#6fb8d6; --lens:#d6a468; --in:#63c4a0; --out:#e08099; }}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;
  padding:2.2rem 1.6rem 4rem;max-width:1080px;margin:0 auto}}
h1{{font-size:1.65rem;margin:0 0 .3rem}}
.sub{{color:var(--dim);max-width:64ch;margin:0 0 1.4rem;font-size:.95rem}}
.bar{{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;padding:.75rem;
  border:1px solid var(--line);border-radius:9px;background:var(--card);
  margin-bottom:1.2rem;font-family:var(--mono);font-size:.78rem}}
select,input{{font:inherit;padding:.28rem .45rem;border:1px solid var(--line);
  border-radius:5px;background:var(--bg);color:var(--ink)}}
label{{color:var(--dim);display:flex;gap:.3rem;align-items:center}}
.p{{border:1px solid var(--line);border-radius:9px;background:var(--card);
  padding:.85rem 1rem;margin-bottom:.8rem}}
.q{{font-family:var(--mono);font-size:.8rem;background:var(--bg);padding:.4rem .55rem;
  border-radius:5px;border:1px solid var(--line);margin-bottom:.6rem;
  white-space:pre-wrap;word-break:break-word}}
.r{{display:grid;grid-template-columns:11rem 1fr;gap:.4rem .8rem;align-items:baseline;
  margin-bottom:.28rem}}
.k{{font-family:var(--mono);font-size:.71rem;color:var(--c);text-align:right;
  font-weight:700;letter-spacing:.02em}}
.v{{display:flex;flex-wrap:wrap;gap:.2rem}}
.v span{{font-family:var(--mono);font-size:.71rem;background:var(--bg);
  border:1px solid var(--line);border-radius:4px;padding:.05rem .28rem;white-space:pre;
  max-width:16ch;overflow:hidden;text-overflow:ellipsis}}
.d{{font-family:var(--mono);font-size:.69rem;color:var(--dim);margin:.5rem 0 .2rem}}
.car{{font-family:var(--mono);font-size:.7rem;color:var(--dim);margin-left:.4rem}}
</style>
<h1>Readout Per Prompt</h1>
<p class="sub">For each prompt, the residual at one layer is pushed through J and read
through the unembedding, four ways. <b>model</b> is what the model actually predicts
next &mdash; the reference. <b>lens</b> is the whole of <code>J h</code>. <b>inside</b> and
<b>outside</b> split the residual by the top-32 subspace of J and push each half
separately. Below each, the individual directions that carried the most of
<code>J h = &Sigma; &sigma;<sub>i</sub> &lt;v<sub>i</sub>,h&gt; u<sub>i</sub></code>, ranked
by J and again after reweighting the input space by the covariance of real
activations. Measured over 153 prompts, none of these readouts identifies its own
prompt above chance &mdash; that is the point of showing them rather than a number.</p>
<div class="bar">
  <label>layer <select id="lay"></select></label>
  <label>show <select id="n"><option>25</option><option>60</option><option>200</option></select></label>
  <label>find <input type="search" id="q" placeholder="prompt or token" size="18"></label>
  <label><input type="checkbox" id="c"> show carrying directions</label>
  <span id="cnt" style="margin-left:auto"></span>
</div>
<div id="g"></div>
<script>
const D={json.dumps(rows)};
const $=i=>document.getElementById(i);
[...new Set(D.map(d=>d.layer))].sort((a,b)=>a-b).forEach(l=>$('lay').add(new Option('L'+l,l)));
const esc=s=>s.replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
const shw=t=>t===''?'\\u2423':t.replace(/\\n/g,'\\u23ce').replace(/ /g,'\\u00b7');
const chips=a=>`<div class="v">${{a.map(t=>`<span>${{esc(shw(t))}}</span>`).join('')}}</div>`;
function draw(){{
  const L=$('lay').value, n=+$('n').value, q=$('q').value.toLowerCase(), sc=$('c').checked;
  let r=D.filter(d=>d.layer==L && (!q||d.prompt.toLowerCase().includes(q)
      ||[...d.lens,...d.inside,...d.outside].some(t=>t.toLowerCase().includes(q))));
  $('cnt').textContent=r.length+' prompts';
  $('g').innerHTML=r.slice(0,n).map(d=>{{
    const row=(k,v,c)=>`<div class="r"><div class="k" style="--c:var(--${{c}})">${{k}}</div>${{chips(v)}}</div>`;
    const car=sc?`<div class="d">carried by, ranked by J</div>`+
      d.carriers_J.map(c=>`<div class="r"><div class="k" style="--c:var(--dim)">d${{c.i}} &middot; ${{(c.share*100).toFixed(1)}}%</div>${{chips(c.tokens)}}</div>`).join('')+
      `<div class="d">carried by, after weighting the input space by real activations</div>`+
      d.carriers_W.map(c=>`<div class="r"><div class="k" style="--c:var(--dim)">d${{c.i}} &middot; ${{(c.share*100).toFixed(1)}}%</div>${{chips(c.tokens)}}</div>`).join(''):'';
    return `<div class="p"><div class="q">${{esc(d.prompt)}}</div>
      ${{row('model predicts',d.model_says,'model')}}
      ${{row('lens: J h',d.lens,'lens')}}
      ${{row('inside top-32 &middot; '+Math.round(d.inside_share*100)+'%',d.inside,'in')}}
      ${{row('outside it',d.outside,'out')}}${{car}}</div>`;
  }}).join('');
}}
['lay','n','q','c'].forEach(i=>{{$(i).addEventListener('input',draw);$(i).addEventListener('change',draw);}});
draw();
</script>"""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(body)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
