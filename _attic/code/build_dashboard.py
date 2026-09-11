"""Build out/DASHBOARD.html from out/labels.json (self-contained, no CDN)."""
from __future__ import annotations
import json
from pathlib import Path
import html

SRC = Path("out/labels.json")
DST = Path("out/DASHBOARD.html")


def main():
    recs = json.loads(SRC.read_text())

    def sort_key(r):
        v = 0 if (r["verdict"] == "validated" and (r.get("steer") or {}).get("steers") == "visibly") else \
            1 if r["verdict"] == "validated" else 2 if r["verdict"] == "rejected" else 3
        return (v, r["model"], r["layer"], r["family"], r["index"])

    recs.sort(key=sort_key)
    data = json.dumps(recs).replace("<", "\\u003c")
    n = len(recs)
    nv = sum(1 for r in recs if r["verdict"] == "validated")
    nvis = sum(1 for r in recs if (r.get("steer") or {}).get("steers") == "visibly")

    DST.write_text("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>J-Lens labels dashboard</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
#filters{position:sticky;top:0;background:#fff;padding:10px;border:1px solid #ddd;margin-bottom:12px}
#filters select{margin-right:8px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0}
.card h3{margin:0 0 4px;font-size:14px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;margin-right:4px}
.v-validated{background:#d6f5d6}.v-rejected{background:#f5d6d6}.v-no-hypothesis{background:#eee}
.s-visibly{background:#b3e0ff}.s-metric-only{background:#fff0b3}.s-no{background:#f0f0f0}
.toks{font-family:monospace;font-size:12px;color:#444;word-break:break-word}
.ex{font-size:13px;margin:4px 0}.ex b{font-weight:600}
.hide{display:none}
</style></head><body>
<h2>J-Lens systematic labels (%d directions, %d validated, %d visibly steering)</h2>
<div id="legend" style="background:#fff;border:1px solid #ddd;padding:8px 10px;margin-bottom:10px;font-size:13px">
<b>Legend.</b> Models: SmolLM2-135M &middot; J &middot; base (pre-fine-tune) |
SmolLM2-135M &middot; &Delta;J &middot; insecure-code fine-tune (J_step600 &minus; J_step0) |
Qwen2.5-0.5B-Instruct &middot; J &middot; all 22 layers | Llama-3.2-1B-Instruct &middot; J &middot; 4 of 16 layers.
<b>All records here have been through the re-audit</b>: a hypothesis is withdrawn unless at least
4 of the direction's 15 top tokens are real words from that concept's lexicon (rule 1), and a
direction is rejected if |score|/null &lt; 1.15 (rule 2). Qwen at full 22-layer coverage yields
<b>zero</b> validated directions &mdash; so the earlier non-replication was not a layer-coverage artifact.
Matrix <b>J</b> = transport Jacobian at a layer; <b>&Delta;J</b> = fine-tune change.
For &Delta;J records the two readouts are labelled <b>responds to (input side)</b>
(what the change reacts to, read via the base transport) and
<b>sends to (output side)</b> (what it adds to the output).
Hypothesis types: <b>pair</b> = contrast of word pairs, <b>set</b> = theme-token cluster vs matched neutrals.
</div>
<div id="filters">
Filter:
model <select id="f_model"><option value="">all</option></select>
matrix <select id="f_matrix"><option value="">all</option></select>
layer <select id="f_layer"><option value="">all</option></select>
family <select id="f_family"><option value="">all</option></select>
verdict <select id="f_verdict"><option value="">all</option></select>
steers <select id="f_steers"><option value="">all</option></select>
hypothesis type <select id="f_type"><option value="">all</option></select>
<span id="count"></span>
</div>
<div id="cards"></div>
<script>
const DATA = %s;
const $ = id => document.getElementById(id);
const NAMES = {'smollm2-ft-step0':'SmolLM2-135M \u00b7 J \u00b7 base (pre-fine-tune)',
 'smollm2-delta':'SmolLM2-135M \u00b7 \u0394J \u00b7 insecure-code fine-tune',
 'qwen-base':'Qwen2.5-0.5B-Instruct \u00b7 J \u00b7 5 layers (superseded)',
 'qwen-dense':'Qwen2.5-0.5B-Instruct \u00b7 J \u00b7 all 22 layers',
 'llama-base':'Llama-3.2-1B-Instruct \u00b7 J \u00b7 4 of 16 layers'};
const MAT = {'J':'J','dJ':'\u0394J'};
function dname(r){return NAMES[r.model]||r.model;}
function sideLabel(h){
  if(!h) return h;
  return h.replace(/ \(in\)$/,' \u2014 responds to (input side)').replace(/ \(out\)$/,' \u2014 sends to (output side)');
}
function uniq(k){return [...new Set(DATA.map(r=>String(r[k])))].sort();}
function fill(id, vals){const s=$(id);vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o);});}
function fillPairs(id, pairs){const s=$(id);pairs.forEach(p=>{const o=document.createElement('option');o.value=p[0];o.textContent=p[1];s.appendChild(o);});}
fillPairs('f_model',uniq('model').map(v=>[v,NAMES[v]||v]));
fillPairs('f_matrix',uniq('matrix').map(v=>[v,MAT[v]||v]));
fill('f_layer',uniq('layer'));
fill('f_family',uniq('family'));
fill('f_verdict',uniq('verdict'));
fill('f_steers',['visibly','metric-only','no','unsteered']);
fill('f_type',['pair','set']);
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function render(){
  const f={model:$('f_model').value,matrix:$('f_matrix').value,layer:$('f_layer').value,
    family:$('f_family').value,verdict:$('f_verdict').value,steers:$('f_steers').value,
    htype:$('f_type').value};
  const box=$('cards');box.innerHTML='';let shown=0;
  for(const r of DATA){
    if(f.model&&String(r.model)!==f.model)continue;
    if(f.matrix&&String(r.matrix)!==f.matrix)continue;
    if(f.layer&&String(r.layer)!==f.layer)continue;
    if(f.family&&String(r.family)!==f.family)continue;
    if(f.verdict&&String(r.verdict)!==f.verdict)continue;
    const st=(r.steer&&r.steer.steers)||'unsteered';
    if(f.steers&&st!==f.steers)continue;
    if(f.htype&&String(r.hypothesis_type||'pair')!==f.htype)continue;
    shown++;
    if(shown>600){continue;}
    const d=document.createElement('div');d.className='card';
    let h='<h3>'+esc(dname(r))+' L'+r.layer+' '+esc(r.family)+'['+r.index+']</h3>';
    h+='<span class="badge v-'+esc(r.verdict)+'">'+esc(r.verdict)+'</span>';
    h+='<span class="badge s-'+esc(st)+'">'+esc(st)+'</span>';
    if(r.hypothesis_type)h+='<span class="badge">'+esc(r.hypothesis_type)+'</span>';
    if(r.hypothesis)h+='<div>hypothesis: <b>'+esc(sideLabel(r.hypothesis))+'</b> score '+r.score.toFixed(1)+' vs thr '+r.null_threshold.toFixed(1)+' (null%% beating '+r.null_pct_beating.toFixed(2)+', strength '+(r.strength!=null?r.strength:'?')+')</div>';
    else h+='<div>hypothesis: null</div>';
    if(r.matrix==='dJ'){
      h+='<div class="toks">sends to (output side) + '+(r.tokens_pos||[]).slice(0,12).map(esc).join(' | ')+'</div>';
      h+='<div class="toks">sends to (output side) - '+(r.tokens_neg||[]).slice(0,12).map(esc).join(' | ')+'</div>';
      if(r.tokens_in_pos)h+='<div class="toks">responds to (input side) + '+(r.tokens_in_pos||[]).slice(0,8).map(esc).join(' | ')+'</div>';
      if(r.tokens_in_neg)h+='<div class="toks">responds to (input side) - '+(r.tokens_in_neg||[]).slice(0,8).map(esc).join(' | ')+'</div>';
    } else {
      h+='<div class="toks">+ '+(r.tokens_pos||[]).slice(0,12).map(esc).join(' | ')+'</div>';
      h+='<div class="toks">- '+(r.tokens_neg||[]).slice(0,12).map(esc).join(' | ')+'</div>';
    }
    if(r.steer&&r.steer.examples)for(const e of r.steer.examples){
      h+='<div class="ex"><b>'+esc(e.prompt)+'</b><br>base: '+esc(e.base)+'<br>plus: '+esc(e.plus)+'<br>minus: '+esc(e.minus)+'</div>';
    }
    if(r.steer&&r.steer.alpha!=null)h+='<div>alpha '+r.steer.alpha+' shift '+r.steer.shift+' rand '+r.steer.random_shift+'</div>';
    if(r.steer&&(r.steer.steer_note||r.steer.note))h+='<div><i>steer note: '+esc(r.steer.steer_note||r.steer.note)+'</i></div>';
    if(r.audit_note)h+='<div><i>'+esc(r.audit_note)+'</i></div>';
    d.innerHTML=h;box.appendChild(d);
  }
  $('count').textContent='showing '+Math.min(shown,600)+' of '+DATA.length+' (render capped at 600; use filters)';
}
document.querySelectorAll('#filters select').forEach(s=>s.addEventListener('change',render));
render();
</script></body></html>
""" % (n, nv, nvis, data))
    print(f"wrote {DST} ({DST.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
