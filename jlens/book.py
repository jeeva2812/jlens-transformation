"""Assemble the book: everything from the top, for a reader starting cold."""
from __future__ import annotations
import base64, sys
from pathlib import Path
sys.path.insert(0, "book")
from style import CSS
from ch1 import PART1
from ch2 import PART2
from ch3 import PART3
from ch4 import PART4
from ch5 import PART5
from ch6 import PART6
from ch7 import PART7
from ch8 import PART8
from ch9 import PART9

OUT = Path("out")
FIGS = {"FIG_F7": "report/F7_subspace_formation.png",
        "FIG_F13": "report/F13_dissociation.png",
        "FIG_F15": "report/F15_position_sweep.png"}

TOC = """
<div class="toc">
<b>Part I &mdash; What we are looking at</b>
<a href="#p1"><i>1</i>What a transformer is doing, in one picture</a>
<a href="#p1"><i>2</i>Lenses, and why anyone wants one</a>
<a href="#p1"><i>3</i>What a Jacobian is, and what J-Lens is</a>
<b>Part II &mdash; Is the instrument working?</b>
<a href="#p2"><i>4</i>Before anything else: checking against someone else's</a>
<a href="#p2"><i>5</i>Four ways to get it wrong without noticing</a>
<a href="#p2"><i>6</i>The metric trap</a>
<b>Part III &mdash; Looking inside</b>
<a href="#p3"><i>7</i>Opening up the matrix: what SVD is</a>
<a href="#p3"><i>8</i>Why so much of it is junk</a>
<a href="#p3"><i>9</i>Telling a label from a measurement</a>
<b>Part IV &mdash; Does it mean anything?</b>
<a href="#p4"><i>10</i>Steering</a>
<a href="#p4"><i>11</i>The bug that ate a finding</a>
<b>Part V &mdash; Change over time</b>
<a href="#p5"><i>12</i>Watching a model learn</a>
<a href="#p5"><i>13</i>Model diffing</a>
<b>Part VI &mdash; The mathematics</b>
<a href="#p6"><i>14</i>The idea that reorganised everything</a>
<a href="#p6"><i>15</i>What it explains, retroactively</a>
<a href="#p6"><i>16</i>The right decomposition for the right question</a>
<b>Part VII &mdash; Two results</b>
<a href="#p7"><i>17</i>Read one way, steer another</a>
<a href="#p7"><i>18</i>Where a change lands, and why depth is a lever</a>
<b>Part VIII &mdash; What broke</b>
<a href="#p8"><i>19</i>The retraction</a>
<a href="#p8"><i>20</i>The full catalogue of what I got wrong</a>
<b>Part IX &mdash; Where it stands</b>
<a href="#p9"><i>21</i>What somebody else found</a>
<a href="#p9"><i>22</i>What is actually true at the end of this</a>
</div>
"""


def main():
    body = "".join([
        f'<div id="p1">{PART1}</div>', f'<div id="p2">{PART2}</div>',
        f'<div id="p3">{PART3}</div>', f'<div id="p4">{PART4}</div>',
        f'<div id="p5">{PART5}</div>', f'<div id="p6">{PART6}</div>',
        f'<div id="p7">{PART7}</div>', f'<div id="p8">{PART8}</div>',
        f'<div id="p9">{PART9}</div>'])
    for key, path in FIGS.items():
        p = OUT / path
        if p.exists():
            b = base64.b64encode(p.read_bytes()).decode()
            body = body.replace(key, f"data:image/png;base64,{b}")
        else:
            print(f"  [miss] {path}")

    parts = ["<title>Reading a Model's Wiring</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500;600&'
             'family=IBM+Plex+Sans:wght@400;500;600&'
             'family=IBM+Plex+Serif:wght@400;500;600&display=swap">',
             f"<style>{CSS}</style>", '<div class="wrap">',
             "<h1>Reading a Model's Wiring</h1>",
             '<p class="sub">What the Jacobian lens shows, what it hides, and '
             'everything that broke along the way.</p>',
             '<p class="byline">Written for a reader starting from basic maths and '
             'basic machine learning. Nothing below assumes prior interpretability '
             'knowledge. Roughly an hour end to end.</p>',
             TOC, body, "</div>",
             '<dialog id=lb><img id=lbi alt=""></dialog>',
             "<script>const lb=document.getElementById('lb'),"
             "i2=document.getElementById('lbi');"
             "document.querySelectorAll('figure img').forEach("
             "i=>i.onclick=()=>{i2.src=i.src;lb.showModal();});"
             "lb.onclick=()=>lb.close();</script>"]
    dest = OUT / "BOOK.html"
    dest.write_text("\n".join(parts))
    words = len(body.split())
    print(f"wrote {dest}  ({words:,} words, {dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
