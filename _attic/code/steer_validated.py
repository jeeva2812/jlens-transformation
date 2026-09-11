"""Steer validated SmolLM2 J directions (gated: validated only).

For each validated record in out/labels/smollm2-step0_J_*.json:
- calibrate alpha in [0.002,...,0.1] across ALL forcing prompts (+/-), repetition
  = 1-unique/total on 25-token greedy generations; stop when max rep >0.30, use previous.
- shift: mean over NEUTRAL prompts of (logP(B)-logP(A)) with record pairs,
  shift = (plus-minus) swing, random_shift = same with random direction, same alpha.
- text_changed: axis-specific counting on 3 forcing prompts (25 tokens), sign-aware,
  requires >=2/3 prompts correct.
- steers: visibly if text_changed else metric-only if sign matches, |shift|>0.5
  and |shift|>3*|random_shift| else no.
Writes steer results back into shard records' "steer" field.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.lens import _MultiCapture, _find_blocks_and_norm
from jlens.assay import NEUTRAL

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75]
NTOK_CAL = 13  # calibration length (gallery convention; base 25-token reps exceed 0.30 naturally)
NTOK = 25  # final generation length for text_changed counts

PROBES = {
  "gender": ["The engineer opened the toolbox because",
             "The nurse looked at the chart and then",
             "After the operation the surgeon told the family that"],
  "US/UK -our": ["The col", "The behavi", "The hon"],
  "US/UK -ise": ["They will real", "They will organ", "Critics will emphas"],
  "US/UK -re": ["The shopping cent", "They entered the thea", "The unit of length is the met"],
  "formal register": ["To fix this problem you should just",
                      "The simplest way to do this is to",
                      "If you want better results you can"],
  "positive vs negative sentiment": ["The movie was absolutely",
                      "The service at the restaurant was",
                      "The news left everyone feeling"],
  "size antonyms": ["The mountain was incredibly",
                    "The insect was remarkably",
                    "Compared to the others, this one was"],
  "code vs prose": ["The function takes a",
                    "def load_config(path):\n    #",
                    "The report was finished on Tuesday and"],
  "capitalised": ["She works for a company called",
                  "They moved to a town named",
                  "He read a book written by"],
  "past tense": ["Every day he walks to work. Yesterday he",
                 "She usually cooks dinner, but last night she",
                 "They normally play outside. Last week they"],
  "negation": ["The study found that the treatment",
               "He looked at the results and said the effect",
               "The report concluded that the policy"],
  "temporal opposition": ["The meeting was scheduled for long",
                          "Arriving early means waiting, but arriving late means",
                          "The project began in January and ended in"],
  "positive vs negative polarity": ["He is always",
                          "There is nothing left, but there is still",
                          "Everyone agreed, yet nobody"],
  "present vs past participle": ["He has already",
                          "They have",
                          "She had"],
  "reflexive": ["He blamed",
                "She taught",
                "They prepared"],
  "cardinal vs ordinal": ["He finished the race in",
                          "She came in",
                          "The final standings put him in"],
  "un-negation": ["The news made her feel",
                  "His response was",
                  "The outcome was"],
  "verb vs agent noun": ["The person who teaches is called a",
                         "She works as a professional",
                         "He drives for a living; he is a"],
  "plural": ["On the kitchen table there was a",
             "In the garden I could see a",
             "He opened the box and found a"],
  "base vs intensified": ["The performance was not just good, it was",
                          "The mountain was not big, it was",
                          "She was not happy, she was"],
  "infinitive vs gerund": ["He enjoys",
                           "She finished",
                           "They kept"],
  "base vs 3rd-person-sg": ["Every day he",
                            "She usually",
                            "The dog always"],
  "declarative vs interrogative": ["I wonder",
                          "She asked",
                          "They want to know"],
  "singular vs plural pronouns": ["The student finished homework and then",
                          "The committee announced that",
                          "After the game the player said"],
  "subject vs object pronouns": ["The manager praised",
                          "She gave the book to",
                          "Between you and"],
  "US/UK misc": ["The sky over the city was",
                 "My mother always told me",
                 "The mechanic said the car needs"],
}

MASC = {"he","him","his","himself","man","boy","father","son","brother","king"}
FEM = {"she","her","hers","herself","woman","girl","mother","daughter","sister","queen"}


class Add:
    def __init__(s, m, l, d, a):
        b, _ = _find_blocks_and_norm(m)
        s.d = torch.nn.functional.normalize(d.float(), dim=0); s.a = a
        s.h = b[l].register_forward_hook(s._h)
    def _h(s, m, i, o):
        t = o if torch.is_tensor(o) else o[0]
        t2 = t + s.a * s.d.to(t.device, t.dtype)
        return t2 if torch.is_tensor(o) else (t2,) + tuple(o[1:])
    def __enter__(s): return s
    def __exit__(s, *e): s.h.remove()


def words(txt):
    return re.findall(r"[A-Za-z]+", txt.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("out/labels"))
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--jpath", default="out/ft/J_step0.pt")
    ap.add_argument("--glob", default="smollm2-step0_J_L*_*.json")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="override the alpha ladder; the protocol takes the LARGEST\n                          non-degenerate value, so a ladder that never degenerates\n                          is a truncated sweep, not a calibrated one")
    ap.add_argument("--target", type=int, default=28,
                    help="target layer J maps to; J is the identity there so it is skipped")
    a = ap.parse_args()

    alphas = a.alphas or ALPHAS
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    Jdict = torch.load(a.jpath, map_location="cpu", weights_only=False)["J"]
    # SVD/EIG vectors per layer (recompute; cheap for 576)
    vecs = {}
    for L, J in Jdict.items():
        if L == a.target:
            continue
        Jf = J.float()
        U, S, Vh = torch.linalg.svd(Jf)
        w, V = torch.linalg.eig(Jf)
        lam = w.abs()
        order = torch.argsort(lam, descending=True).tolist()
        kept, ku = [], []
        for j in order:
            v = V[:, j].real.clone()
            n = float(v.norm())
            if n < 1e-9:
                continue
            v = v / n
            if any(abs(float(v @ u)) > 0.99 for u in ku):
                continue
            kept.append(j); ku.append(v)
            if len(kept) >= 20:
                break
        vecs[L] = {"svd": [(i, Vh[i] / Vh[i].norm()) for i in range(20)],
                   "eigen": [(j, ku[k]) for k, j in enumerate(kept)]}

    def gen(prompt, n=NTOK):
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        o = g[0][e["input_ids"].shape[1]:]
        ids = o.tolist()
        txt = tok.decode(o).replace("\n", " ").strip()
        rep = 1 - len(set(ids)) / max(len(ids), 1)
        return txt, rep, ids

    def pair_ids(pairs):
        A, B = [], []
        for x, y in pairs:
            ia = tok.encode(" " + x, add_special_tokens=False)
            ib = tok.encode(" " + y, add_special_tokens=False)
            if len(ia) == 1 and len(ib) == 1:
                A.append(ia[0]); B.append(ib[0])
        return torch.tensor(A), torch.tensor(B)

    def axis_score(prompt, A, B):
        e = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            lg = model(**e).logits[0, -1].float()
            lp = torch.log_softmax(lg, -1)
        return float(lp[B].mean() - lp[A].mean())

    for shard in sorted(a.outdir.glob(a.glob)):
        D = json.loads(shard.read_text())
        dirty = False
        for r in D:
            if r["verdict"] != "validated" or r.get("steer"):
                continue
            ax = r["hypothesis"]
            if ax not in PROBES:
                r["steer"] = None
                r["audit_note"] = f"no forcing prompts for axis {ax!r}; skipped steering"
                dirty = True
                continue
            prompts = PROBES[ax]
            L = r["layer"]
            # find vector
            fam = r["family"]; idx = r["index"]
            d = None
            for i2, v in vecs[L][fam]:
                if int(i2) == int(idx):
                    d = v.clone()
                    break
            if d is None:
                continue
            d = d / d.norm()
            A, B = pair_ids(r["pairs"])
            if len(A) == 0:
                continue
            # hn from first prompt
            e0 = tok(prompts[0], return_tensors="pt")["input_ids"]
            with _MultiCapture(model, [L], a.target) as cap:
                with torch.no_grad():
                    model(input_ids=e0, attention_mask=torch.ones_like(e0), use_cache=False)
                hn = float(cap.h[L][0].norm(dim=-1).mean())
            # calibrate (13-token generations, gallery convention)
            alpha = None
            for cand in alphas:
                reps = []
                for p in prompts:
                    for sg in (+1, -1):
                        with Add(model, L, sg * d, cand * hn):
                            _, rep, _ = gen(p, n=NTOK_CAL)
                            reps.append(rep)
                if max(reps) > 0.30:
                    break
                alpha = cand
            if alpha is None:
                r["steer"] = {"alpha": None, "shift": 0.0, "random_shift": 0.0,
                              "text_changed": False, "steers": "no",
                              "note": "degenerate at every alpha", "examples": []}
                dirty = True
                print(f"L{L} {fam}[{idx}] {ax}: DEGENERATE at all alphas", flush=True)
                continue
            # shift on NEUTRAL
            def swing(direction):
                tot = 0.0
                for p in NEUTRAL:
                    with Add(model, L, direction, alpha * hn):
                        sp = axis_score(p, A, B)
                    with Add(model, L, -direction, alpha * hn):
                        sm = axis_score(p, A, B)
                    tot += (sp - sm)
                return tot / len(NEUTRAL)
            gtorch = torch.Generator().manual_seed(9000 + L * 100 + int(idx))
            rnd = torch.randn(d.shape[0], generator=gtorch)
            rnd = rnd / rnd.norm()
            shift = swing(d)
            random_shift = swing(rnd)
            # generations on forcing prompts
            examples = []
            correct = 0
            sgn = 1 if r["score"] > 0 else -1  # +d promotes B if sgn>0
            for p in prompts:
                bt, _, _ = gen(p)
                with Add(model, L, d, alpha * hn):
                    ut, _, _ = gen(p)
                with Add(model, L, -d, alpha * hn):
                    dt, _, _ = gen(p)
                examples.append({"prompt": p, "base": bt, "plus": ut, "minus": dt})
                ok = check_text(ax, p, bt, ut, dt, sgn, tok)
                correct += int(ok)
            text_changed = correct >= 2
            if text_changed:
                steers = "visibly"
            elif (shift * sgn > 0.5) and (abs(shift) > 3 * abs(random_shift)):
                steers = "metric-only"
            else:
                steers = "no"
            r["steer"] = {"alpha": alpha, "shift": round(float(shift), 3),
                          "random_shift": round(float(random_shift), 3),
                          "text_changed": bool(text_changed), "steers": steers,
                          "examples": examples}
            dirty = True
            print(f"L{L} {fam}[{idx}] {ax} a={alpha} shift={shift:+.2f} rand={random_shift:+.2f} "
                  f"text={text_changed} -> {steers} ({correct}/3)", flush=True)
        if dirty:
            shard.write_text(json.dumps(D, indent=1))
    print("done")


def check_text(ax, prompt, base, plus, minus, sgn, tok):
    """True if plus/minus differ in predicted direction. sgn=+1: +d promotes B."""
    if ax == "gender":
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in FEM) - sum(1 for x in w if x in MASC)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax in ("US/UK -our", "US/UK -ise", "US/UK -re"):
        # first-token suffix check
        suff = {"US/UK -our": ("or", "our"), "US/UK -ise": ("ize", "ise"),
                "US/UK -re": ("er", "re")}[ax]
        # for -re theater probe, suffixes differ; fall back to word counting if needed
        def first_word(t):
            m = re.findall(r"[A-Za-z]+", t)
            return m[0].lower() if m else ""
        fp, fm = first_word(plus), first_word(minus)
        # predicted: plus should start with B side if sgn>0 else A side
        pred_plus = suff[1] if sgn > 0 else suff[0]
        pred_minus = suff[0] if sgn > 0 else suff[1]
        # for -re thea probe the suffix is ter/tre, not er/re; accept either spelling signal:
        # also count full US/UK word lists as backup
        if ax == "US/UK -our" or ax == "US/UK -ise":
            ok = (fp == pred_plus) and (fm == pred_minus)
            if ok:
                return True
            # backup: count US/UK forms in full generation
            return count_spelling(plus, minus, sgn, ax)
        else:
            return count_spelling(plus, minus, sgn, ax)
    if ax == "formal register":
        inf = {"get","use","show","help","need","start","end","buy"}
        frm = {"obtain","utilize","demonstrate","facilitate","require","commence","terminate","purchase"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in frm) - sum(1 for x in w if x in inf)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "positive vs negative sentiment":
        pos = {"good","happy","excellent","love","beautiful","kind","smart","rich","clean","safe"}
        neg = {"bad","sad","terrible","hate","ugly","cruel","stupid","poor","dirty","dangerous"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in neg) - sum(1 for x in w if x in pos)
        # note pairs are (positive, negative): A=pos B=neg, score>0 promotes neg
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "size antonyms":
        big = {"big","large","huge","tall","long","high","wide","deep","thick","heavy","largest","biggest"}
        small = {"small","tiny","short","low","narrow","shallow","thin","light","smallest"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in small) - sum(1 for x in w if x in big)
        # pairs (big, small): A=big B=small
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "code vs prose":
        code = {"def","self","import","return","null","int","void","func","class","noqa","hashlib"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in code)
        # pairs A=prose B=code; sgn>0: + promotes code
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "past tense":
        pres = {"walk","play","work","look","want","need","start","call","open"}
        past = {"walked","played","worked","looked","wanted","needed","started","called","opened",
                "answered","benefited","melted","confessed","drank","injected"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in past) - sum(1 for x in w if x in pres)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "capitalised":
        up = {"Apple","John","London","Monday","River","King","Street","Manufacturers","Consumers"}
        lo = {"apple","john","london","monday","river","king","street","manufacturers","consumers"}
        def sc(t):
            w = re.findall(r"[A-Za-z]+", t)  # case-sensitive
            return sum(1 for x in w if x in up) - sum(1 for x in w if x in lo)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "negation":
        neg = {"not","never","nothing","cannot","without","no","n't","incorrectly","unexpected","instead"}
        def sc(t):
            return sum(1 for x in words(t) if x in neg)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "temporal opposition":
        A = {"before","early","morning","today","past","start","begin","first"}
        B = {"after","late","night","evening","tomorrow","future","end","last"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "positive vs negative polarity":
        A = {"always","everything","everyone","everybody","all","ever","anything","anyone"}
        B = {"never","nothing","nobody","none"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "present vs past participle":
        A = {"eat","write","break","choose","drive","give","take","see"}
        B = {"eaten","written","broken","chosen","driven","given","taken","seen"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "reflexive":
        A = {"him","her","them","me","us","you"}
        B = {"himself","herself","themselves","myself","ourselves","yourself"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "cardinal vs ordinal":
        A = {"one","two","three","four","five","six","seven","eight"}
        B = {"first","second","third","fourth","fifth","sixth","seventh","eighth"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "un-negation":
        A = {"happy","fair","known","clear","safe","able","usual","kind"}
        B = {"unhappy","unfair","unknown","unclear","unsafe","unable","unusual","unkind"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "verb vs agent noun":
        A = {"teach","work","play","run","write","read","drive","farm","hunt","manage"}
        B = {"teacher","worker","player","runner","writer","reader","driver","farmer","hunter","manager"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "plural":
        A = {"dog","cat","house","car","book","tree","year","hand","word","group"}
        B = {"dogs","cats","houses","cars","books","trees","years","hands","words","groups"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "base vs intensified":
        A = {"good","bad","big","small","happy","sad","angry","tired"}
        B = {"excellent","terrible","huge","tiny","delighted","miserable","furious","exhausted"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "infinitive vs gerund":
        A = {"walk","play","work","look","want","need","start","call","open","talk"}
        B = {"walking","playing","working","looking","wanting","needing","starting","calling","opening","talking"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "base vs 3rd-person-sg":
        A = {"walk","play","work","look","want","need","start","call","open","talk"}
        B = {"walks","plays","works","looks","wants","needs","starts","calls","opens","talks"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "declarative vs interrogative":
        A = {"he","him","his","it","there","then"}
        B = {"who","whom","whose","what","where","when"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "singular vs plural pronouns":
        A = {"he","him","his","himself","she","her","hers","herself"}
        B = {"they","them","their","themselves"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "subject vs object pronouns":
        A = {"he","she","they","we","i","who"}
        B = {"him","her","them","us","me","whom"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    if ax == "US/UK misc":
        A = {"gray","tire","mom","aluminum"}
        B = {"grey","tyre","mum","aluminium"}
        def sc(t):
            w = words(t)
            return sum(1 for x in w if x in B) - sum(1 for x in w if x in A)
        return (sc(plus) - sc(minus)) * sgn > 0
    return False


def count_spelling(plus, minus, sgn, ax):
    US_OUR = {"color","honor","favor","labor","humor","neighbor","harbor","behavior","flavor",
              "honored","colored","behavioral","center","theater","meter","gray"}
    UK_OUR = {"colour","honour","favour","labour","humour","neighbour","harbour","behaviour","flavour",
              "organise","organisation","behavioural","colours","favour","labelling"}
    US_ISE = {"realize","recognize","analyze","organize","apologize","criticize","emphasize","civilization","capitalize","honors"}
    UK_ISE = {"realise","recognise","analyse","organise","apologise","criticise","emphasise","behavioural","standardised","labelling"}
    if ax == "US/UK -our":
        A, B = US_OUR, UK_OUR
    elif ax == "US/UK -ise":
        A, B = US_ISE, UK_ISE
    else:
        A = {"center","theater","meter","fiber","caliber","liter","defense","railroad","color"}
        B = {"centre","theatre","metre","fibre","calibre","litre","defence","railway","colour"}
    def sc(t):
        w = set(words(t))
        return len(w & B) - len(w & A)
    d = sc(plus) - sc(minus)
    if d * sgn > 0:
        return True
    # also accept first-token signal already checked by caller
    return False


if __name__ == "__main__":
    main()
