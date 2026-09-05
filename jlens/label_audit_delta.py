"""Audit ΔJ raw validations with a token-coherence gate.

For each validated record, check the winning side's readout (in/out) for
axis-relevant markers (substring lists). If absent -> downgrade to no-hypothesis.
Also marks top-SVD code-coherent-but-null-failing directions as rejected
(linter signal: noqa/pylint on one pole, prose on the other).
"""
from __future__ import annotations
import json
from pathlib import Path
import glob

SPELL = ["olour","onor","avou","abou","ehavio","neighbo","harbo","flavo",
 "organise","organize","realise","realize","recognis","recogniz","analys",
 "apologis","criticis","emphas","centre","center","theatre","theater",
 "metre","meter","fibre","fiber","calibre","caliber","litre","liter",
 "grey","gray","tyre","tire","cheque","pyjama","pajama","aluminium","aluminum",
 "whilst","labell","modell","paediatr","gynaec","haemat","leukaem","odour",
 "tumo","behavio","favo","labo","hono","colo","minimis","minimiz","maximis",
 "maximiz","fertilis","fertiliz","practis","practic","standardis","standardiz",
 "characteris","characteriz","defen","keit","diarro","encyclopae"]
GENDER = [" he"," she"," him"," her"," his"," hers"," they"," them"," their",
 " man"," woman"," boy"," girl"," father"," mother"," son"," daughter",
 " brother"," sister"," king"," queen"," himself"," herself"]
CODE = ["noqa","pylint","kwargs","ndarray","filepath","<filename>","metainfo",
 "intf","flake","eslint","import","return","def ","self","null","void","func","class"]
CAP = ["Apple","John","London","Monday","Manufacturers","Consumers"]
SENT = ["coward","parano","betray","sabot","deceiv","stupid","nasty","rotten",
 "poison","worthless","parent","growth","famil"]
SIZE = ["largest","smallest","biggest","huge","tiny","sizes"]
PAST = ["walked","played","worked","looked","wanted","needed","started","called",
 "opened","answered","benefited","melted","confessed","drank"," was "]
PLUR = ["dogs","cats","houses","cars","books","product","products"]
FORMAL_INF = ["guys","gonna","yeah","kids","stuff","poop"]
NEG = ["not","never","nothing","cannot","without"]

def has_any(toks, markers):
    s = " ".join(t.lower() for t in toks)
    return any(m.lower() in s for m in markers)

def markers_for(hyp):
    h = hyp.lower()
    if "us/uk" in h:
        return SPELL
    if "gender" in h or "pronoun" in h or "person" in h or "reflexive" in h or "possessive" in h or "subject vs object" in h:
        return GENDER
    if "code" in h:
        return CODE
    if "capitalised" in h:
        return CAP
    if "sentiment" in h:
        return SENT
    if "size" in h or "comparative" in h or "superlative" in h:
        return SIZE
    if "past" in h or "participle" in h or "gerund" in h or "3rd-person" in h or "plural" in h or "tense" in h:
        return PAST + PLUR
    if "formal" in h:
        return FORMAL_INF
    if "negation" in h or "polarity" in h or "question" in h or "interrogative" in h or "definite" in h or "modality" in h or "adverb" in h or "agent noun" in h or "temporal" in h or "spatial" in h or "un-negation" in h or "ordinal" in h or "mass" in h:
        return None  # skip auto-check; manual
    return None

def main():
    files = sorted(glob.glob("out/labels/smollm2-delta_dJ_L*_*.json"))
    n_keep = n_down = n_manual = 0
    for f in files:
        D = json.loads(Path(f).read_text())
        dirty = False
        for r in D:
            if r["verdict"] != "validated":
                continue
            hyp = r["hypothesis"]
            side = "in" if "(in)" in hyp else "out"
            toks = r["tokens_in_pos"] + r["tokens_in_neg"] if side == "in" else r["tokens_pos"] + r["tokens_neg"]
            mk = markers_for(hyp)
            if mk is None:
                r["audit_note"] = f"manual review needed for {hyp}; kept pending"
                n_manual += 1
                continue
            if has_any(toks, mk):
                n_keep += 1
            else:
                r["verdict"] = "no-hypothesis"; r["hypothesis"] = None
                r["pairs"] = []
                r["audit_note"] = f"auto-downgrade: winning side ({side}) shows no markers for {hyp}"
                n_down += 1
                dirty = False
            # note: dirty handling below
        # rewrite always to be safe
        Path(f).write_text(json.dumps(D, indent=1))
    print(f"keep={n_keep} downgraded={n_down} manual={n_manual}")
    # mark linter SVD0s as rejected where code-coherent but null-failing
    n_rej = 0
    for f in files:
        D = json.loads(Path(f).read_text())
        for r in D:
            if r["family"] != "svd" or r["index"] != 0:
                continue
            toks = r["tokens_pos"] + r["tokens_neg"] + r.get("tokens_in_pos", []) + r.get("tokens_in_neg", [])
            s = " ".join(t.lower() for t in toks)
            if ("noqa" in s or "pylint" in s) and r["verdict"] == "no-hypothesis":
                r["verdict"] = "rejected"
                r["hypothesis"] = "code vs prose"
                r["pairs"] = [["the", "def"], ["and", "self"], ["with", "import"],
                              ["from", "return"], ["was", "null"], ["said", "void"],
                              ["very", "func"], ["about", "class"]]
                r["audit_note"] = ("top-dJ direction is code-coherent (noqa/pylint vs prose) "
                                   "but code axis failed the 99.9th null (ratios ~0.5-0.99); "
                                   "rejected pending a linter-specific axis")
                n_rej += 1
        Path(f).write_text(json.dumps(D, indent=1))
    print(f"marked rejected (linter): {n_rej}")
    oq = Path("out/OPEN_QUESTIONS.md")
    oq.write_text((oq.read_text() if oq.exists() else "") +
        ("\n# ΔJ audit\n- 206/560 raw validated, mostly (out) spelling with punctuation readouts; "
         "auto-downgraded those lacking axis markers (conservative: no-hypothesis).\n"
         "- Top-dJ SVD0 per layer reads noqa/pylint (linter) vs prose but code axis fails null; "
         "marked rejected. Q: build a linter-specific axis (noqa/pylint-centred pairs)? "
         "Conservative: did not invent weak pairs; left rejected.\n"))

if __name__ == "__main__":
    main()
