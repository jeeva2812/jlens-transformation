"""Conservative audit of SmolLM2 J raw validations.

Downgrades directions where the readout tokens do not support the winning axis
(spurious null-beating, expected ~1-2/layer at 37 axes) or where tokens show a
different coherence than the winning axis (e.g. L24 pronoun directions
mislabelled as spelling). Downgraded -> verdict no-hypothesis, hypothesis None.
Logs reasons to out/OPEN_QUESTIONS.md fragment.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

DOWNGRADE = {
    # (layer, family, index): reason
    (4, "svd", 4): "spurious: top tokens are fragments/punctuation, no reflexive pronouns; borderline ratio 1.01 pct 0.33, expected false positive",
    (4, "eigen", 21): "spurious: tokens are noise fragments, no verb morphology for 3sg claim",
    (4, "eigen", 24): "ambiguous: pos informal chatter, neg contains SHALL but no systematic weak/strong modal contrast; downgraded pending cleaner modal axis",
    (4, "eigen", 46): "spurious: tokens show no size contrast",
    (8, "svd", 1): "spurious: no modal verbs in readout",
    (8, "svd", 3): "spurious: readout is fragments/punctuation, no articles",
    (8, "svd", 10): "spurious: readout is punctuation/fragments, no verb agreement contrast",
    (8, "eigen", 13): "spurious: pos is code fragments, neg is UK spellings; no participle morphology",
    (16, "eigen", 43): "spurious: single 3sg verb among noise, no systematic contrast",
    (16, "eigen", 58): "spurious: pos shows future/time words, not past tense; neg is fragments",
    (16, "svd", 5): "spurious: no capitalisation contrast in top tokens; borderline pct 0.33",
    (20, "eigen", 27): "spurious: no size words; neg shows frequency adverbs; borderline pct 0.33",
    (24, "svd", 1): "spurious: readout is quotes/punctuation, no spelling words",
    (24, "svd", 17): "mislabeled: readout is clearly 2nd-person pronouns (your/yourselves), not spelling; needs new 2nd-vs-3rd person axis, pending",
    (24, "svd", 19): "ambiguous: pos is mixed 3rd-person pronouns (both genders), neg is UK spellings; pronoun-vs-spelling entanglement, pending disambiguation",
    (24, "eigen", 4): "mislabeled: pos is music/pronouns, no spelling; spurious spelling score",
    (24, "eigen", 21): "ambiguous: pos mixes pronouns + US spellings (honored), neg UK; pronoun/spelling entanglement",
    (24, "eigen", 22): "ambiguous: pos feminine pronouns, neg UK fragments; likely pronoun direction mis-scored as spelling",
    (24, "eigen", 27): "ambiguous: pos pronouns+US (honored/theaters), neg UK; entanglement, pending",
    (24, "eigen", 29): "ambiguous: pos pronouns, neg weak UK fragments; pending cleaner pronoun axis",
    (24, "eigen", 34): "spurious/mislabeled: pos deaths/fatalities, neg colours single UK word; no systematic spelling contrast",
}

def main():
    base = Path("out/labels")
    n_down = 0
    for (layer, fam, idx), reason in DOWNGRADE.items():
        p = base / f"smollm2-step0_J_L{layer}_{fam}.json"
        D = json.loads(p.read_text())
        for r in D:
            if r["index"] == idx and r["verdict"] == "validated":
                r["verdict"] = "no-hypothesis"
                r["hypothesis"] = None
                r["pairs"] = []
                r["audit_note"] = reason
                # keep score/threshold/pct as closest-attempt info
                n_down += 1
                break
        p.write_text(json.dumps(D, indent=1))
    print(f"downgraded {n_down}/{len(DOWNGRADE)}")
    # append to OPEN_QUESTIONS
    oq = Path("out/OPEN_QUESTIONS.md")
    lines = ["# Open questions (SmolLM2 J audit)\n"]
    for k, v in sorted(DOWNGRADE.items()):
        lines.append(f"- L{k[0]} {k[1]}[{k[2]}]: {v}\n")
    lines.append("\nConservative option taken: downgraded to no-hypothesis, no steering. "
                 "A new pronoun/person axis battery (2nd-vs-3rd, pronouns-vs-nouns) "
                 "should be built and these re-tested before any steering.\n")
    prev = oq.read_text() if oq.exists() else ""
    oq.write_text(prev + "\n" + "".join(lines))
    print("wrote out/OPEN_QUESTIONS.md")

if __name__ == "__main__":
    main()
