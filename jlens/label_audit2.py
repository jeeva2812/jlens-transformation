"""Audit patch for new SmolLM2 J layers (0,2,6,10,14,18,22,26). See prior audit for rationale."""
from __future__ import annotations
import json
from pathlib import Path

DOWNGRADE = {
    (0, "svd", 5): "spurious: fragments/punct, no adj/adv",
    (0, "svd", 6): "spurious: no negation contrast; neg is code tokens",
    (0, "svd", 8): "spurious: fragmentary, no clear sentiment words",
    (0, "eigen", 5): "mislabeled: neg shows UK spellings, not adj/adv",
    (0, "eigen", 17): "mislabeled: both poles show US/UK spellings, not adj/adv",
    (0, "eigen", 57): "spurious: no temporal words",
    (0, "eigen", 53): "spurious: no verb morphology",
    (2, "svd", 4): "mislabeled: poles show UK vs US spellings, not articles",
    (2, "svd", 6): "spurious: quotes/punct only",
    (2, "eigen", 44): "spurious: no temporal contrast",
    (2, "eigen", 45): "mislabeled: pos shows noun number (product/products), no pronouns",
    (2, "eigen", 37): "spurious: no size words",
    (2, "eigen", 54): "spurious: single weak word, no systematic temporal contrast",
    (6, "svd", 4): "spurious: punct vs fragments, no sentiment",
    (6, "svd", 5): "spurious: no articles",
    (6, "svd", 8): "spurious: quotes/punct, no verb agreement",
    (10, "eigen", 5): "mislabeled: pos is UK spellings, not size",
    (10, "eigen", 48): "spurious: no comparatives",
    (14, "svd", 2): "spurious: punct vs adverbs, no polarity",
    (14, "svd", 8): "spurious: punct/adjectives, no tense verbs",
    (14, "eigen", 41): "spurious: no interrogative pro-forms",
    (18, "eigen", 28): "spurious: single comparative among noise",
    (18, "eigen", 17): "spurious: no plural morphology",
    (18, "eigen", 34): "mislabeled: dominated by UK spellings + was; not tense",
    (22, "svd", 3): "spurious/weak: sports words are not gender minimal pairs",
    (22, "eigen", 80): "spurious: revenues, no size contrast",
    (26, "svd", 3): "spurious: punct only, no spelling",
    (26, "eigen", 16): "weak: pos incoherent; downgraded",
}

def main():
    base = Path("out/labels")
    n = 0
    for (L, fam, idx), reason in DOWNGRADE.items():
        p = base / f"smollm2-step0_J_L{L}_{fam}.json"
        D = json.loads(p.read_text())
        for r in D:
            if r["index"] == idx and r["verdict"] == "validated":
                r["verdict"] = "no-hypothesis"; r["hypothesis"] = None
                r["pairs"] = []; r["audit_note"] = reason; n += 1
                break
        p.write_text(json.dumps(D, indent=1))
    print(f"downgraded {n}/{len(DOWNGRADE)}")
    oq = Path("out/OPEN_QUESTIONS.md")
    lines = ["\n# Audit round 2 (layers 0,2,6,10,14,18,22,26)\n"]
    for k, v in sorted(DOWNGRADE.items()):
        lines.append(f"- L{k[0]} {k[1]}[{k[2]}]: {v}\n")
    oq.write_text((oq.read_text() if oq.exists() else "") + "".join(lines))

if __name__ == "__main__":
    main()
