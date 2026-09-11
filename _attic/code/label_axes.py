"""Extended axis battery for systematic labelling.

Keeps the ten hand-written axes from jlens.axes (imported, not copied) and adds
~25 invented axes covering morphological / syntactic / lexical oppositions that
are clean minimal pairs (differ only in the named idea) and usually single
tokens with a leading space.

Each axis maps name -> list[(a, b)] where score = mean logit(b) - mean logit(a).
Positive score means the direction promotes the B side.
"""
from __future__ import annotations
import torch
from jlens.axes import AXES as BASE_AXES

INVENTED_AXES = {
  # --- verb morphology ---
  "infinitive vs gerund": [("walk", "walking"), ("play", "playing"),
    ("work", "working"), ("look", "looking"), ("want", "wanting"),
    ("need", "needing"), ("start", "starting"), ("call", "calling"),
    ("open", "opening"), ("talk", "talking")],
  "base vs 3rd-person-sg": [("walk", "walks"), ("play", "plays"),
    ("work", "works"), ("look", "looks"), ("want", "wants"),
    ("need", "needs"), ("start", "starts"), ("call", "calls"),
    ("open", "opens"), ("talk", "talks")],
  "present vs past participle": [("eat", "eaten"), ("write", "written"),
    ("break", "broken"), ("choose", "chosen"), ("drive", "driven"),
    ("give", "given"), ("take", "taken"), ("see", "seen")],
  # --- adjective / adverb ---
  "comparative": [("big", "bigger"), ("small", "smaller"),
    ("high", "higher"), ("low", "lower"), ("fast", "faster"),
    ("slow", "slower"), ("great", "greater"), ("strong", "stronger"),
    ("weak", "weaker"), ("long", "longer")],
  "superlative": [("big", "biggest"), ("small", "smallest"),
    ("high", "highest"), ("low", "lowest"), ("fast", "fastest"),
    ("slow", "slowest"), ("great", "greatest"), ("long", "longest")],
  "adjective vs adverb": [("quick", "quickly"), ("slow", "slowly"),
    ("happy", "happily"), ("sad", "sadly"), ("clear", "clearly"),
    ("loud", "loudly"), ("soft", "softly"), ("bright", "brightly")],
  "base vs intensified": [("good", "excellent"), ("bad", "terrible"),
    ("big", "huge"), ("small", "tiny"), ("happy", "delighted"),
    ("sad", "miserable"), ("angry", "furious"), ("tired", "exhausted")],
  # --- nominalisation ---
  "verb vs agent noun": [("teach", "teacher"), ("work", "worker"),
    ("play", "player"), ("run", "runner"), ("write", "writer"),
    ("read", "reader"), ("drive", "driver"), ("farm", "farmer"),
    ("hunt", "hunter"), ("manage", "manager")],
  # --- pronouns / person ---
  "subject vs object pronouns": [("he", "him"), ("she", "her"),
    ("they", "them"), ("we", "us"), ("I", "me"), ("who", "whom")],
  "possessive": [("I", "my"), ("he", "his"), ("she", "her"),
    ("they", "their"), ("we", "our"), ("you", "your"), ("it", "its")],
  "first vs third person": [("I", "he"), ("me", "him"), ("we", "they"),
    ("us", "them"), ("my", "his"), ("our", "their"), ("am", "is"),
    ("have", "has")],
  "singular vs plural pronouns": [("he", "they"), ("him", "them"),
    ("his", "their"), ("himself", "themselves"), ("she", "they")],
  "reflexive": [("him", "himself"), ("her", "herself"),
    ("them", "themselves"), ("me", "myself"), ("us", "ourselves"),
    ("you", "yourself")],
  # --- deixis / space / time ---
  "proximal vs distal": [("this", "that"), ("these", "those"),
    ("here", "there"), ("now", "then")],
  "spatial opposition": [("in", "out"), ("up", "down"),
    ("above", "below"), ("over", "under"), ("left", "right"),
    ("north", "south"), ("east", "west"), ("inside", "outside"),
    ("forward", "backward"), ("front", "back")],
  "temporal opposition": [("before", "after"), ("early", "late"),
    ("morning", "night"), ("today", "tomorrow"), ("past", "future"),
    ("start", "end"), ("begin", "end"), ("first", "last")],
  # --- modality / polarity ---
  "weak vs strong modality": [("might", "must"), ("could", "should"),
    ("may", "must"), ("can", "should"), ("could", "will"),
    ("might", "will"), ("may", "should"), ("can", "will")],
  "present vs remote modals": [("will", "would"), ("can", "could"),
    ("shall", "should"), ("may", "might")],
  "positive vs negative polarity": [("always", "never"),
    ("everything", "nothing"), ("everyone", "nobody"),
    ("everybody", "nobody"), ("all", "none"), ("ever", "never"),
    ("anything", "nothing"), ("anyone", "nobody")],
  # --- lexis ---
  "positive vs negative sentiment": [("good", "bad"), ("happy", "sad"),
    ("excellent", "terrible"), ("love", "hate"), ("beautiful", "ugly"),
    ("kind", "cruel"), ("smart", "stupid"), ("rich", "poor"),
    ("clean", "dirty"), ("safe", "dangerous")],
  "size antonyms": [("big", "small"), ("large", "small"),
    ("huge", "tiny"), ("tall", "short"), ("long", "short"),
    ("high", "low"), ("wide", "narrow"), ("deep", "shallow"),
    ("thick", "thin"), ("heavy", "light")],
  "un-negation": [("happy", "unhappy"), ("fair", "unfair"),
    ("known", "unknown"), ("clear", "unclear"), ("safe", "unsafe"),
    ("able", "unable"), ("usual", "unusual"), ("kind", "unkind")],
  # --- numbers ---
  "number word vs digit": [("one", "1"), ("two", "2"), ("three", "3"),
    ("four", "4"), ("five", "5"), ("six", "6"), ("seven", "7"),
    ("eight", "8"), ("nine", "9"), ("ten", "10")],
  "cardinal vs ordinal": [("one", "first"), ("two", "second"),
    ("three", "third"), ("four", "fourth"), ("five", "fifth"),
    ("six", "sixth"), ("seven", "seventh"), ("eight", "eighth")],
  "count vs mass quantifiers": [("many", "much"), ("few", "little"),
    ("fewer", "less"), ("number", "amount")],
  # --- orthography ---
  "US/UK -re": [("center", "centre"), ("theater", "theatre"),
    ("meter", "metre"), ("fiber", "fibre"), ("caliber", "calibre"),
    ("liter", "litre")],
  "US/UK misc": [("gray", "grey"), ("tire", "tyre"),
    ("mom", "mum"), ("aluminum", "aluminium")],
  # --- determiners / pro-forms ---
  "indefinite vs definite": [("a", "the"), ("an", "the"),
    ("some", "the"), ("any", "the")],
  "declarative vs interrogative": [("he", "who"), ("him", "whom"),
    ("his", "whose"), ("it", "what"), ("there", "where"),
    ("then", "when")],
  # --- punctuation (single tokens, no leading space needed) ---
  "period vs comma": [(".", ","), ("!", "?"), (";", ":")],
}

ALL_AXES = {**BASE_AXES, **INVENTED_AXES}


def build_extended(tok, min_pairs=4):
    """Keep only pairs where both sides are a single token with leading space.

    Punctuation axis is special: tokens without leading space are allowed.
    Returns {name: (A ids tensor, B ids tensor, kept pairs list)}.
    """
    out = {}
    for name, pairs in ALL_AXES.items():
        A, B, kept = [], [], []
        for a, b in pairs:
            if name == "period vs comma":
                ia = tok.encode(a, add_special_tokens=False)
                ib = tok.encode(b, add_special_tokens=False)
                # allow with or without leading space forms
                if len(ia) == 1 and len(ib) == 1:
                    A.append(ia[0]); B.append(ib[0]); kept.append((a, b))
                else:
                    ia2 = tok.encode(" " + a, add_special_tokens=False)
                    ib2 = tok.encode(" " + b, add_special_tokens=False)
                    if len(ia2) == 1 and len(ib2) == 1:
                        A.append(ia2[0]); B.append(ib2[0]); kept.append((a, b))
                continue
            ia = tok.encode(" " + a, add_special_tokens=False)
            ib = tok.encode(" " + b, add_special_tokens=False)
            if len(ia) == 1 and len(ib) == 1:
                A.append(ia[0]); B.append(ib[0]); kept.append((a, b))
        if len(A) >= min_pairs:
            out[name] = (torch.tensor(A), torch.tensor(B), kept)
    return out


def held_out_filter(axis_name, kept_pairs, readout_tokens, tok=None):
    """Drop pairs where either side appears in the direction's readout.

    readout_tokens: list of decoded token strings (top15 pos+neg).
    Comparison is on stripped, case-insensitive word form.
    """
    readout_set = {t.strip().lower().lstrip("ġ▁ ") for t in readout_tokens}
    out = []
    for a, b in kept_pairs:
        if a.strip().lower() in readout_set or b.strip().lower() in readout_set:
            continue
        out.append((a, b))
    return out
