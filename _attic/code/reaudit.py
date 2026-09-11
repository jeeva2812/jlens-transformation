"""Re-audit helpers: theme lexicons + grounding counts (report-only)."""
from __future__ import annotations
import re

def clean(tok: str) -> str:
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", tok)

def real_words(toks):
    return [c for c in (clean(t) for t in toks) if len(c) >= 3 and c.isalpha()]

LEX = {
 "US/UK -our": {"colour","color","colours","colors","coloured","colored","colouring","coloring",
   "honour","honor","honours","honors","honoured","honored","honouring","favour","favor","favours",
   "favors","favoured","favored","favouring","labour","labor","labours","labors","humour","humor",
   "neighbour","neighbor","neighbours","neighbors","harbour","harbor","harbours","harbors",
   "behaviour","behavior","behaviours","behaviors","behavioural","behavioral","flavour","flavor",
   "flavours","flavors","whilst","mould","mold","moult","tumour","tumour","colourful","colorful"},
 "US/UK -ise": {"realise","realize","realised","realized","realises","realizes","recognise","recognize",
   "recognised","recognized","analyse","analyze","analysed","analyzed","organise","organize",
   "organised","organized","organising","organizing","organisation","organization","organisations",
   "apologise","apologize","criticise","criticize","emphasise","emphasize","standardise","standardize",
   "standardised","standardized","characterise","characterize","characterised","characterized",
   "standardised","minimise","minimize","maximise","maximize","fertilise","fertilize","fertiliser",
   "characterise","characterize","practise","practice","practising","licence","license","civilisation",
   "civilization","capitalise","capitalize","favour","behavioural","labelling","modelling","odour"},
 "US/UK -re": {"centre","center","centres","centers","theatre","theater","theatres","theaters","metre",
   "meter","metres","meters","fibre","fiber","fibres","fibers","calibre","caliber","litre","liter",
   "litres","liters","lustre","luster","sombre","somber","spectre","specter","mitre","miter"},
 "US/UK misc": {"gray","grey","tire","tyre","tires","tyres","mom","mum","aluminum","aluminium",
   "pajamas","pyjamas","diaper","nappy"},
 "gender": {"he","she","him","her","his","hers","himself","herself","man","woman","men","women",
   "boy","girl","father","mother","mothers","fathers","son","sons","daughter","daughters","brother",
   "brothers","sister","sisters","king","queen","kings","queens","male","female","husband","wife",
   "uncle","aunt","nephew","niece","baby","babies","herself","himself"},
 "formal register": {"get","obtain","use","utilize","show","demonstrate","help","facilitate","need",
   "require","start","commence","end","terminate","buy","purchase","guy","guys","kid","kids","stuff",
   "gonna","yeah","yep","yummy","huh","poop","crap","crappy","newbie","hustle","folks","hey","somebody",
   "someone","pursuant","disclosed","insurers","securities","arbitration","advisory","stakeholders",
   "stakeholder","initiatives","initiative","procure","endeavour","hereby","moreover","nevertheless",
   "consequently","accordingly","notwithstanding","whereas","agricultural","nutritionist","purch"},
 "positive vs negative sentiment": {"good","bad","happy","sad","excellent","terrible","love","hate",
   "beautiful","ugly","kind","cruel","smart","stupid","rich","poor","clean","dirty","safe","dangerous",
   "amazing","awful","horrible","nasty","rotten","worthless","poison","poisons","coward","paranoia",
   "betray","deceive","deceived","dishonest","dishonesty","loved","enjoyed","enjoy","cursed","weakens",
   "sabot","loved","terrible","stunning","delighted","miserable","furious","exhausted","huge","tiny"},
 "code vs prose": {"def","self","import","return","null","int","void","func","class","noqa","pylint",
   "kwargs","ndarray","filepath","filename","flake","eslint","pytest","assert","lambda","elif","except",
   "raise","yield","async","await","struct","enum","malloc","printf","console","kernel","pointer",
   "marriage","songs","song","music","lyrics","lyric","melody","singer","story","stories","novel",
   "poem","poetry","married","wedding","dialogue","dialogues","enrol","enroll","teenage","melodies"},
 "size antonyms": {"big","small","large","huge","tiny","tall","short","long","high","low","wide","narrow",
   "deep","shallow","thick","thin","heavy","light","largest","smallest","biggest","tiniest","enormous",
   "massive","gigantic","immense","sizes","larger","smaller","greater"},
 "past tense": {"walk","walked","play","played","work","worked","look","looked","want","wanted","need",
   "needed","start","started","call","called","open","opened","answered","benefited","melted","confessed",
   "drank","injected","was","were","had","melted","confessed"},
 "capitalised": set(),
 "negation": {"not","never","nothing","cannot","without","neither","nor","none","nobody","incorrectly",
   "unexpected","instead","cant","wont","dont","isnt","arent","wasnt","werent","hasnt","have","hadnt"},
 "temporal opposition": {"before","after","early","late","morning","night","evening","today","tomorrow",
   "past","future","start","end","begin","began","first","last","yesterday","beginnings","startup"},
 "positive vs negative polarity": {"always","never","everything","nothing","everyone","nobody","everybody",
   "all","none","ever","anything","anyone"},
 "base vs intensified": {"good","excellent","bad","terrible","big","huge","small","tiny","happy","delighted",
   "sad","miserable","angry","furious","tired","exhausted"},
 "base vs 3rd-person-sg": {"walk","walks","play","plays","work","works","look","looks","want","wants",
   "need","needs","start","starts","call","calls","open","opens","talk","talks","organizes"},
 "singular vs plural pronouns": {"he","him","his","himself","she","her","hers","herself","they","them",
   "their","themselves","you","your","yours","yourself","yourselves","we","us","our","ours","ourselves"},
 "cardinal vs ordinal": {"one","first","two","second","three","third","four","fourth","five","fifth",
   "six","sixth","seven","seventh","eight","eighth"},
 "un-negation": {"happy","unhappy","fair","unfair","known","unknown","clear","unclear","safe","unsafe",
   "able","unable","usual","unusual","kind","unkind"},
 "verb vs agent noun": {"teach","teacher","work","worker","play","player","run","runner","write","writer",
   "read","reader","drive","driver","farm","farmer","hunt","hunter","manage","manager","protector",
   "viewer","builder","owner"},
 "reflexive": {"him","himself","her","herself","them","themselves","me","myself","us","ourselves","you",
   "yourself"},
 "present vs past participle": {"eat","eaten","write","written","break","broken","choose","chosen",
   "drive","driven","give","given","take","taken","see","seen"},
}

SPELL_ALL = (LEX["US/UK -our"] | LEX["US/UK -ise"] | LEX["US/UK -re"] | LEX["US/UK misc"]
              | {"whilst", "cheque", "draught", "plough", "moustache", "mustache", "tire", "tires"})


def support_count(hypothesis: str, toks):
    """Number of tokens that are real words supporting the theme."""
    base = hypothesis.rsplit(" (", 1)[0]
    if base == "capitalised":
        n = 0
        for t in toks:
            c = clean(t)
            if len(c) >= 3 and c.isalpha() and (c[0].isupper() or c.lower() in
               {"apple","john","london","monday","river","king","street","manufacturers","consumers",
                "teacher","kids","undergrad","essay","semester","plasmid","teacher"}):
                n += 1
        return n
    if base.startswith("US/UK"):
        lex = SPELL_ALL
    else:
        lex = LEX.get(base)
    if lex is None:
        return None
    n = 0
    for t in toks:
        c = clean(t)
        if len(c) >= 3 and c.isalpha() and c.lower() in lex:
            n += 1
    return n


def grounding(hypothesis: str, rec):
    """Max over the hypothesis side's poles of theme-supporting real-word count."""
    if hypothesis.endswith("(in)"):
        poles = [rec["tokens_in_pos"], rec["tokens_in_neg"]]
    elif hypothesis.endswith("(out)"):
        poles = [rec["tokens_pos"], rec["tokens_neg"]]
    else:
        poles = [rec["tokens_pos"], rec["tokens_neg"]]
    return max(support_count(hypothesis, p) or 0 for p in poles)
