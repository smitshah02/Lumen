"""
Deterministic query classification
==================================
Triage used to be an unconditional LLM call on the critical path: one FAST
generation before any retrieval had started, to pick one of five categories.
For the overwhelming majority of clinical questions the category is decidable
from the wording alone, and paying a model round trip to learn that a question
containing "what is the most recent creatinine" is a lab lookup is pure latency.

So classification runs in two stages:

    classify(query) -> Decision
        confident=True   -> use it, no model call at all
        confident=False  -> the caller falls back to the FAST triage prompt

Only the ambiguous tail reaches a model. Nothing here decides an answer; it
decides which *path* runs, and every path still grounds and cites the same way.

Two outputs:
  query_type  the existing QueryType routing category
  complexity  "simple"  -> one fact, one place to look (FAST synthesis is enough)
              "complex" -> across time, across admissions, or conflicting
                           evidence (MAIN synthesis)
Complexity is deliberately biased toward "complex": a simple question answered
by the big model is slow, a complex question answered by the small one is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# --- category signals ------------------------------------------------------
# Order matters: the first rule that fires wins, mirroring the precedence the
# LLM triage prompt spells out (recommendation > measurement > chart lookup).

_GUIDELINE = [
    r"\bshould (?:we|i|they|the patient|he|she)\b", r"\bis .{0,30}indicated\b",
    r"\brecommend(?:ed|ation|ations)?\b", r"\bguidelines?\b", r"\bstandard of care\b",
    r"\bappropriate (?:to|for)\b", r"\bwhat dose should\b", r"\bcontraindicat",
]
_LITERATURE = [
    r"\b(?:published|literature|studies|study|trials?|evidence base|pubmed|meta-?analys)",
    r"\bwhat does the (?:research|evidence|literature)\b",
]
_UNSUPPORTED = [
    r"\bwhat(?:'s| is) the prognosis\b", r"\bhow long (?:do|does) .{0,25}(?:have to )?live\b",
    r"\bwill (?:he|she|they|the patient) (?:die|survive|recover)\b",
    r"\bdiagnose\b", r"\bwhat should i (?:eat|do about my)\b",
]
# A measured quantity: either an explicit "value/level/result" phrasing, or one
# of the analytes the structured lab resolver already knows how to fetch.
_LAB_PHRASE = [
    r"\b(?:lab|labs|laboratory) (?:value|values|result|results|work)\b",
    r"\b(?:value|values|level|levels|reading|readings|result|results)\b",
    r"\bhow (?:high|low) (?:was|is|did)\b",
]

_COMPLEX = [
    r"\btrend(?:ing|ed|s)?\b", r"\bover time\b", r"\bprogress(?:ion|ed|ing)\b",
    r"\bcompare[ds]?\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bbetween (?:the )?(?:two|both|his|her|their) admissions?\b",
    r"\bacross (?:admissions|visits|encounters|time)\b",
    r"\bchang(?:e|ed|es|ing)\b", r"\bevolution\b", r"\bserial\b",
    r"\bsummar(?:y|ise|ize)\b", r"\bhistory of\b", r"\bwhy\b", r"\bexplain\b",
    r"\bconflict(?:ing)?\b", r"\bdiscrepan", r"\binconsistent\b",
    r"\bimprov(?:e|ed|ing)\b", r"\bworsen(?:ed|ing)?\b", r"\bdeteriorat",
    r"\ball (?:of )?(?:his|her|their|the) (?:admissions|visits|encounters)\b",
    r"\beach admission\b", r"\bevery admission\b",
    # anchoring one event against another is a comparison, however short the
    # sentence: "when was X removed relative to the Y admission"
    r"\brelative to\b", r"\bcompared (?:to|with)\b", r"\bbefore or after\b",
    # a recommendation-shaped negative ("which drug should NOT be used") needs
    # the reason, which usually lives in a different note than the fact
    r"\bshould not\b", r"\bavoid(?:ed)?\b", r"\bfuture\b",
    # allergy status is a reconciliation across the allergy list, the narrative
    # and what the patient actually tolerated — the small model flattens it
    r"\ballerg",
]
_SIMPLE = [
    r"\bwhat (?:was|is) the (?:most recent|latest|last|current)\b",
    # the adverb forms matter: detect_temporal_mode classes "most recently" as
    # "recent", not "latest", so the phrase has to be recognised here too
    r"\bmost recent(?:ly)?\b", r"\blatest\b", r"\bcurrent(?:ly)?\b", r"\brecently\b",
    r"\bis (?:he|she|the patient|they) (?:on|taking)\b",
    # bare "which"/"who" were too permissive: "which antibiotic should not be
    # used" is a reasoning question, not a lookup. Complexity defaults to
    # complex when nothing simple fires, which is the safe direction.
    r"\bwhen (?:was|did)\b", r"\bwho\b",
]

# Questions that ask for more than one thing at once are never "simple", however
# simple each half looks: "most recent creatinine and is it improving?".
_MULTIPART = [r"\band (?:was|is|did|does|how|why|what|has|have)\b", r"\?.+\?", r";"]

_MED = [r"\bmedication", r"\bmeds\b", r"\bdrugs?\b", r"\bprescri", r"\bdischarged on\b",
        r"\btaking\b", r"\bdose\b", r"\bdosage\b", r"\bstart(?:ed|ing)?\b", r"\bstopp?(?:ed)?\b",
        r"\bswitch(?:ed)?\b", r"\bheld\b", r"\brestart", r"\banticoagul", r"\ballerg",
        # drug-class words, so "which antibiotic..." is recognised as a
        # medication question rather than falling through to the model
        r"\bantibiotics?\b", r"\bantiviral", r"\bantifungal", r"\binsulin\b", r"\bstatins?\b",
        r"\bdiuretics?\b", r"\binhalers?\b", r"\bsteroids?\b", r"\bopioids?\b",
        r"\bbiologics?\b", r"\bchemotherapy\b", r"\bvaccin"]

# "What happened to this patient" vocabulary. These only decide chart_review,
# which is also the fallback category — so the risk of a wrong rule here is
# confidence, not a different destination. Guideline, literature and refusal
# signals are all tested BEFORE this, and still win.
_CHART = [
    r"\b(?:admission|admitted|readmi|discharge|hospitali[sz])",
    r"\b(?:diagnos|comorbid|problem list|condition)",
    r"\b(?:x-?ray|radiograph|imaging|ct scan|\bct\b|mri|ultrasound|echocardiogram|scan|biopsy|"
    r"endoscopy|colonoscopy|bronchoscopy|angiogra)",
    r"\b(?:surgery|surgical|operation|procedure|resection|repair|amputation|transplant|graft)",
    r"\b(?:oxygen|ventilat|intubat|transfus|dialysis|infusion|culture)",
    r"\b(?:treated|treatment|managed|management|therapy)\b",
    r"\b(?:did|was|were|has|have|had|does|is) (?:he|she|they|the patient)\b",
    r"\bwhat happened\b", r"\bhow much\b", r"\bhow many\b",
    r"\b(?:his|her|their|the patient'?s?) (?:history|record|chart|course|stay)\b",
]

# Analyte words the structured lab path understands. Kept in sync by test:
# every key of lab_query.SYNONYMS must be matchable here.
_ANALYTE = r"|".join([
    "potassium", "sodium", "chloride", "bicarbonate", "magnesium", "calcium", "phosphate",
    "phosphorus", "glucose", "blood sugar", "creatinine", "bun", "urea", "kidney function",
    "renal function", "hemoglobin", "haemoglobin", "hematocrit", "platelets?", "white blood cells?",
    "wbc", "red blood cells?", "rbc", "blood count", "cbc", "anemia", "inr", "ptt", "lactate",
    "albumin", "bilirubin", "alt", "ast", "alkaline phosphatase", "liver function", "lft",
    "troponin", "lipase", "cholesterol", "a1c", "hba1c", "neutrophil",
])
_ANALYTE_RE = re.compile(rf"(?<![a-z])(?:{_ANALYTE})(?![a-z])", re.I)


def _any(patterns, q: str) -> bool:
    return any(re.search(p, q) for p in patterns)


@dataclass(frozen=True)
class Decision:
    query_type: str        # chart_review | guideline_check | lab_trend | literature | unsupported
    complexity: str        # simple | complex
    confident: bool        # False -> caller should ask the FAST triage model
    reason: str            # which rule fired; safe to log (no query text)

    def as_dict(self) -> dict:
        return asdict(self)


def classify(query: str, temporal_mode: str = "all") -> Decision:
    """Classify without a model call. `temporal_mode` comes from the retriever's
    existing detect_temporal_mode(), so the two stay consistent by construction."""
    q = (query or "").strip().lower()
    if not q:
        return Decision("chart_review", "complex", False, "empty")

    complex_hit = _any(_COMPLEX, q) or temporal_mode == "trend" or len(q.split()) > 22
    multipart = _any(_MULTIPART, q)
    simple_hit = _any(_SIMPLE, q) or temporal_mode == "latest"
    complexity = "complex" if (complex_hit or multipart or not simple_hit) else "simple"

    # Category, in the precedence the triage prompt defines.
    if _any(_UNSUPPORTED, q):
        # Refusal is a safety decision; never make it on a keyword alone.
        return Decision("unsupported", complexity, False, "unsupported_signal")
    if _any(_GUIDELINE, q):
        return Decision("guideline_check", complexity, True, "guideline_phrase")
    if _any(_LITERATURE, q):
        return Decision("literature", complexity, True, "literature_phrase")
    if _ANALYTE_RE.search(q) or _any(_LAB_PHRASE, q):
        return Decision("lab_trend", complexity, True, "analyte_or_value_phrase")
    if _any(_MED, q) or temporal_mode in ("latest", "trend", "recent"):
        return Decision("chart_review", complexity, True, "medication_or_temporal")
    if _any(_SIMPLE, q):
        return Decision("chart_review", complexity, True, "simple_lookup_phrase")
    if _any(_CHART, q):
        return Decision("chart_review", complexity, True, "chart_event_phrase")

    # Nothing recognisable fired. Hand it to the model rather than guessing.
    return Decision("chart_review", complexity, False, "no_rule_matched")


def wants_deterministic_lab(d: Decision, temporal_mode: str, subject_id) -> bool:
    """Is this a question the structured `labevents` path can answer outright?

    Every condition has to hold: a confident single-analyte latest-value lookup
    for one patient. Anything broader (a trend, a comparison, a second clause,
    an uncertain classification) goes through normal retrieval and synthesis."""
    return (subject_id is not None
            and d.confident
            and d.query_type == "lab_trend"
            and d.complexity == "simple"
            and temporal_mode == "latest")
