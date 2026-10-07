"""Deterministic admission resolver (data-foundation plan, E3 / decision R2 step 1).

Works out which admission a question names, using fixed rules only:

    an explicit admission id    the request field, or "hadm_id 22595853" in the text
    an explicit date            "the admission that ended on 2180-05-07",
                                "admitted on 2180-05-06", "discharged on 2180-05-07"
    first / last / most recent  "during her last admission"
    nth admission               "the second admission", "her 3rd hospitalization"
    previous admission          only relative to an anchor: one other explicit
                                reference in the same question
                                ("the admission before her last admission")

A request-level hadm_id always wins over anything in the question text.

It never chooses between candidates. Zero matches is `unresolved`, several is
`ambiguous`, and either leaves the question untouched so retrieval stays
patient-wide. Only a resolved reference is removed from the retrieval text.

A date counts as an admission reference only when it is attached to an
admission word; "the creatinine on 2180-05-06" is a clinical date and is left
alone. The model-assisted step for descriptive references is E10.

ponytail: "the admission before the one that ended on <date>" (a pronoun
anchor) and "next admission" are not recognised; add when a question set needs them.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Optional

import sqlalchemy as sa

_PREP = r"(?:(?:during|in|at|for|from|on|of)\s+)?"
_DET = r"(?:(?:the\s+patient's|the|her|his|their|this|that)\s+)?"
_NOUN = r"(?:admission|hospitali[sz]ation|hospital\s+stay|stay|encounter)"
_DATE = r"\d{4}-\d{2}-\d{2}"
_END_VERBS = ("ended", "ending", "ends")

_ID_RE = re.compile(
    rf"{_PREP}{_DET}(?:hadm(?:[ _]?id)?|admission(?:\s+id)?|encounter(?:\s+id)?)\s*[#:]?\s*(?P<id>\d{{6,9}})\b", re.I)
_DATE_AFTER_RE = re.compile(
    rf"{_PREP}{_DET}(?P<noun>{_NOUN}|discharge)\s+(?:that\s+)?"
    rf"(?:(?P<verb>ended|ending|ends|started|starting|began|beginning)\s+)?(?:(?:on|of|dated|from)\s+)?(?P<date>{_DATE})\b", re.I)
_DATE_BEFORE_RE = re.compile(rf"{_PREP}{_DET}(?P<date>{_DATE})\s+(?P<noun>{_NOUN}|discharge)\b", re.I)
_ORDINAL_RE = re.compile(
    rf"{_PREP}{_DET}(?P<ord>last|latest|most[- ]recent|first|earliest)\s+{_NOUN}\b", re.I)
# "admitted on <date>": the verb is part of the question, so only " on <date>" is removed.
_VERB_DATE_RE = re.compile(rf"\b(?P<verb>admitted|discharged)(?P<tail>\s+on\s+(?P<date>{_DATE}))\b", re.I)
_NTH = {w: n for n, w in enumerate("second third fourth fifth sixth seventh eighth ninth tenth".split(), 2)}
_NTH_RE = re.compile(
    rf"{_PREP}{_DET}(?P<nth>{'|'.join(_NTH)}|\d{{1,2}}(?:st|nd|rd|th))\s+{_NOUN}\b", re.I)
_PREVIOUS_RE = re.compile(
    rf"{_PREP}(?:{_DET}(?:previous|prior|preceding)\s+{_NOUN}"
    # "the admission before ...": a determiner is required, so "on admission before the stent" is not a reference
    rf"|(?:the\s+patient's|the|her|his|their|this|that)\s+{_NOUN}\s+(?:before|prior\s+to|preceding))\b", re.I)
_PATTERNS = (("previous", _PREVIOUS_RE), ("hadm_id", _ID_RE), ("date", _DATE_AFTER_RE), ("date", _DATE_BEFORE_RE),
             ("verb_date", _VERB_DATE_RE), ("ordinal", _ORDINAL_RE), ("nth", _NTH_RE))


@dataclass(frozen=True)
class AdmissionResolution:
    status: str                  # "resolved" | "unresolved" | "ambiguous" | "none" (no admission named)
    retrieval_query: str         # the question, minus a resolved admission phrase
    hadm_id: Optional[int] = None
    source: Optional[str] = None  # "request" | "rule:hadm_id" | "rule:date" | "rule:first" | "rule:last"
    reason: Optional[str] = None  # why it is unresolved or ambiguous
    phrase: Optional[str] = None  # the admission reference found in the question

    def as_state(self) -> dict:
        return asdict(self)


def load_admissions(subject_id: int) -> list[tuple]:
    """(hadm_id, admittime, dischtime) for one patient, oldest first."""
    from src.storage import engine
    with engine.connect() as c:
        return [tuple(r) for r in c.execute(sa.text(
            "SELECT hadm_id, admittime, dischtime FROM admissions "
            "WHERE subject_id = :sid AND admittime IS NOT NULL ORDER BY admittime, hadm_id"), {"sid": subject_id})]


# The stay window used to scope unlinked notes (decision R1): ED registration
# when there is one, else the admit time, through the discharge time.
STAY_WINDOW_SQL = ("SELECT COALESCE(edregtime, admittime) AS stay_start, dischtime AS stay_end "
                   "FROM admissions WHERE hadm_id = :hadm_id")


def load_stay_window(hadm_id: int) -> Optional[tuple]:
    """(start, end) of one admission, or None when either bound is unknown."""
    from src.storage import engine
    with engine.connect() as c:
        row = c.execute(sa.text(STAY_WINDOW_SQL), {"hadm_id": hadm_id}).first()
    return (row[0], row[1]) if row and row[0] is not None and row[1] is not None else None


def _day(value) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _strip(query: str, spans: list[tuple[int, int]]) -> str:
    """Remove the phrases and tidy the joins. Keeps the question whole when too
    little would be left to search with ("When was her last admission?")."""
    rest = query
    for start, end in sorted(spans, reverse=True):
        rest = rest[:start] + " " + rest[end:]
    rest = re.sub(r"\s+", " ", re.sub(r"^[,;:\s]+", "", rest.strip()))
    rest = re.sub(r"\s+([?.!,;:])", r"\1", rest)
    return rest if len(re.findall(r"[A-Za-z0-9]+", rest)) >= 3 else query.strip()


def _by_date(admissions: list[tuple], day: date, noun: str, verb: Optional[str]) -> list[int]:
    ends = noun == "discharge" or (verb or "") in _END_VERBS
    starts = bool(verb) and not ends
    hits = []
    for hadm_id, admit, disch in admissions:
        a, d = _day(admit), _day(disch)
        if ends:
            match = d == day
        elif starts:
            match = a == day
        else:                      # "the admission on <date>": any day of the stay
            match = a is not None and a <= day <= (d or a)
        if match:
            hits.append(int(hadm_id))
    return hits


def _tied(admissions: list[tuple], i: int) -> bool:
    """Position i is not well defined when a neighbour shares its admit time."""
    t = admissions[i][1]
    return any(0 <= k < len(admissions) and admissions[k][1] == t for k in (i - 1, i + 1))


def _hits(kind: str, m: re.Match, admissions: list[tuple], known: set[int]) -> tuple[str, list[int], str]:
    """(source, matching hadm_ids, reason when there are none) for one reference."""
    if kind == "hadm_id":
        return "rule:hadm_id", [h for h in (int(m.group("id")),) if h in known], \
            f"hadm_id {m.group('id')} is not an admission of this patient"
    if kind in ("date", "verb_date"):
        day = _day(m.group("date"))
        noun, verb = (("discharge", None) if m.group("verb").lower() == "discharged" else ("admission", "started")) \
            if kind == "verb_date" else (m.group("noun").lower(), (m.groupdict().get("verb") or "").lower())
        return "rule:date", _by_date(admissions, day, noun, verb) if day else [], f"no admission matches {m.group('date')}"
    if kind == "nth":
        word = m.group("nth").lower()
        n = _NTH.get(word) or int(re.match(r"\d+", word).group())
        if not 1 <= n <= len(admissions):
            return "rule:ordinal", [], f"this patient has {len(admissions)} admissions on record, not {n}"
        if _tied(admissions, n - 1):
            t = admissions[n - 1][1]
            return "rule:ordinal", [int(a[0]) for a in admissions if a[1] == t], ""
        return "rule:ordinal", [int(admissions[n - 1][0])], ""
    last = m.group("ord").lower() not in ("first", "earliest")
    edge = (max if last else min)((a[1] for a in admissions), default=None)
    return ("rule:last" if last else "rule:first"), [int(a[0]) for a in admissions if a[1] == edge], \
        "this patient has no admissions on record"


def _span(kind: str, m: re.Match) -> tuple[int, int]:
    return (m.start("tail"), m.end()) if kind == "verb_date" else (m.start(), m.end())


def resolve_admission(query: str, admissions: list[tuple],
                      request_hadm_id: Optional[int] = None) -> AdmissionResolution:
    """Resolve the admission a question names against `admissions`, a list of
    (hadm_id, admittime, dischtime) ordered oldest first."""
    query = query or ""
    q = query.strip()
    known = {int(a[0]) for a in admissions}
    if request_hadm_id is not None and int(request_hadm_id) not in known:
        return AdmissionResolution("unresolved", q, source="request",
                                   reason=f"hadm_id {request_hadm_id} is not an admission of this patient")

    refs = []                                            # (start, end, kind, match)
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(query):
            if not any(m.start() < e and s < m.end() for s, e, _, _ in refs):
                refs.append((m.start(), m.end(), kind, m))
    refs.sort()
    previous = [r for r in refs if r[2] == "previous"]
    others = [r for r in refs if r[2] != "previous"]
    named = "; ".join(query[s:e].strip() for s, e, _, _ in refs) or None

    if request_hadm_id is not None:                      # the caller stated it; this wins over the text
        return AdmissionResolution("resolved", q, int(request_hadm_id), "request")
    if previous:
        return _resolve_previous(query, admissions, known, previous, others, named)
    if not refs:
        return AdmissionResolution("none", q)
    if len(refs) > 1:
        return AdmissionResolution("ambiguous", q, reason="the question names more than one admission", phrase=named)

    _, _, kind, m = refs[0]
    source, hits, missing = _hits(kind, m, admissions, known)
    if len(hits) == 1:
        return AdmissionResolution("resolved", _strip(query, [_span(kind, m)]), hits[0], source, phrase=named)
    if not hits:
        return AdmissionResolution("unresolved", q, source=source, reason=missing, phrase=named)
    return AdmissionResolution("ambiguous", q, source=source, phrase=named,
                               reason=f"{len(hits)} admissions match: {sorted(hits)}")


def _resolve_previous(query, admissions, known, previous, others, named) -> AdmissionResolution:
    """"The previous admission" means nothing without an anchor, and the only
    anchor is one other explicit reference in the same question."""
    q, source = query.strip(), "rule:previous"

    def no(status: str, reason: str) -> AdmissionResolution:
        return AdmissionResolution(status, q, source=source, reason=reason, phrase=named)

    if len(previous) > 1 or len(others) > 1:
        return no("ambiguous", "the question names more than one admission")
    spans = [(previous[0][0], previous[0][1])]
    if not others:
        return no("unresolved", "'previous admission' has no anchor admission to be previous to")
    _, _, kind, m = others[0]
    _, hits, missing = _hits(kind, m, admissions, known)
    if len(hits) != 1:
        return no("ambiguous" if hits else "unresolved",
                  f"the anchor admission is not unique: {sorted(hits)}" if hits else f"the anchor admission is unknown: {missing}")
    anchor = hits[0]
    spans.append(_span(kind, m))

    i = next(k for k, a in enumerate(admissions) if int(a[0]) == anchor)
    if i == 0:
        return no("unresolved", f"no admission precedes hadm_id {anchor}")
    if _tied(admissions, i) or (i >= 2 and admissions[i - 1][1] == admissions[i - 2][1]):
        return no("ambiguous", f"admissions around hadm_id {anchor} share an admit time")
    return AdmissionResolution("resolved", _strip(query, spans), int(admissions[i - 1][0]), source, phrase=named)


# ===========================================================================
# Model-assisted step for descriptive references (plan E10 / decisions R2, R2b)
# ===========================================================================
# "the stay when she had the stent" names an admission the rules above cannot
# resolve. Here a local model may pick ONE admission, but only from a list of
# the patient's own admissions and only by quoting evidence from that list. Its
# choice is applied only when code can verify it: the id is a candidate, the
# quoted evidence really is in that admission's evidence, that evidence is about
# what the question describes, and the same evidence is in no other admission's.
# The model's confidence is logged and decides nothing. Anything else is
# unresolved or ambiguous, never a guess.

MAX_CANDIDATES = 12           # more admissions than this are not compared: unresolved, not truncated
LINES_SHOWN = 25              # evidence lines per admission shown to the model; the check uses all of them
MIN_EVIDENCE_CHARS = 5
_DESCRIPTIVE_RE = re.compile(
    rf"{_PREP}(?:the\s+patient's|the|her|his|their|this|that)\s+{_NOUN}\s+"
    r"(?:when|where|in\s+which|during\s+which|for\s+which|for|with)\b[^?.!;]*", re.I)


def descriptive_reference(query: str) -> Optional[re.Match]:
    """A phrase that describes an admission by what happened in it, or None."""
    return _DESCRIPTIVE_RE.search(query or "")


def _norm(value: str) -> str:
    return " ".join(str(value or "").lower().split())


# Words that say nothing about WHICH admission: function words, and the generic
# clinical nouns and verbs every admission shares. Overlap on these never counts.
_GENERIC = frozenset("""
    the a an and or of in on at to for from with by as her his their she he they this that these those it its
    when where which while during after before had has have having was were is are been being did does do done
    admission admissions admitted hospital hospitalization hospitalisation hospitalized stay stays encounter visit
    patient patients procedure procedures procedural surgery surgeries surgical operation operations operative
    treatment treatments treated therapy diagnosis diagnoses diagnostic diagnosed note placed performed underwent
    received given started got made taken removed new first last time other unspecified left right bilateral
    not but nor yet all any can may one two who whom whose how why what out off per due see now too via non pre
    you your our own could would should will shall might must cannot there then than also into onto over under
    about above below between through within without upon some each every such only very more most much many
    same just still again once here both either neither because since until
    elsewhere classified specified mention mentioned site type part parts initial subsequent sequela
""".split())


def _singular(word: str) -> str:
    """Plain plural to singular: "stents" -> "stent", "biopsies" -> "biopsy".
    Words in -ss, -us and -is are left alone ("abscess", "thrombus", "dialysis")."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_terms(text: str) -> set[str]:
    """The distinctive words of `text`, normalised for comparison: lower case,
    punctuation dropped, generic and function words dropped, plain plurals made
    singular ("stents" -> "stent", "biopsies" -> "biopsy"), then each word cut
    to its first six letters so close inflections agree ("endoscopy" /
    "endoscopic"). Three-letter words count only as whole words ("EEG", "MRI");
    shorter ones are ignored. Nothing here bridges a prefix: "dialysis" and
    "hemodialysis" stay different, which costs a match and never invents one."""
    words = [_singular(w) for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if w not in _GENERIC]
    words = [w for w in words if w not in _GENERIC]
    return {w[:6] for w in words if len(w) >= 4} | {w for w in words if len(w) == 3 and w.isalpha()}


def evidence_is_relevant(evidence: str, phrase: str) -> bool:
    """Does the quoted evidence mention what the descriptive phrase describes?
    A deterministic overlap of distinctive terms; no model, no confidence."""
    return bool(content_terms(evidence) & content_terms(phrase))


def load_candidates(subject_id: int) -> list[dict]:
    """The patient's admissions, oldest first, each with its evidence lines:
    coded diagnosis and procedure titles, and the chief complaint, procedure and
    discharge-diagnosis lines of its discharge summary."""
    from src.retrieval.section_labels import parse_sections
    from src.storage import engine, readiness
    readiness.require(readiness.STRUCTURED)
    params = {"sid": subject_id}
    with engine.connect() as c:
        candidates = {int(r[0]): {"hadm_id": int(r[0]), "admitted": str(r[1])[:10], "discharged": str(r[2])[:10], "evidence": []}
                      for r in c.execute(sa.text(
                          "SELECT hadm_id, admittime, dischtime FROM admissions WHERE subject_id = :sid "
                          "AND admittime IS NOT NULL ORDER BY admittime, hadm_id"), params)}
        for kind, table, titles in (("diagnosis", "diagnoses_icd", "d_icd_diagnoses"),
                                    ("procedure", "procedures_icd", "d_icd_procedures")):
            for hadm_id, title in c.execute(sa.text(
                    f"SELECT f.hadm_id, t.long_title FROM {table} f JOIN {titles} t "       # table names are the literals above
                    "ON t.icd_code = trim(f.icd_code) AND t.icd_version = f.icd_version "
                    "WHERE f.subject_id = :sid ORDER BY f.hadm_id, f.seq_num"), params):
                if hadm_id in candidates and title:
                    candidates[int(hadm_id)]["evidence"].append(f"{kind}: {title}")
        for hadm_id, note in c.execute(sa.text(
                "SELECT hadm_id, COALESCE(text_original, text_deid) FROM clinical_notes "
                "WHERE subject_id = :sid AND note_type = 'discharge' ORDER BY hadm_id, note_id"), params):
            if hadm_id not in candidates:
                continue
            for section in parse_sections(note, "discharge"):
                if section.name in ("Chief Complaint", "Major Surgical or Invasive Procedure", "Discharge Diagnosis"):
                    for line in section.text.split("\n")[1:]:
                        if len(line.strip()) >= MIN_EVIDENCE_CHARS:
                            candidates[int(hadm_id)]["evidence"].append(f"note: {line.strip()[:160]}")
    for cand in candidates.values():
        cand["evidence"] = list(dict.fromkeys(cand["evidence"]))
    return list(candidates.values())


def candidate_prompt(question: str, candidates: list[dict]) -> str:
    blocks = []
    for c in candidates:
        lines = "\n".join(f"- {line}" for line in c["evidence"][:LINES_SHOWN]) or "- (no evidence on record)"
        blocks.append(f"ADMISSION {c['hadm_id']} | admitted {c['admitted']} | discharged {c['discharged']}\n{lines}")
    return f"QUESTION: {question}\n\n" + "\n\n".join(blocks)


def accept_model_choice(output: dict, candidates: list[dict], phrase: Optional[str] = None
                        ) -> tuple[Optional[int], str, Optional[str]]:
    """(hadm_id, status, reason) for the model's structured output. Resolved only
    when all four hold: the id is a candidate; the quoted evidence is in that
    admission's evidence; the evidence shares a distinctive term with `phrase`,
    the description that triggered this step; and the evidence is in no other
    candidate's. Confidence is not consulted."""
    by_id = {c["hadm_id"]: c for c in candidates}
    try:
        hadm_id = int(output.get("hadm_id")) if output.get("hadm_id") is not None else None
    except (TypeError, ValueError, AttributeError):
        return None, "unresolved", "the model did not return a usable admission id"
    if hadm_id is None:
        return None, "unresolved", "the model selected no admission"
    if hadm_id not in by_id:
        return None, "unresolved", f"the model returned hadm_id {hadm_id}, which is not one of this patient's admissions"
    quote = _norm(output.get("evidence"))
    if len(quote) < MIN_EVIDENCE_CHARS:
        return None, "unresolved", "the model cited no evidence for its choice"

    def has(candidate: dict) -> bool:
        return any(quote in _norm(line) for line in candidate["evidence"])

    if not has(by_id[hadm_id]):
        return None, "unresolved", "the evidence the model cited is not in that admission's record"
    if not evidence_is_relevant(quote, phrase or ""):
        return None, "unresolved", "the evidence the model cited does not mention what the question describes"
    others = sorted(c["hadm_id"] for c in candidates if c["hadm_id"] != hadm_id and has(c))
    if others:
        return None, "ambiguous", f"the cited evidence also appears in admission(s) {others}"
    return hadm_id, "resolved", None


def resolve_with_model(query: str, candidates: list[dict], ask) -> AdmissionResolution:
    """Resolve a descriptive admission reference. `ask(prompt)` returns the
    model's JSON text; any failure of it is unresolved, never a guess."""
    import json
    import logging
    q = query.strip()
    m = descriptive_reference(query)
    phrase = m.group().strip() if m else None

    def no(status: str, reason: str) -> AdmissionResolution:
        return AdmissionResolution(status, q, source="model", reason=reason, phrase=phrase)

    if not candidates:
        return no("unresolved", "this patient has no admissions on record")
    if len(candidates) > MAX_CANDIDATES:
        return no("unresolved", f"this patient has {len(candidates)} admissions; more than {MAX_CANDIDATES} are not compared")
    try:
        output = json.loads(ask(candidate_prompt(query, candidates)))
        if not isinstance(output, dict):
            raise ValueError("not an object")
    except Exception as exc:                     # model down, timeout, malformed JSON: all the same outcome
        return no("unresolved", f"the admission model did not return a usable answer ({type(exc).__name__})")
    hadm_id, status, reason = accept_model_choice(output, candidates, phrase)
    logging.getLogger(__name__).info(
        "[admission_model] status=%s hadm_id=%s model_status=%s model_confidence=%s reason=%s",
        status, hadm_id, output.get("status"), output.get("confidence"), reason)
    if status != "resolved":
        return no(status, reason)
    return AdmissionResolution("resolved", _strip(query, [(m.start(), m.end())]) if m else q, hadm_id, "model", phrase=phrase)
