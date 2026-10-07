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
alone. The model-assisted step for descriptive references is E10, and applying
the scope to retrieval is E4.

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
