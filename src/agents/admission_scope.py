"""Deterministic admission resolver (data-foundation plan, E3 / decision R2 step 1).

Works out which admission a question names, using fixed rules only:

    an explicit admission id   the request field, or "hadm_id 22595853" in the text
    an explicit date           "the admission that ended on 2180-05-07"
    first / last / most recent "during her last admission"

It never chooses between candidates. Zero matches is `unresolved`, several is
`ambiguous`, and either leaves the question untouched so retrieval stays
patient-wide. Only a resolved reference is removed from the retrieval text.

A date counts as an admission reference only when it is attached to an
admission word; "the creatinine on 2180-05-06" is a clinical date and is left
alone. The model-assisted step for descriptive references is E10, and applying
the scope to retrieval is E4.

ponytail: "admitted on <date>" / "discharged on <date>" and nth-admission
("second admission") are not recognised; add when a question set needs them.
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


def _strip(query: str, start: int, end: int) -> str:
    """Remove one phrase and tidy the join. Keeps the question whole when too
    little would be left to search with ("When was her last admission?")."""
    rest = (query[:start] + " " + query[end:]).strip()
    rest = re.sub(r"\s+", " ", re.sub(r"^[,;:\s]+", "", rest))
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


def resolve_admission(query: str, admissions: list[tuple],
                      request_hadm_id: Optional[int] = None) -> AdmissionResolution:
    """Resolve the admission a question names against `admissions`, a list of
    (hadm_id, admittime, dischtime) ordered oldest first."""
    query = query or ""
    known = {int(a[0]) for a in admissions}

    if request_hadm_id is not None:                      # the caller stated it; this wins over the text
        if int(request_hadm_id) in known:
            return AdmissionResolution("resolved", query.strip(), int(request_hadm_id), "request")
        return AdmissionResolution("unresolved", query.strip(), source="request",
                                   reason=f"hadm_id {request_hadm_id} is not an admission of this patient")

    refs = []                                            # (start, end, kind, match)
    for kind, pattern in (("hadm_id", _ID_RE), ("date", _DATE_AFTER_RE), ("date", _DATE_BEFORE_RE),
                          ("ordinal", _ORDINAL_RE)):
        for m in pattern.finditer(query):
            if not any(m.start() < e and s < m.end() for s, e, _, _ in refs):
                refs.append((m.start(), m.end(), kind, m))
    if not refs:
        return AdmissionResolution("none", query.strip())
    if len(refs) > 1:
        return AdmissionResolution("ambiguous", query.strip(),
                                   reason="the question names more than one admission",
                                   phrase="; ".join(query[s:e].strip() for s, e, _, _ in sorted(refs)))

    start, end, kind, m = refs[0]
    phrase = query[start:end].strip()
    if kind == "hadm_id":
        source, hits = "rule:hadm_id", [h for h in (int(m.group("id")),) if h in known]
        missing = f"hadm_id {m.group('id')} is not an admission of this patient"
    elif kind == "date":
        day = _day(m.group("date"))
        source = "rule:date"
        hits = _by_date(admissions, day, m.group("noun").lower(), (m.groupdict().get("verb") or "").lower()) if day else []
        missing = f"no admission matches {m.group('date')}"
    else:
        last = m.group("ord").lower() not in ("first", "earliest")
        source = "rule:last" if last else "rule:first"
        edge = (max if last else min)((a[1] for a in admissions), default=None)
        hits = [int(a[0]) for a in admissions if a[1] == edge]
        missing = "this patient has no admissions on record"

    if len(hits) == 1:
        return AdmissionResolution("resolved", _strip(query, start, end), hits[0], source, phrase=phrase)
    if not hits:
        return AdmissionResolution("unresolved", query.strip(), source=source, reason=missing, phrase=phrase)
    return AdmissionResolution("ambiguous", query.strip(), source=source, phrase=phrase,
                               reason=f"{len(hits)} admissions match: {sorted(hits)}")
