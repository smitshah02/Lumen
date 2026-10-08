"""Structured SQL answers (data-foundation plan, E9 / decision R5).

Three questions a table answers better than five note chunks, each for ONE
resolved admission:

    prescriptions   inpatient medication orders        (prescriptions)
    diagnoses       coded diagnoses, in coded order    (diagnoses_icd + d_icd_diagnoses)
    procedures      coded procedures with their dates  (procedures_icd + d_icd_procedures)

The gate is strict, like the lab and admissions paths: a question is routed here
only when every word of it is understood and it clearly asks for that source.
Anything about discharge, home or current medications, and any narrative
question ("what was the diagnosis?"), is left to note retrieval, because
`prescriptions` is orders placed during the stay, not the discharge list, and
the coded tables are billing codes, not the clinician's wording.

One named drug is a second, narrower door (`drug_order_question`): the dose,
route, doses per 24 hours or whole inpatient order of ONE drug in the resolved
admission. The drug is whatever single run of words the path does not
otherwise understand, and it must match exactly one drug in that admission's
orders (`match_drug`); a similar name is never substituted. A yes/no question
("was she given heparin?") is still not routed here: an order is not proof the
drug was given.
"""
from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import text

from src.storage import readiness

_COMMON = frozenset("""
    what which list show all the a an of were was is are did does do patient patient's s
    she he her his their they has have had been during in on at for to this that these those
    stay admission hospitalization hospitalisation hospital and any please while there with
""".split())
_VOCABULARY = {
    "prescriptions": _COMMON | frozenset("medication medications meds drug drugs give given receive received "
                                         "administer administered order ordered orders inpatient".split()),
    "diagnoses": _COMMON | frozenset("icd icd9 icd10 9 10 coded code codes diagnosis diagnoses diagnostic billing "
                                     "assigned listed recorded".split()),
    "procedures": _COMMON | frozenset("icd icd9 icd10 9 10 coded code codes procedure procedures billing assigned "
                                      "listed recorded performed".split()),
}
_MEDICATION = frozenset("medication medications meds drug drugs".split())
# The plain forms of four verbs, listed out: no stemming, no fuzzy matching. "me" is
# not a known word, so "give me the medications" is a request, not an inpatient verb.
_INPATIENT = frozenset("give given receive received administer administered order ordered orders inpatient".split())
_CODED = frozenset("icd icd9 icd10 coded code codes".split())
_DIAGNOSIS = frozenset("diagnosis diagnoses diagnostic".split())
_PROCEDURE = frozenset("procedure procedures".split())


# --- one named drug ---------------------------------------------------------------------------------
# Every word of a single-drug order question is one of these or part of the drug's name.
_ORDER_WORDS = _COMMON | _INPATIENT | _MEDICATION | frozenset("""
    dose doses dosage dosing route routes by how many often per times time day daily hours hour 24
    frequency each every full complete entire
""".split())
# Anything about what the patient went home on, takes now or was prescribed is the discharge
# list or the home list, which this table is not. Such a question is never routed here.
_NOT_INPATIENT = frozenset("""
    discharge discharged home current currently outpatient sent taking takes take prescribed
    prescription prescriptions
""".split())
_ORDER_DOSE = frozenset("dose doses dosage dosing".split())
_ORDER_ROUTE = frozenset("route routes".split())
_ORDER_NOUN = frozenset("order orders".split())
_ORDER_RATE_RE = re.compile(r"\b(?:per|a|each|every) (?:24 hours?|day)\b|\bhow often\b|\bfrequency\b")
MAX_DRUG_WORDS = 5


def drug_order_question(question: str) -> Optional[str]:
    """The drug a question asks about, when it asks for the dose, route, doses
    per 24 hours (or times a day) or whole inpatient order of ONE named drug;
    else None.

    As strict as structured_kind: every word must be understood. The one thing
    allowed beyond the known words is a single unbroken run of up to five words,
    which is taken to be the drug's name. A second unknown word anywhere else
    ("why", "discharged", a second drug) sends the question to retrieval. Whether
    that name is a drug in this admission is decided against the table rows, by
    match_drug, never here."""
    tokens = re.findall(r"[a-z0-9]+", (question or "").lower())
    words = set(tokens)
    if words & _NOT_INPATIENT:
        return None
    asks = (words & _ORDER_DOSE or words & _ORDER_ROUTE or words & _ORDER_NOUN
            or _ORDER_RATE_RE.search(" ".join(tokens)))
    unknown = [i for i, t in enumerate(tokens) if t not in _ORDER_WORDS]
    if not asks or not unknown or len(unknown) > MAX_DRUG_WORDS or unknown[-1] - unknown[0] + 1 != len(unknown):
        return None
    name = [tokens[i] for i in unknown]
    if not any(len(t) >= 3 and t.isalpha() for t in name):
        return None
    return " ".join(name)


def _drug_words(name) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", str(name or "").lower()))


def match_drug(drug: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """(the rows of the one drug `drug` names, []) or ([], the drug names it
    could mean). Whole words only, no stemming and no spelling tolerance:

      a drug whose name is exactly those words wins ("heparin" is Heparin, not
      Heparin Flush);
      otherwise the name must contain every word, and exactly one drug in the
      admission may do so ("ampicillin" is Ampicillin Sodium when that is the
      only one). Two or more is ambiguous and matches nothing."""
    want = _drug_words(drug)
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        by_name.setdefault(" ".join(str(row["drug"] or "").split()), []).append(row)
    exact = [n for n in by_name if _drug_words(n) == want]
    if exact:
        return [r for n in exact for r in by_name[n]], []
    containing = sorted(n for n in by_name if want and want <= _drug_words(n))
    if len({_drug_words(n) for n in containing}) == 1:
        return [r for n in containing for r in by_name[n]], []
    return [], containing


def structured_kind(question: str) -> Optional[str]:
    """Which structured source the question clearly asks for, or None.
    None unless every word is one the path understands, so an extra clause
    ("... and why was she admitted?") sends the whole question to retrieval."""
    words = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    if words & _MEDICATION and words & _INPATIENT and words <= _VOCABULARY["prescriptions"]:
        return "prescriptions"
    if words & _CODED and words & _PROCEDURE and words <= _VOCABULARY["procedures"]:
        return "procedures"
    if words & _CODED and (words & _DIAGNOSIS or not words & _PROCEDURE) and words <= _VOCABULARY["diagnoses"]:
        return "diagnoses"
    return None


SQL = {
    "prescriptions": """
        SELECT drug, dose_val_rx, dose_unit_rx, route, doses_per_24_hrs, starttime, stoptime
        FROM prescriptions
        WHERE subject_id = :sid AND hadm_id = :hadm
        ORDER BY starttime IS NULL, starttime, drug, dose_val_rx, route""",
    "diagnoses": """
        SELECT d.seq_num, trim(d.icd_code) AS icd_code, d.icd_version, t.long_title
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses t ON t.icd_code = trim(d.icd_code) AND t.icd_version = d.icd_version
        WHERE d.subject_id = :sid AND d.hadm_id = :hadm
        ORDER BY d.seq_num, icd_code""",
    "procedures": """
        SELECT p.seq_num, trim(p.icd_code) AS icd_code, p.icd_version, t.long_title, p.chartdate
        FROM procedures_icd p
        LEFT JOIN d_icd_procedures t ON t.icd_code = trim(p.icd_code) AND t.icd_version = p.icd_version
        WHERE p.subject_id = :sid AND p.hadm_id = :hadm
        ORDER BY p.chartdate IS NULL, p.chartdate, p.seq_num, icd_code""",
}
SOURCE = {
    "prescriptions": ("prescriptions", "Inpatient medication orders"),
    "diagnoses": ("diagnoses_icd", "Coded diagnoses"),
    "procedures": ("procedures_icd", "Coded procedures"),
}
LABEL = "R1"
# How many rows an answer spells out. Presentation only: the query returns every
# row, the evidence object holds every row and the answer states the total and
# how many it left out. Not a setting; change it here.
ANSWER_DISPLAY_LIMIT = 50


def fetch(kind: str, subject_id: int, hadm_id: int, conn_factory=None) -> list[dict]:
    """The rows of one admission. Raises DataSourceNotReady when the profile's
    structured data is not usable; an empty list is an ordinary miss."""
    readiness.require(readiness.STRUCTURED)
    if conn_factory is None:
        from src.storage import engine
        conn_factory = engine.connect
    with conn_factory() as c:
        return [dict(r) for r in c.execute(text(SQL[kind]), {"sid": subject_id, "hadm": hadm_id}).mappings()]


def _day(value) -> str:
    return str(value)[:10] if value is not None else "not recorded"


def _code(row: dict) -> str:
    return f"ICD-{row['icd_version']} {row['icd_code']}"


def _order(row: dict, name_missing: bool = False) -> str:
    """One prescription row as the table has it: drug, dose, unit, route, and
    doses_per_24_hrs stated as a number. The table has no frequency text, so
    none is written or inferred (no "BID", no "daily"). `name_missing` says
    which of dose, route and rate the row does not have, for a question that
    asked about one drug's order."""
    dose = " ".join(str(v).strip() for v in (row["dose_val_rx"], row["dose_unit_rx"]) if v not in (None, ""))
    rate = row["doses_per_24_hrs"]
    route = str(row["route"] or "").strip()
    parts = [str(row["drug"]).strip(), dose, route,
             f"{rate:g} dose(s) per 24 hours" if rate is not None else ""]
    missing = [label for label, value in (("dose", dose), ("route", route), ("doses per 24 hours", "" if rate is None else "x"))
               if not value] if name_missing else []
    return " ".join(p for p in parts if p) + (f" ({', '.join(missing)} not recorded)" if missing else "")


def render(kind: str, rows: list[dict], subject_id: int, hadm_id: int, drug: Optional[str] = None) -> tuple[list[str], dict]:
    """(answer sentences, one evidence object) for a non-empty result. `drug`
    is set when `rows` are the orders of one requested drug (match_drug)."""
    table, title = SOURCE[kind]
    where = f"admission {hadm_id}"
    if kind == "prescriptions":
        if drug:
            names = sorted({" ".join(str(r["drug"] or "").split()) for r in rows})
            where, title = f"{' / '.join(names)} in admission {hadm_id}", f"{title} for {' / '.join(names)}"
        distinct: dict[str, list] = {}
        for row in rows:                                   # same order repeated: one entry, earliest start to latest stop
            entry = distinct.setdefault(_order(row, name_missing=bool(drug)), [row["starttime"], row["stoptime"]])
            if row["stoptime"] is not None and (entry[1] is None or row["stoptime"] > entry[1]):
                entry[1] = row["stoptime"]
        lines = [f"{order} (ordered {_day(start)} to {_day(stop)})" for order, (start, stop) in distinct.items()]
        sentences = [f"Inpatient medication orders for {where}, from the prescriptions table: {len(rows)} order(s), "
                     f"{len(lines)} distinct. These are orders placed during the stay; they are not a record of what "
                     f"was administered and not the discharge medication list"]
    elif kind == "diagnoses":
        lines = [f"{r['seq_num']}. {r['long_title']} ({_code(r)})" if r["long_title"] else
                 f"{r['seq_num']}. {_code(r)} (no title on record)" for r in rows]
        sentences = [f"Coded diagnoses for {where}, in coded order, from the diagnoses_icd table: {len(rows)}. These are "
                     f"billing codes assigned to the admission, not the clinician's wording in the discharge summary"]
    else:
        lines = [f"{_day(r['chartdate'])}: {r['long_title']} ({_code(r)})" if r["long_title"] else
                 f"{_day(r['chartdate'])}: {_code(r)} (no title on record)" for r in rows]
        sentences = [f"Coded procedures for {where}, by date, from the procedures_icd table: {len(rows)}. These are "
                     f"billing codes assigned to the admission"]
    sentences += lines[:ANSWER_DISPLAY_LIMIT]
    if len(lines) > ANSWER_DISPLAY_LIMIT:
        sentences.append(f"{len(lines) - ANSWER_DISPLAY_LIMIT} further entr{'y is' if len(lines) - ANSWER_DISPLAY_LIMIT == 1 else 'ies are'} "
                         f"in the source and not listed here")
    evidence = {"chunk_id": -1, "source_type": table, "note_type": table, "label": LABEL, "score": 1.0,
                "subject_id": subject_id, "hadm_id": hadm_id, "charttime": None,
                "text": f"{title}, {len(rows)} row(s): " + "; ".join(lines)
                        + f". Source: {table} table, subject {subject_id}, hadm_id {hadm_id}."}
    return sentences, evidence
