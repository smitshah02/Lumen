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

ponytail: a question about one named drug ("was she given heparin?") is not
routed here; it needs drug-name matching. Add when a question set needs it.
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


def _order(row: dict) -> str:
    """One prescription row as the table has it: drug, dose, unit, route, and
    doses_per_24_hrs stated as a number. The table has no frequency text, so
    none is written or inferred (no "BID", no "daily")."""
    dose = " ".join(str(v).strip() for v in (row["dose_val_rx"], row["dose_unit_rx"]) if v not in (None, ""))
    rate = row["doses_per_24_hrs"]
    parts = [str(row["drug"]).strip(), dose, str(row["route"] or "").strip(),
             f"{rate:g} dose(s) per 24 hours" if rate is not None else ""]
    return " ".join(p for p in parts if p)


def render(kind: str, rows: list[dict], subject_id: int, hadm_id: int) -> tuple[list[str], dict]:
    """(answer sentences, one evidence object) for a non-empty result."""
    table, title = SOURCE[kind]
    where = f"admission {hadm_id}"
    if kind == "prescriptions":
        distinct: dict[str, list] = {}
        for row in rows:                                   # same order repeated: one entry, earliest start to latest stop
            entry = distinct.setdefault(_order(row), [row["starttime"], row["stoptime"]])
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
