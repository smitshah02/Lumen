"""Structured SQL answer paths (data-foundation plan, E9 / decision R5). No
services: the queries run on an in-memory SQLite database with invented rows."""
import sqlite3
from contextlib import contextmanager

import pytest

from src.generation import structured_lookup as sl
from src.storage.readiness import STRUCTURED, DataSourceNotReady

SID, HADM, OTHER = 7, 22595853, 22841357


# --- routing by wording -----------------------------------------------------------------------
ROUTES = [
    ("What medications were given?", "prescriptions"),
    ("Which medications did she receive?", "prescriptions"),
    ("Which medications did she receive during this admission?", "prescriptions"),
    ("What medications did they give her?", "prescriptions"),
    ("What medications were given?", "prescriptions"),
    ("What drugs did they administer?", "prescriptions"),
    ("Which medications were administered?", "prescriptions"),
    ("What medications did they order?", "prescriptions"),
    ("Which medications were ordered?", "prescriptions"),
    ("Give me the medications.", None),                                 # "give me" is a request, not an inpatient verb
    ("What medications did she receive at discharge?", None),
    ("What home medications did she receive?", None),
    ("What current medications was she given?", None),
    ("What medications was she given when sent home?", None),
    ("What medications were administered during the stay?", "prescriptions"),
    ("What drugs were ordered during the admission?", "prescriptions"),
    ("List the inpatient medications.", "prescriptions"),
    ("What medications had she received?", "prescriptions"),
    # not the discharge list, not home or current medications
    ("What medications was she discharged on?", None),
    ("What are the discharge medications?", None),
    ("What medications was she sent home on?", None),
    ("What home medications was she given?", None),
    ("What are her current medications?", None),
    ("What medications is she taking?", None),
    ("Was she given heparin?", None),                                   # one named drug: not routed
    ("What medications were given and why was she admitted?", None),    # an extra clause sends it all to retrieval
    # diagnoses: only when coded / ICD is asked for
    ("What were the ICD diagnosis codes?", "diagnoses"),
    ("List the coded diagnoses.", "diagnoses"),
    ("What are the ICD codes for the admission?", "diagnoses"),
    ("What ICD-10 diagnoses were assigned?", "diagnoses"),
    ("What was the diagnosis?", None),
    ("Why was the patient admitted?", None),
    ("What did the discharge diagnosis say?", None),
    ("What was the primary diagnosis code and how was it treated?", None),
    # procedures: only when coded / ICD is asked for
    ("What were the coded procedures?", "procedures"),
    ("List the ICD procedure codes.", "procedures"),
    ("What procedure did she undergo?", None),
    ("What procedures were performed?", None),
    ("", None),
]


@pytest.mark.parametrize("question,kind", ROUTES)
def test_only_clear_wording_reaches_a_structured_source(question, kind):
    assert sl.structured_kind(question) == kind


# --- the queries, on real SQL -------------------------------------------------------------------
@pytest.fixture
def db(monkeypatch):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE prescriptions (subject_id, hadm_id, drug, dose_val_rx, dose_unit_rx, route, doses_per_24_hrs, starttime, stoptime);
        CREATE TABLE diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version);
        CREATE TABLE procedures_icd (subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version);
        CREATE TABLE d_icd_diagnoses (icd_code, icd_version, long_title);
        CREATE TABLE d_icd_procedures (icd_code, icd_version, long_title);
    """)
    c.executemany("INSERT INTO prescriptions VALUES (?,?,?,?,?,?,?,?,?)", [
        (SID, HADM, "Furosemide", "40", "mg", "IV", 2, "2180-05-06 23:00:00", "2180-05-07 08:00:00"),
        (SID, HADM, "Furosemide", "40", "mg", "IV", 2, "2180-05-07 08:00:00", "2180-05-07 17:00:00"),   # the same order renewed
        (SID, HADM, "Heparin", "5000", "UNIT", "SC", None, "2180-05-06 22:30:00", None),
        (SID, OTHER, "Warfarin", "5", "mg", "PO", 1, "2180-06-26 20:00:00", "2180-06-27 10:00:00"),      # another admission
        (SID + 1, HADM, "Insulin", "10", "UNIT", "SC", 3, "2180-05-06 23:00:00", None),                  # another patient
    ])
    c.executemany("INSERT INTO diagnoses_icd VALUES (?,?,?,?,?)", [
        (SID, HADM, 2, "I10", 10), (SID, HADM, 1, "I5023 ", 10), (SID, HADM, 3, "ZZZ99", 10), (SID, OTHER, 1, "4280", 9)])
    c.executemany("INSERT INTO d_icd_diagnoses VALUES (?,?,?)", [
        ("I5023", 10, "Acute on chronic systolic (congestive) heart failure"), ("I10", 10, "Essential (primary) hypertension"),
        ("I10", 9, "A different ICD-9 title that must not be joined"), ("4280", 9, "Congestive heart failure, unspecified")])
    c.executemany("INSERT INTO procedures_icd VALUES (?,?,?,?,?,?)", [
        (SID, HADM, 1, "2180-05-07", "4A023N7", 10), (SID, HADM, 2, "2180-05-06", "QQQ0000", 10), (SID, OTHER, 1, "2180-06-26", "3722", 9)])
    c.executemany("INSERT INTO d_icd_procedures VALUES (?,?,?)", [
        ("4A023N7", 10, "Measurement of Cardiac Sampling and Pressure, Left Heart, Percutaneous Approach"),
        ("3722", 9, "Left heart cardiac catheterization")])

    class _Conn:
        def execute(self, stmt, params):
            cursor = c.execute(str(stmt), params)
            return type("R", (), {"mappings": lambda self: [dict(r) for r in cursor.fetchall()]})()

    @contextmanager
    def connect():
        yield _Conn()
    return connect


def test_prescriptions_are_this_admissions_orders_with_their_fields(db):
    rows = sl.fetch("prescriptions", SID, HADM, db)
    assert [r["drug"] for r in rows] == ["Heparin", "Furosemide", "Furosemide"]           # by start time; no Warfarin, no Insulin
    assert rows[0] == {"drug": "Heparin", "dose_val_rx": "5000", "dose_unit_rx": "UNIT", "route": "SC",
                       "doses_per_24_hrs": None, "starttime": "2180-05-06 22:30:00", "stoptime": None}
    sentences, evidence = sl.render("prescriptions", rows, SID, HADM)
    assert sentences[0].startswith(f"Inpatient medication orders for admission {HADM}, from the prescriptions table: 3 order(s), 2 distinct.")
    assert "not a record of what was administered and not the discharge medication list" in sentences[0]
    assert sentences[1:] == ["Heparin 5000 UNIT SC (ordered 2180-05-06 to not recorded)",
                             "Furosemide 40 mg IV 2 dose(s) per 24 hours (ordered 2180-05-06 to 2180-05-07)"]
    assert "discharged on" not in " ".join(sentences).lower() and "was given" not in " ".join(sentences).lower()
    assert (evidence["label"], evidence["source_type"], evidence["chunk_id"], evidence["hadm_id"]) == ("R1", "prescriptions", -1, HADM)
    assert evidence["text"].endswith(f"Source: prescriptions table, subject {SID}, hadm_id {HADM}.")


def test_rate_is_stated_as_a_number_and_no_frequency_wording_is_invented():
    import re
    rows = [{"drug": "Drug", "dose_val_rx": "1", "dose_unit_rx": "mg", "route": "PO", "doses_per_24_hrs": rate,
             "starttime": "2180-05-06 10:00:00", "stoptime": "2180-05-07 10:00:00"} for rate in (1, 2, 3, 4, 0.5, None)]
    sentences, evidence = sl.render("prescriptions", rows, SID, HADM)
    assert sentences[1:] == [f"Drug 1 mg PO {r} dose(s) per 24 hours (ordered 2180-05-06 to 2180-05-07)" for r in ("1", "2", "3", "4", "0.5")] \
        + ["Drug 1 mg PO (ordered 2180-05-06 to 2180-05-07)"]                           # no value, nothing said
    written = " ".join(sentences) + evidence["text"]
    assert not re.search(r"\b(BID|TID|QID|QD|QHS|Q\d+H|daily|twice|three times|four times|every|frequency|hourly|PRN)\b", written, re.I)
    assert "not a record of what was administered and not the discharge medication list" in sentences[0]


@pytest.mark.parametrize("kind,row", [
    ("prescriptions", lambda i: {"drug": f"Drug{i}", "dose_val_rx": "1", "dose_unit_rx": "mg", "route": "PO",
                                 "doses_per_24_hrs": 1, "starttime": "2180-05-06 10:00:00", "stoptime": None}),
    ("diagnoses", lambda i: {"seq_num": i, "icd_code": f"I{i:03d}", "icd_version": 10, "long_title": f"Title {i}"}),
    ("procedures", lambda i: {"seq_num": i, "icd_code": f"P{i:03d}", "icd_version": 10, "long_title": f"Title {i}",
                              "chartdate": "2180-05-06"}),
])
def test_display_limit_cuts_the_answer_only_and_says_how_much(monkeypatch, kind, row):
    assert sl.ANSWER_DISPLAY_LIMIT == 50
    rows = [row(i) for i in range(1, sl.ANSWER_DISPLAY_LIMIT + 8)]                          # 57 rows
    sentences, evidence = sl.render(kind, rows, SID, HADM)
    assert len(sentences) == 1 + 50 + 1 and sentences[-1] == "7 further entries are in the source and not listed here"
    assert "57" in sentences[0] and "57 row(s)" in evidence["text"]                         # the total is always stated
    assert f"{57 if kind != 'prescriptions' else 'Drug57'}" in evidence["text"]            # the evidence holds every row
    # a different limit changes what is displayed and nothing else
    monkeypatch.setattr(sl, "ANSWER_DISPLAY_LIMIT", 5)
    fewer, same_evidence = sl.render(kind, rows, SID, HADM)
    assert len(fewer) == 1 + 5 + 1 and fewer[-1] == "52 further entries are in the source and not listed here"
    assert same_evidence == evidence and fewer[0] == sentences[0]
    one_over = sl.render(kind, rows[:6], SID, HADM)[0]
    assert one_over[-1] == "1 further entry is in the source and not listed here"
    monkeypatch.setattr(sl, "ANSWER_DISPLAY_LIMIT", 50)
    assert sl.render(kind, rows[:50], SID, HADM)[0][-1] != "0 further entries are in the source and not listed here"
    assert len(sl.render(kind, rows[:50], SID, HADM)[0]) == 51                              # exactly at the limit: nothing omitted


def test_diagnoses_come_in_coded_order_with_titles_and_bare_codes_are_kept(db):
    rows = sl.fetch("diagnoses", SID, HADM, db)
    assert [(r["seq_num"], r["icd_code"], r["icd_version"]) for r in rows] == [(1, "I5023", 10), (2, "I10", 10), (3, "ZZZ99", 10)]
    sentences, evidence = sl.render("diagnoses", rows, SID, HADM)
    assert "billing codes assigned to the admission, not the clinician's wording" in sentences[0]
    assert sentences[1:] == ["1. Acute on chronic systolic (congestive) heart failure (ICD-10 I5023)",
                             "2. Essential (primary) hypertension (ICD-10 I10)",           # the ICD-10 title, not the ICD-9 one
                             "3. ICD-10 ZZZ99 (no title on record)"]
    assert evidence["source_type"] == "diagnoses_icd" and "3 row(s)" in evidence["text"]


def test_procedures_keep_their_dates_titles_and_bare_codes(db):
    rows = sl.fetch("procedures", SID, HADM, db)
    assert [r["icd_code"] for r in rows] == ["QQQ0000", "4A023N7"]                        # by date; nothing from the other admission
    sentences, evidence = sl.render("procedures", rows, SID, HADM)
    assert sentences[0].startswith(f"Coded procedures for admission {HADM}, by date, from the procedures_icd table: 2.")
    assert sentences[1:] == ["2180-05-06: ICD-10 QQQ0000 (no title on record)",
                             "2180-05-07: Measurement of Cardiac Sampling and Pressure, Left Heart, Percutaneous Approach (ICD-10 4A023N7)"]
    assert evidence["source_type"] == "procedures_icd"


@pytest.mark.parametrize("kind", ["prescriptions", "diagnoses", "procedures"])
def test_an_admission_with_no_rows_returns_nothing(db, kind):
    assert sl.fetch(kind, SID, 99999999, db) == []
    assert ":sid" in sl.SQL[kind] and ":hadm" in sl.SQL[kind] and "{" not in sl.SQL[kind]   # parameters only


# --- the graph node and its routing ---------------------------------------------------------------
RESOLVED = {"status": "resolved", "hadm_id": HADM, "source": "rule:last", "reason": None,
            "phrase": "during her last admission", "retrieval_query": "What medications were given?"}


def _state(question="What medications were given during her last admission?", scope=RESOLVED, retrieval=None):
    scope = None if scope is None else {**scope, "retrieval_query": retrieval or scope.get("retrieval_query") or question}
    state = {"query": question, "subject_id": SID, "query_type": "chart_review", "temporal_mode": "all",
             "query_complexity": "simple", "classified_by": "rules"}
    if scope is not None:
        state["admission_scope"] = scope
    return state


@pytest.fixture
def g(monkeypatch):
    import src.agents.graph as graph
    monkeypatch.setattr(graph, "PROFILE_SETTINGS", {"admission_scope": True, "sql_paths": True})
    return graph


def test_structured_profile_routes_clear_questions_with_a_resolved_admission(g):
    assert g.route_from_triage(_state()) == "structured_lookup"
    assert g.route_from_triage(_state(retrieval="What were the ICD diagnosis codes?")) == "structured_lookup"
    assert g.route_from_triage(_state(retrieval="What were the coded procedures?")) == "structured_lookup"
    # the wording gate reads the question without its admission phrase, so "discharge" in that phrase does not matter
    assert g.route_from_triage(_state(question="What medications were given in the admission with discharge on 2180-05-07?",
                                      retrieval="What medications were given?")) == "structured_lookup"


@pytest.mark.parametrize("retrieval", ["What medications was she discharged on?", "What are her current medications?",
                                       "What was the diagnosis?", "What procedure did she undergo?"])
def test_narrative_and_discharge_questions_stay_on_note_retrieval(g, retrieval):
    assert g.route_from_triage(_state(question=retrieval, retrieval=retrieval)) == "patient_retrieval"


@pytest.mark.parametrize("scope", [None, {"status": "none", "hadm_id": None}, {"status": "unresolved", "hadm_id": None},
                                   {"status": "ambiguous", "hadm_id": None}])
def test_without_a_resolved_admission_nothing_is_queried(g, monkeypatch, scope):
    monkeypatch.setattr(g.structured, "fetch", lambda *a, **k: pytest.fail("no admission: must not query"))
    assert g._structured_request(_state(scope=scope, retrieval="What medications were given?")) is None
    plain = _state(question="What medications were given?", scope=scope)
    assert g.route_from_triage(plain) == "patient_retrieval"                              # no guess, no all-admissions query


@pytest.mark.parametrize("settings", [{"admission_scope": False, "sql_paths": False},     # control
                                      {"admission_scope": True, "sql_paths": False}])     # scoped
def test_control_and_scoped_never_use_the_structured_paths(monkeypatch, settings):
    import src.agents.graph as graph
    monkeypatch.setattr(graph, "PROFILE_SETTINGS", settings)
    monkeypatch.setattr(graph.structured, "fetch", lambda *a, **k: pytest.fail("disabled profile: must not query"))
    plain = _state(question="What medications were given?")                              # a resolved admission is in state
    assert graph._structured_request(plain) is None and graph.route_from_triage(plain) == "patient_retrieval"
    assert graph.route_from_triage(_state()) != "structured_lookup"


def test_node_answers_from_the_rows_and_marks_scope_applied(g, monkeypatch):
    rows = [{"drug": "Heparin", "dose_val_rx": "5000", "dose_unit_rx": "UNIT", "route": "SC", "doses_per_24_hrs": None,
             "starttime": "2180-05-06 22:30:00", "stoptime": None}]
    asked = []
    monkeypatch.setattr(g.structured, "fetch", lambda kind, sid, hadm: asked.append((kind, sid, hadm)) or rows)
    out = g.structured_lookup(_state())
    assert asked == [("prescriptions", SID, HADM)]
    assert out["final_answer"] == out["draft_answer"] and out["final_answer"].count("[R1]") == 2
    assert out["review_status"] == "auto_approved" and not out["needs_human_review"] and out["admission_scope_applied"] is True
    assert all(c["verified"] and c["label"] == "R1" and c["chunk_id"] == -1 for c in out["citations"])
    assert "prescriptions table" in out["citations"][0]["verification_note"]
    (evidence,) = out["structured_evidence"]
    assert evidence["label"] == "R1" and evidence["source_type"] == "prescriptions"
    assert not any(e.get("label", "").startswith("S") for e in out["structured_evidence"])     # no note citation is invented
    assert g.route_after_structured_lookup(out) == "finalize"


def test_no_rows_is_a_miss_that_falls_through_to_retrieval(g, monkeypatch):
    monkeypatch.setattr(g.structured, "fetch", lambda *a: [])
    out = g.structured_lookup(_state())
    assert out == {"node_trail": ["structured_lookup"]}
    assert g.route_after_structured_lookup(out) == "patient_retrieval"


def test_unready_structured_data_is_raised_not_retrieved_around(g, monkeypatch):
    def not_ready(*a):
        raise DataSourceNotReady(STRUCTURED, "the latest structured load is running")
    monkeypatch.setattr(g.structured, "fetch", not_ready)
    with pytest.raises(DataSourceNotReady):
        g.structured_lookup(_state())


def test_fetch_checks_readiness_before_reading(monkeypatch):
    def not_ready(component):
        raise DataSourceNotReady(component, "no structured load has been run")
    monkeypatch.setattr(sl.readiness, "require", not_ready)
    with pytest.raises(DataSourceNotReady):
        sl.fetch("diagnoses", SID, HADM, lambda: pytest.fail("must stop at the guard"))


def test_lab_and_admissions_routing_is_untouched_by_the_new_path(g):
    labs = {**_state(question="most recent creatinine", scope={"status": "none", "hadm_id": None}), "query_type": "lab_trend",
            "temporal_mode": "latest"}
    assert g.route_from_triage(labs) == "lab_lookup"
    count = _state(question="How many admissions has the patient had?", scope={"status": "none", "hadm_id": None})
    assert g.route_from_triage(count) == "encounter_lookup"


def test_structured_answers_reach_the_api_sources():
    import src.api.app as api
    assert 'st.get("structured_evidence")' in open(api.__file__).read()
