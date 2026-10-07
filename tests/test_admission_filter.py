"""Admission-scoped retrieval filter (data-foundation plan, E4 / decision R1).

The filter and the stay-window query are plain SQL, so they are run here on an
in-memory SQLite database: the same text Postgres runs, no service needed."""
import sqlite3

import pytest

from src.agents.admission_scope import STAY_WINDOW_SQL
from src.retrieval.hybrid_retriever_v2 import ADMISSION_SCOPE_SQL, _admission_clause

HADM, OTHER = 22595853, 22841357


@pytest.fixture
def db():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE admissions (hadm_id INTEGER, edregtime TEXT, admittime TEXT, dischtime TEXT);
        CREATE TABLE cn (note_id INTEGER, charttime TEXT);
        CREATE TABLE nc (chunk_id INTEGER, note_id INTEGER, hadm_id INTEGER);
    """)
    c.executemany("INSERT INTO admissions VALUES (?,?,?,?)", [
        (HADM, "2180-05-06 19:17:00", "2180-05-06 22:23:00", "2180-05-07 17:15:00"),    # came through the ED
        (OTHER, None, "2180-06-26 18:27:00", "2180-06-27 18:49:00"),                    # direct admission
    ])
    notes = [  # chunk_id, hadm_id, charttime
        (1, HADM, "2180-05-07 00:00:00"),       # this admission's discharge summary
        (2, OTHER, "2180-06-27 00:00:00"),      # another admission's
        (3, OTHER, "2180-05-06 23:00:00"),      # another admission's id, but charted inside this stay
        (4, None, "2180-05-06 20:00:00"),       # unlinked radiology in the ED, before the admit time
        (5, None, "2180-05-07 10:00:00"),       # unlinked radiology during the stay
        (6, None, "2180-05-06 19:00:00"),       # unlinked, minutes before ED registration
        (7, None, "2180-05-07 17:30:00"),       # unlinked, minutes after discharge
        (8, None, "2180-06-27 09:00:00"),       # unlinked, inside the OTHER stay
    ]
    c.executemany("INSERT INTO cn VALUES (?,?)", [(i, t) for i, _, t in notes])
    c.executemany("INSERT INTO nc VALUES (?,?,?)", [(i, i, h) for i, h, _ in notes])
    return c


def _window(db, hadm_id):
    return db.execute(STAY_WINDOW_SQL, {"hadm_id": hadm_id}).fetchone()


def _chunks(db, hadm_id, stay_window):
    params = {}
    sql = "SELECT nc.chunk_id FROM nc JOIN cn ON cn.note_id = nc.note_id WHERE 1=1" \
          + _admission_clause(hadm_id, stay_window, params) + " ORDER BY nc.chunk_id"
    return [r[0] for r in db.execute(sql, params)]


def test_window_starts_at_ed_registration_when_there_is_one_else_admit_time(db):
    assert _window(db, HADM) == ("2180-05-06 19:17:00", "2180-05-07 17:15:00")
    assert _window(db, OTHER) == ("2180-06-26 18:27:00", "2180-06-27 18:49:00")


def test_scope_keeps_the_admissions_notes_and_unlinked_notes_inside_the_stay(db):
    # 1: same hadm_id. 4: unlinked, in the ED (only inside because the window starts at ED registration). 5: unlinked, in the stay.
    assert _chunks(db, HADM, _window(db, HADM)) == [1, 4, 5]


def test_another_admissions_id_is_never_pulled_in_by_an_overlapping_time(db):
    assert 3 not in _chunks(db, HADM, _window(db, HADM))


def test_unlinked_notes_outside_the_stay_are_excluded(db):
    scoped = _chunks(db, HADM, _window(db, HADM))
    assert 6 not in scoped and 7 not in scoped and 8 not in scoped


def test_admit_time_bounds_the_window_when_there_is_no_ed_registration(db):
    db.execute("INSERT INTO cn VALUES (9, '2180-06-26 18:00:00')")       # before the admit time; no ED visit to extend it
    db.execute("INSERT INTO nc VALUES (9, 9, NULL)")
    assert _chunks(db, OTHER, _window(db, OTHER)) == [2, 3, 8]


def test_no_admission_means_no_filter_and_an_id_alone_is_the_old_equality(db):
    params = {}
    assert _admission_clause(None, None, params) == "" and params == {}
    assert _admission_clause(None, ("a", "b"), params) == "" and params == {}
    assert _chunks(db, None, None) == [1, 2, 3, 4, 5, 6, 7, 8]              # patient-wide, untouched
    assert _admission_clause(HADM, None, params) == " AND nc.hadm_id = :hadm_id" and params == {"hadm_id": HADM}
    assert _chunks(db, HADM, None) == [1]
    assert _admission_clause(HADM, ("a", "b"), {}) == ADMISSION_SCOPE_SQL


class _Retriever:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return []


QUESTION = "During her last admission, what happened to her kidney function?"
RESOLVED = {"status": "resolved", "hadm_id": HADM, "source": "rule:last", "reason": None,
            "phrase": "During her last admission", "retrieval_query": "what happened to her kidney function?"}
PATIENT_WIDE = {"query": QUESTION, "subject_id": 7, "temporal_filter": "all"}


def _retrieve(monkeypatch, profile_scoping, scope):
    import src.agents.graph as g
    retriever, windows = _Retriever(), []
    monkeypatch.setattr(g, "get_retrievers", lambda: (retriever, None))
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": profile_scoping})
    monkeypatch.setattr(g, "load_stay_window", lambda hadm: windows.append(hadm) or ("s", "e"))
    state = {"query": QUESTION, "subject_id": 7, "temporal_mode": "all"}
    if scope is not None:
        state["admission_scope"] = scope
    g.patient_retrieval(state)
    (call,) = retriever.calls
    call.pop("top_k")
    return call, windows


def test_control_profile_searches_patient_wide_even_if_a_scope_is_in_state(monkeypatch):
    assert _retrieve(monkeypatch, False, RESOLVED) == (PATIENT_WIDE, [])
    assert _retrieve(monkeypatch, False, None) == (PATIENT_WIDE, [])


@pytest.mark.parametrize("scope", [
    None,
    {"status": "none", "hadm_id": None, "retrieval_query": QUESTION},
    {"status": "unresolved", "hadm_id": None, "retrieval_query": QUESTION, "reason": "no admission matches"},
    {"status": "ambiguous", "hadm_id": None, "retrieval_query": QUESTION, "reason": "2 admissions match"},
])
def test_scoping_profile_stays_patient_wide_when_nothing_resolved(monkeypatch, scope):
    assert _retrieve(monkeypatch, True, scope) == (PATIENT_WIDE, [])


def test_scoping_profile_passes_the_resolved_admission_window_and_stripped_question(monkeypatch):
    call, windows = _retrieve(monkeypatch, True, RESOLVED)
    assert windows == [HADM]
    assert call == {"query": "what happened to her kidney function?", "subject_id": 7, "temporal_filter": "all",
                    "hadm_id": HADM, "stay_window": ("s", "e")}
