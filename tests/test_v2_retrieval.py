"""v2 retrieval (data-foundation plan, E14). No database, no models: the engine is
a recorder that returns canned rows, so what is asserted is the SQL each search
sends and what the code does with the rows. Real Postgres behaviour is covered
by the live smoke check against the pilot build."""
import re
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

import src.retrieval.hybrid_retriever_v2 as hr
from src.config import PROFILES
from src.retrieval.index_provenance import RETRIEVER_CHUNK_TABLES
from src.storage import readiness as rd

BUILD, OTHER_BUILD, SID, HADM = "build-0001", "build-0002", 7, 22595853
WINDOW = ("2180-05-06 19:17:00", "2180-05-07 17:15:00")


class _Engine:
    def __init__(self, rows=()):
        self.rows, self.calls = list(rows), []

    @contextmanager
    def connect(self):
        yield self

    def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.calls.append((sql, dict(params or {})))
        if "plainto_tsquery" in sql:
            return SimpleNamespace(scalar=lambda: "'chest' & 'pain'")
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: [dict(r) for r in self.rows]))


@pytest.fixture
def db(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(hr, "engine", engine)
    return engine


def _last(db):
    return db.calls[-1]


# --- every v2 query is pinned to the build and the patient ------------------------------------------
def test_lexical_search_reads_the_stored_columns_of_one_build_and_one_patient(db):
    db.rows = [_row(chunk_id=5, bm25_score=4.0), _row(chunk_id=9, bm25_score=1.0)]
    out = hr.lexical_search_v2("chest pain", SID, BUILD, top_n=60)
    sql, params = _last(db)
    assert "FROM note_chunks_v2 nc WHERE nc.build_id = :build_id AND nc.subject_id = :subject_id" in sql
    assert (params["build_id"], params["subject_id"], params["terms"], params["top_n"]) == (BUILD, SID, ["chest", "pain"], 60)
    assert "nc.text_search" in sql and "nc.section_search" in sql and "b.section_search @@ w.tq" in sql   # stored, not rebuilt per row
    assert "to_tsvector" not in sql and "chunk_search_labels" not in sql and "clinical_notes" not in sql
    assert "nc.embed" not in sql and "token_count >=" not in sql          # lexical-only chunks take part
    assert "ORDER BY bm25_score DESC, b.chunk_id ASC" in sql
    assert [r["bm25_score"] for r in out] == [1.0, 0.0]                   # normalised for display, order kept


def test_vector_search_is_an_exact_scan_of_embedded_chunks_only(db):
    db.rows = [_row(chunk_id=5, vector_score=0.9)]
    out = hr.vector_search_v2(np.array([0.1, 0.2]), SID, BUILD, top_n=60)
    assert [(r["chunk_id"], r["vector_score"]) for r in out] == [(5, 0.9)]
    sql, params = _last(db)
    assert "FROM note_chunks_v2 nc WHERE nc.build_id = :build_id AND nc.subject_id = :subject_id AND nc.embed AND nc.embedding IS NOT NULL" in sql
    assert "ORDER BY nc.embedding <=> CAST(:query_vec AS vector), nc.chunk_id LIMIT :top_n" in sql
    assert (params["build_id"], params["subject_id"]) == (BUILD, SID) and params["query_vec"] == "[0.1,0.2]"
    assert len(db.calls) == 1 and "hnsw" not in sql.lower()               # no index hint, no ef_search
    assert hr.vector_search_v2(np.array([np.nan, 0.2]), SID, BUILD) == [] and len(db.calls) == 1


@pytest.mark.parametrize("search", [
    lambda **k: hr.lexical_search_v2("chest pain", **k),
    lambda **k: hr.vector_search_v2(np.array([0.1, 0.2]), **k),
])
def test_admission_scope_on_v2_keeps_the_e4_rule_with_the_chunks_own_chart_time(db, search):
    search(subject_id=SID, build_id=BUILD, hadm_id=HADM, stay_window=WINDOW)
    sql, params = _last(db)
    assert ("AND (nc.hadm_id = :hadm_id OR (nc.hadm_id IS NULL AND nc.charttime BETWEEN :stay_start AND :stay_end))") in sql
    assert "cn." not in sql                                               # no join: the time is on the chunk row
    assert (params["hadm_id"], params["stay_start"], params["stay_end"]) == (HADM, *WINDOW)
    assert (params["build_id"], params["subject_id"]) == (BUILD, SID)     # scope narrows the patient's build; it never replaces it
    search(subject_id=SID, build_id=BUILD)
    assert "hadm_id =" not in _last(db)[0] and "hadm_id" not in _last(db)[1]            # no scope: the whole patient, this build


@pytest.mark.parametrize("build_id,subject_id", [(None, SID), ("", SID), (BUILD, None)])
def test_a_v2_search_without_a_build_or_a_patient_is_refused(db, build_id, subject_id):
    with pytest.raises(ValueError, match="one build and one patient"):
        hr.lexical_search_v2("chest pain", subject_id, build_id)
    with pytest.raises(ValueError, match="one build and one patient"):
        hr.vector_search_v2(np.array([0.1, 0.2]), subject_id, build_id)
    assert not any("note_chunks_v2" in sql for sql, _ in db.calls)        # nothing unpinned ever reached the table


# --- section as parent -----------------------------------------------------------------------------------
def _result(chunk_id, text="x", tokens=100):
    return hr.RetrievalResult(chunk_id=chunk_id, note_id=1, subject_id=SID, hadm_id=HADM, note_type="discharge",
                              chunk_index=0, chunk_text=text, token_count=tokens)


def _section(match_id, parts):
    return [{"match_id": match_id, "chunk_id": cid, "chunk_ord": ordn, "chunk_text": text, "token_count": tok}
            for cid, ordn, text, tok in parts]


def test_matches_are_widened_to_their_own_section_in_one_query(db):
    db.rows = (_section(11, [(10, 0, "A0 ", 200), (11, 1, "A1 ", 200), (12, 2, "A2 ", 150), (13, 3, "A3 ", 300)])
               + _section(20, [(20, 0, "B0 ", 90)])
               + _section(31, [(30, 0, "C0 ", 500), (31, 1, "C1 ", 450)]))
    results = [_result(11), _result(20), _result(31), _result(99, text="orphan")]
    out = hr.expand_context_v2(results, BUILD, max_context_tokens=600)
    assert len(db.calls) == 1                                             # one query for four results, not four
    sql, params = _last(db)
    assert params == {"build_id": BUILD, "ids": [11, 20, 31, 99]}
    assert "s.build_id = m.build_id AND s.mimic_note_id = m.mimic_note_id AND s.section_ord = m.section_ord" in sql
    assert "WHERE m.build_id = :build_id AND m.chunk_id = ANY(:ids)" in sql
    by = {r.chunk_id: r for r in out}
    # 11: itself (200), then nearest first: 10 (200) and 12 (150) fit in 600; 13 (300) does not. In note order.
    assert (by[11].context_text, by[11].token_count) == ("A0 A1 A2 ", 550)
    assert (by[20].context_text, by[20].token_count) == ("B0 ", 90)       # a one-chunk section is just itself
    assert (by[31].context_text, by[31].token_count) == ("C1 ", 450)      # the match stays even when its neighbour cannot fit
    assert by[99].context_text == "orphan"                                # nothing returned for it: never another chunk in its place
    assert all(r.token_count <= 600 for r in out)


def test_expansion_of_nothing_asks_nothing(db):
    assert hr.expand_context_v2([], BUILD) == [] and db.calls == []


# --- which table a search uses ----------------------------------------------------------------------------
def _search(monkeypatch, table):
    used = []
    monkeypatch.setattr(hr, "CHUNK_TABLE", table)
    monkeypatch.setattr(hr, "CHUNK_BUILD", BUILD)
    monkeypatch.setattr(hr.readiness, "require", lambda component: used.append(("require", component)))
    for name in ("bm25_search", "vector_search", "expand_context", "lexical_search_v2", "vector_search_v2", "expand_context_v2"):
        def fake(*a, _name=name, **k):
            used.append((_name, a, k))
            return a[0] if "expand" in _name else []
        monkeypatch.setattr(hr, name, fake)
    me = SimpleNamespace(bm25_top_n=60, vector_top_n=60, use_query_expansion=False, min_chunk_tokens=40, max_per_note=3,
                         rerank_candidates=20, use_context_window=True, context_window=1, reranker=None,
                         bm25_weight=1.5, vector_weight=0.75, overlap_bonus=0.0,
                         embedder=SimpleNamespace(embed_query=lambda q: np.array([0.1, 0.2])))
    hr.HybridRetriever.search(me, query="chest pain", subject_id=SID, hadm_id=HADM, stay_window=WINDOW, temporal_filter="all", top_k=5)
    return used


def test_v2_profile_searches_the_selected_build_of_note_chunks_v2(monkeypatch):
    used = _search(monkeypatch, "note_chunks_v2")
    assert [u[0] for u in used] == ["require", "lexical_search_v2", "vector_search_v2", "expand_context_v2"]
    assert used[0] == ("require", rd.CHUNKS)                              # the readiness guard still runs first
    lexical, vector, expand = used[1], used[2], used[3]
    assert lexical[1] == ("chest pain", SID, BUILD) and lexical[2]["hadm_id"] == HADM and lexical[2]["stay_window"] == WINDOW
    assert vector[1][1:] == (SID, BUILD) and vector[2]["hadm_id"] == HADM and vector[2]["stay_window"] == WINDOW
    assert expand[1][1] == BUILD and expand[2] == {"max_context_tokens": 600}


@pytest.mark.parametrize("profile", ["control", "scoped", "structured"])
def test_other_profiles_make_exactly_the_calls_they_made_before(monkeypatch, profile):
    used = _search(monkeypatch, PROFILES[profile]["chunk_table"])
    assert PROFILES[profile]["chunk_table"] == "note_chunks"
    assert [u[0] for u in used] == ["require", "bm25_search", "vector_search", "expand_context"]
    assert used[1][2] == {"query": "chest pain", "expansions": [], "subject_id": SID, "hadm_id": HADM, "note_type": None,
                          "top_n": 60, "min_tokens": 40, "stay_window": WINDOW}
    assert set(used[2][2]) == {"query_embedding", "subject_id", "hadm_id", "note_type", "top_n", "min_tokens", "stay_window"}
    assert used[3][2] == {"window": 1, "max_context_tokens": 600}


def test_profiles_select_their_table_and_only_v2_selects_the_v2_table():
    assert {p: s["chunk_table"] for p, s in PROFILES.items()} == {
        "control": "note_chunks", "scoped": "note_chunks", "structured": "note_chunks", "v2": "note_chunks_v2"}
    assert hr.CHUNK_TABLE == "note_chunks" and hr.V2_TABLE == "note_chunks_v2"           # this process runs the control profile
    assert RETRIEVER_CHUNK_TABLES == ("note_chunks", "note_chunks_v2")


def test_control_search_functions_still_name_only_note_chunks():
    import inspect
    for fn in (hr._patient_lexical_search, hr._strict_lexical_search, hr.bm25_search, hr.vector_search,
               hr.fetch_adjacent_chunks, hr.expand_context):
        assert "note_chunks_v2" not in inspect.getsource(fn), fn.__name__
    for fn in (hr.lexical_search_v2, hr.vector_search_v2, hr.expand_context_v2):
        source = inspect.getsource(fn)
        assert not re.search(r"\bnote_chunks\b(?!_v2)", source) and "clinical_notes" not in source, fn.__name__


# --- readiness: satisfied only by a completed selected build AND a retriever that can read it -------------
class _Ready:
    def __init__(self, build_status="completed", chunks=True):
        self.build_status, self.chunks = build_status, chunks

    @contextmanager
    def connect(self):
        yield self

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "to_regclass" in sql:
            return SimpleNamespace(scalar=lambda: "x")
        if "FROM structured_load_runs" in sql:
            return SimpleNamespace(first=lambda: ("completed", {"labevents_full": 1, "d_icd_diagnoses": 1, "d_icd_procedures": 1}))
        if "FROM note_index_runs" in sql:
            return SimpleNamespace(scalar=lambda: self.build_status)
        if "FROM note_chunks_v2" in sql:
            return SimpleNamespace(scalar=lambda: self.chunks)
        return SimpleNamespace(scalar=lambda: True)


def test_v2_is_ready_with_a_completed_selected_build_now_that_the_retriever_reads_it():
    assert rd.problems(_Ready().connect, PROFILES["v2"], BUILD) == []
    rd.require(rd.CHUNKS, _Ready().connect, PROFILES["v2"], BUILD)       # does not raise


@pytest.mark.parametrize("db,reason", [
    (_Ready(build_status="running"), "is running"), (_Ready(build_status="failed"), "is failed"),
    (_Ready(build_status=None), "does not exist"), (_Ready(chunks=False), "has no chunks"),
])
def test_v2_is_still_refused_when_the_selected_build_is_not_usable(db, reason):
    (problem,) = rd.problems(db.connect, PROFILES["v2"], BUILD)
    assert problem["component"] == rd.CHUNKS and reason in problem["reason"]
    with pytest.raises(rd.DataSourceNotReady):
        rd.require(rd.CHUNKS, db.connect, PROFILES["v2"], BUILD)
    assert rd.problems(_Ready().connect, PROFILES["v2"], None)[0]["reason"].startswith("no build selected")
    assert rd.problems(_Ready().connect, PROFILES["v2"], BUILD, ("note_chunks",))[0]["component"] == rd.RETRIEVER


def test_api_subject_check_follows_the_profile(monkeypatch):
    import src.api.app as api
    asked = []

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params):
            asked.append((" ".join(str(stmt).split()), params))
            return SimpleNamespace(scalar=lambda: params.get("s") == SID)
    monkeypatch.setattr(api.storage, "engine", SimpleNamespace(connect=lambda: _Conn()))
    monkeypatch.setattr(api, "PROFILE_SETTINGS", PROFILES["v2"])
    monkeypatch.setattr(api, "CHUNK_BUILD", BUILD)
    api._ensure_subject(SID)
    assert asked[-1] == ("SELECT EXISTS (SELECT 1 FROM note_chunks_v2 WHERE build_id = :b AND subject_id = :s)", {"b": BUILD, "s": SID})
    with pytest.raises(api.SubjectNotFound):                             # a patient outside the selected build is not searchable
        api._ensure_subject(SID + 1)
    monkeypatch.setattr(api, "PROFILE_SETTINGS", PROFILES["control"])
    api._ensure_subject(SID)
    assert asked[-1] == ("SELECT EXISTS (SELECT 1 FROM note_chunks WHERE subject_id = :s)", {"s": SID})


# --- source identity: chunk_id is a handle, provenance is the identity ----------------------------

def _row(build=BUILD, chunk_id=5, **over):
    return {"chunk_id": chunk_id, "note_id": 900, "subject_id": SID, "hadm_id": HADM, "note_type": "discharge",
            "chunk_index": 2001, "chunk_text": "chest pain", "token_count": 3, "charttime": None,
            "build_id": build, "mimic_note_id": "10000032-DS-21", "section_ord": 2, "chunk_ord": 1,
            "char_start": 120, "char_end": 131, "bm25_score": 1.0, "vector_score": 0.5, **over}


@pytest.mark.parametrize("search", ["lexical", "vector"])
def test_v2_results_carry_stable_provenance(db, search):
    db.rows = [_row()]
    rows = (hr.lexical_search_v2("chest pain", SID, BUILD) if search == "lexical"
            else hr.vector_search_v2(np.array([0.1, 0.2]), SID, BUILD))
    assert all(f"nc.{c}" in _last(db)[0] for c in ("build_id", "mimic_note_id", "section_ord", "chunk_ord", "char_start", "char_end"))
    assert rows[0]["provenance"] == {
        "source_id": f"note_chunks_v2:{BUILD}:10000032-DS-21:2:1", "data_profile": hr.DATA_PROFILE,
        "table": "note_chunks_v2", "build_id": BUILD, "mimic_note_id": "10000032-DS-21", "section_ord": 2,
        "chunk_ord": 1, "start_offset": 120, "end_offset": 131, "chunk_id": 5}
    fused = hr.reciprocal_rank_fusion(rows, rows)
    assert fused[0].provenance == rows[0]["provenance"] and fused[0].chunk_id == 5


def test_control_queries_do_not_read_v2_identity_columns():
    import inspect
    for fn in (hr._patient_lexical_search, hr._strict_lexical_search, hr.bm25_search, hr.vector_search, hr.expand_context):
        source = inspect.getsource(fn)
        assert not any(c in source for c in ("mimic_note_id", "build_id", "section_ord", "char_start", "provenance")), fn.__name__


def test_the_same_chunk_id_in_another_build_or_representation_is_a_different_source():
    from src.agents.graph import _to_evidence
    from src.api.schemas import Source
    control = hr.reciprocal_rank_fusion([{k: v for k, v in _row().items() if k in (
        "chunk_id", "note_id", "subject_id", "hadm_id", "note_type", "chunk_index", "chunk_text", "token_count", "charttime")}], [])[0]
    first, second = (hr.reciprocal_rank_fusion([dict(_row(b), provenance=hr.v2_provenance(_row(b)))], [])[0] for b in (BUILD, OTHER_BUILD))
    assert control.chunk_id == first.chunk_id == second.chunk_id == 5       # the handle collides, by design
    assert control.provenance is None
    assert first.provenance["source_id"] != second.provenance["source_id"]
    assert len({first.provenance["build_id"], second.provenance["build_id"]}) == 2

    evidence = _to_evidence([control, first, second], "S", "note")
    assert "provenance" not in evidence[0]                                   # control evidence is exactly what it was
    assert set(evidence[0]) == {"chunk_id", "note_id", "subject_id", "hadm_id", "chunk_index", "source_type", "text",
                                "charttime", "note_type", "score", "label"}
    assert evidence[1]["provenance"]["source_id"] != evidence[2]["provenance"]["source_id"]

    def source(e):
        return Source(label=e["label"], chunk_id=e["chunk_id"], source_type=e["source_type"], note_id=e["note_id"],
                      subject_id=e["subject_id"], hadm_id=e["hadm_id"], chunk_index=e["chunk_index"],
                      note_type=e["note_type"], charttime=e["charttime"], provenance=e.get("provenance")).model_dump(mode="json")
    out = [source(e) for e in evidence]
    assert "provenance" not in out[0]                                        # and so is the control response
    assert set(out[0]) == {"label", "chunk_id", "source_type", "note_id", "subject_id", "hadm_id", "chunk_index", "note_type", "charttime"}
    assert out[1]["provenance"]["table"] == "note_chunks_v2" and out[1]["provenance"]["start_offset"] == 120
    assert out[1]["provenance"] != out[2]["provenance"] and out[1]["chunk_id"] == out[2]["chunk_id"]


# --- query expansion: refused for v2, untouched elsewhere ---------------------------------------

def test_v2_with_query_expansion_is_refused_at_construction_and_at_search(monkeypatch):
    monkeypatch.setattr(hr, "CHUNK_TABLE", "note_chunks_v2")
    with pytest.raises(rd.DataSourceNotReady, match="query expansion") as err:
        hr.HybridRetriever(use_query_expansion=True)                         # before any model is loaded
    assert err.value.component == rd.RETRIEVER
    monkeypatch.setattr(hr.readiness, "require", lambda component: None)
    called = []
    monkeypatch.setattr(hr, "lexical_search_v2", lambda *a, **k: called.append(a))
    with pytest.raises(rd.DataSourceNotReady, match="query expansion"):
        hr.HybridRetriever.search(SimpleNamespace(use_query_expansion=True), query="chest pain", subject_id=SID)
    assert called == []                                                      # nothing was searched


@pytest.mark.parametrize("profile", ["control", "scoped", "structured"])
def test_other_profiles_keep_query_expansion_as_it_was(profile):
    assert rd.expansion_problem(True, PROFILES[profile]) is None
    assert rd.expansion_problem(False, PROFILES["v2"]) is None
    assert "query expansion" in rd.expansion_problem(True, PROFILES["v2"])


def test_ready_reports_the_unsupported_combination(monkeypatch):
    assert not any("expansion" in p["reason"] for p in rd.problems(_Ready().connect, PROFILES["v2"], BUILD))
    monkeypatch.setattr(rd, "QUERY_EXPANSION", True)
    monkeypatch.setattr(rd.expansion_problem, "__defaults__", (True, None))
    found = rd.problems(_Ready().connect, PROFILES["v2"], BUILD)
    assert [p["component"] for p in found] == [rd.RETRIEVER] and "query expansion" in found[0]["reason"]
    assert rd.problems(settings=PROFILES["control"]) == []
