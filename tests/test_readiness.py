"""Readiness guards for the data profiles (data-foundation plan, E7 / decision G2).
No database: a small fake answers the handful of queries the checks make."""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.pool import NullPool

import src.api.app as api
from src.config import PROFILES
from src.storage import readiness as rd
from src.retrieval.index_provenance import RETRIEVER_CHUNK_TABLES
from src.storage.readiness import CHUNKS, RETRIEVER, STRUCTURED, DataSourceNotReady

BUILD = "build-0001"
ALL_TABLES = {"labevents_full", "d_icd_diagnoses", "d_icd_procedures", "structured_load_runs",
              "note_chunks_v2", "note_index_runs", "idx_labfull_subject_item_time"}
COUNTS = {"labevents_full": 9803443, "d_icd_diagnoses": 112107, "d_icd_procedures": 86423}


class _PgError(Exception):
    """What the driver raises, as far as the check looks: an error with a Postgres code."""
    def __init__(self, pgcode):
        super().__init__(pgcode)
        self.pgcode = pgcode


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value

    def first(self):
        return self.value


class _Db:
    """What exists, the latest structured load, and the index builds."""
    def __init__(self, tables=ALL_TABLES, load=("completed", COUNTS), empty=(), builds=None, build_chunks=(BUILD,)):
        self.tables, self.load, self.empty = set(tables), load, set(empty)
        self.builds = {BUILD: "completed"} if builds is None else builds
        self.build_chunks, self.queries, self.rollbacks = set(build_chunks), 0, 0

    @contextmanager
    def connect(self):
        yield self

    def rollback(self):
        self.rollbacks += 1

    def execute(self, stmt, params=None):
        self.queries += 1
        sql, params = str(stmt), params or {}
        named = (*rd.STRUCTURED_TABLES, "structured_load_runs")
        if "LEFT JOIN LATERAL" in sql:                    # the one-query structured state; Postgres refuses it when a table is missing
            if any(t not in self.tables for t in named):
                raise ProgrammingError(sql, {}, _PgError("42P01"))
            status, counts = self.load or (None, None)
            return _Result((status, counts, *(t not in self.empty for t in rd.STRUCTURED_TABLES),
                            "idx_labfull_subject_item_time" in self.tables))
        if sql.count("to_regclass") == len(named):        # which of the tables are there
            return _Result(tuple(t in self.tables for t in named))
        if "to_regclass(:name)" in sql:
            return _Result(params["name"] if params["name"] in self.tables else None)
        if "FROM note_index_runs" in sql:
            return _Result(self.builds.get(params["b"]))
        if "FROM note_chunks_v2" in sql:
            return _Result(params["b"] in self.build_chunks)
        raise AssertionError(f"unexpected query: {sql}")


V2_CAPABLE = ("note_chunks", "note_chunks_v2")      # a retriever that can also query the v2 table (what E14 delivers)


def _problems(profile, db, build_id=BUILD, supported=V2_CAPABLE):
    return rd.problems(db.connect, PROFILES[profile], build_id, supported)


# --- what each profile depends on -----------------------------------------------------------
def test_profiles_depend_on_exactly_what_the_plan_says():
    needs = {p: (rd.needs(STRUCTURED, s), rd.needs(CHUNKS, s)) for p, s in PROFILES.items()}
    assert needs == {"control": (False, False), "scoped": (False, False), "structured": (True, False), "v2": (True, True)}


@pytest.mark.parametrize("profile", ["control", "scoped"])
def test_control_and_scoped_are_ready_whatever_the_new_data_looks_like(profile):
    broken = _Db(tables=set(), load=("failed", {}), builds={})
    assert _problems(profile, broken) == [] and broken.queries == 0          # not even looked at
    rd.require(STRUCTURED, broken.connect, PROFILES[profile])
    rd.require(CHUNKS, broken.connect, PROFILES[profile])
    assert broken.queries == 0


# --- structured load -------------------------------------------------------------------------
def test_completed_structured_load_may_serve():
    assert _problems("structured", _Db()) == []
    rd.require(STRUCTURED, _Db().connect, PROFILES["structured"])            # does not raise


STRUCTURED_REFUSED = [
    ("missing load", _Db(load=None), "no structured load has been run"),
    ("running load", _Db(load=("running", {})), "the latest structured load is running"),
    ("failed load", _Db(load=("failed", {})), "the latest structured load is failed"),
    ("table missing despite a completed run", _Db(tables=ALL_TABLES - {"labevents_full"}), "missing table(s): labevents_full"),
    ("no run table at all", _Db(tables=ALL_TABLES - {"structured_load_runs"}), "missing table(s): structured_load_runs"),
    ("completed but a table is empty", _Db(empty={"d_icd_diagnoses"}), "d_icd_diagnoses is empty although the load recorded 112107 rows"),
    ("completed but recorded no rows", _Db(load=("completed", {**COUNTS, "labevents_full": 0})), "the completed load recorded no rows for labevents_full"),
    ("index missing", _Db(tables=ALL_TABLES - {"idx_labfull_subject_item_time"}), "index idx_labfull_subject_item_time is missing"),
]


@pytest.mark.parametrize("name,db,reason", STRUCTURED_REFUSED, ids=[c[0] for c in STRUCTURED_REFUSED])
def test_structured_is_refused_unless_the_load_is_complete_and_present(name, db, reason):
    assert _problems("structured", db) == [{"component": STRUCTURED, "reason": reason}]
    with pytest.raises(DataSourceNotReady) as e:
        rd.require(STRUCTURED, db.connect, PROFILES["structured"])
    assert (e.value.component, e.value.reason) == (STRUCTURED, reason)


# --- one query, the same answers ---------------------------------------------------------------
def test_a_usable_structured_load_is_confirmed_in_one_statement():
    db = _Db()
    assert rd.structured_problem(db) is None and (db.queries, db.rollbacks) == (1, 0)
    db = _Db()
    rd.require(STRUCTURED, db.connect, PROFILES["structured"])
    assert db.queries == 1
    for name, refused, reason in STRUCTURED_REFUSED:       # every refusal that is not a missing table: still one statement
        if not reason.startswith("missing table"):
            refused.queries = 0
            assert rd.structured_problem(refused) == reason and refused.queries == 1, name


def test_a_missing_table_takes_one_more_statement_to_name_every_missing_table():
    db = _Db(tables=ALL_TABLES - {"d_icd_procedures", "labevents_full", "structured_load_runs"})
    assert rd.structured_problem(db) == "missing table(s): labevents_full, d_icd_procedures, structured_load_runs"
    assert (db.queries, db.rollbacks) == (2, 1)            # the refused statement is rolled back before the second one
    v2 = _Db(tables=ALL_TABLES - {"labevents_full"})       # the same connection then checks the build, as before
    assert [p["component"] for p in _problems("v2", v2)] == [STRUCTURED]


@pytest.mark.parametrize("pgcode", ["42501", "57014", None])       # permission denied, cancelled, no code at all
def test_any_other_database_error_is_raised_and_never_read_as_ready(pgcode):
    class _Broken(_Db):
        def execute(self, stmt, params=None):
            raise ProgrammingError(str(stmt), {}, _PgError(pgcode))
    db = _Broken()
    with pytest.raises(ProgrammingError):
        rd.structured_problem(db)
    with pytest.raises(ProgrammingError):
        rd.require(STRUCTURED, db.connect, PROFILES["structured"])
    assert db.rollbacks == 0


def test_a_table_that_appears_between_the_two_statements_is_an_error_not_a_pass():
    class _Racing(_Db):
        def execute(self, stmt, params=None):
            if "LEFT JOIN LATERAL" in str(stmt):
                raise ProgrammingError(str(stmt), {}, _PgError("42P01"))
            return super().execute(stmt, params)
    with pytest.raises(ProgrammingError):
        rd.structured_problem(_Racing())


# The same cases against real Postgres: session-temporary tables in the server's
# maintenance database, seen through a search path that hides everything else.
PG_CASES = [("ready", dict(), None)] + [
    (name, spec, reason) for (name, _, reason), spec in zip(STRUCTURED_REFUSED, [
        dict(load=None), dict(load=("running", {})), dict(load=("failed", {})),
        dict(tables=ALL_TABLES - {"labevents_full"}), dict(tables=ALL_TABLES - {"structured_load_runs"}),
        dict(empty={"d_icd_diagnoses"}), dict(load=("completed", {**COUNTS, "labevents_full": 0})),
        dict(tables=ALL_TABLES - {"idx_labfull_subject_item_time"})])] + [
    ("several tables missing", dict(tables=ALL_TABLES - {"labevents_full", "d_icd_procedures"}),
     "missing table(s): labevents_full, d_icd_procedures"),
    ("an older completed load does not excuse a newer failed one", dict(load=("failed", {}), older=("completed", COUNTS)),
     "the latest structured load is failed"),
]


@pytest.fixture
def pg():
    from src import storage
    engine = create_engine(make_url(storage.DATABASE_URL).set(database="postgres"), poolclass=NullPool,
                           connect_args={"connect_timeout": 3})
    try:
        conn = engine.connect()
    except OperationalError:
        pytest.skip("Postgres not reachable")
    try:
        yield conn
    finally:
        conn.close()                                       # ends the session: its temporary tables go with it
        engine.dispose()


@pytest.mark.parametrize("name,spec,reason", PG_CASES, ids=[c[0] for c in PG_CASES])
def test_the_real_query_gives_the_same_answer_on_postgres(pg, name, spec, reason):
    tables, load, empty = spec.get("tables", ALL_TABLES), spec.get("load", ("completed", COUNTS)), spec.get("empty", set())
    pg.execute(text("SET search_path TO pg_temp"))
    for table in rd.STRUCTURED_TABLES:
        if table in tables:
            pg.execute(text(f"CREATE TEMP TABLE {table} (subject_id int, itemid int, charttime timestamp)"))
            if table not in empty:
                pg.execute(text(f"INSERT INTO {table} VALUES (1, 1, now())"))
    if "labevents_full" in tables and "idx_labfull_subject_item_time" in tables:
        pg.execute(text("CREATE INDEX idx_labfull_subject_item_time ON labevents_full (subject_id, itemid, charttime)"))
    if "structured_load_runs" in tables:
        pg.execute(text("CREATE TEMP TABLE structured_load_runs (status text, row_counts jsonb, started_at timestamptz)"))
        runs = [(spec["older"], "2026-01-01")] if "older" in spec else []
        for (status, counts), started in runs + ([(load, "2026-02-01")] if load else []):
            pg.execute(text("INSERT INTO structured_load_runs VALUES (:s, CAST(:c AS jsonb), CAST(:t AS timestamptz))"),
                       {"s": status, "c": __import__("json").dumps(counts), "t": started})
    pg.commit()
    assert rd.structured_problem(pg) == reason
    assert rd.structured_problem(pg) == reason             # and the connection is still usable afterwards


def test_structured_does_not_need_a_v2_build():
    no_v2 = _Db(tables=ALL_TABLES - {"note_chunks_v2"}, builds={})
    assert rd.problems(no_v2.connect, PROFILES["structured"], None) == []
    rd.require(CHUNKS, no_v2.connect, PROFILES["structured"], None)          # not this profile's concern


# --- v2 build ---------------------------------------------------------------------------------
def test_completed_selected_build_is_acceptable():
    assert rd.chunk_build_problem(_Db(), BUILD) is None
    assert _problems("v2", _Db()) == []                                        # with a retriever that can query it
    rd.require(CHUNKS, _Db().connect, PROFILES["v2"], BUILD, V2_CAPABLE)


V2_REFUSED = [
    ("no build selected", _Db(), None, "no build selected: set LUMEN_CHUNK_BUILD to a completed build id"),
    ("missing build", _Db(builds={}), BUILD, f"selected build {BUILD} does not exist"),
    ("running build", _Db(builds={BUILD: "running"}), BUILD, f"selected build {BUILD} is running"),
    ("failed build", _Db(builds={BUILD: "failed"}), BUILD, f"selected build {BUILD} is failed"),
    ("completed run but no chunks", _Db(build_chunks=()), BUILD, f"selected build {BUILD} is completed but has no chunks"),
    ("completed run but table missing", _Db(tables=ALL_TABLES - {"note_chunks_v2"}), BUILD, "missing table(s): note_chunks_v2"),
    # another build being complete does not make the selected one usable
    ("a different build is the completed one", _Db(builds={"other": "completed"}, build_chunks=("other",)), BUILD,
     f"selected build {BUILD} does not exist"),
]


@pytest.mark.parametrize("name,db,build_id,reason", V2_REFUSED, ids=[c[0] for c in V2_REFUSED])
def test_v2_is_refused_unless_the_selected_build_is_complete_and_present(name, db, build_id, reason):
    assert _problems("v2", db, build_id) == [{"component": CHUNKS, "reason": reason}]
    with pytest.raises(DataSourceNotReady) as e:
        rd.require(CHUNKS, db.connect, PROFILES["v2"], build_id, V2_CAPABLE)
    assert (e.value.component, e.value.reason) == (CHUNKS, reason)


def test_v2_also_needs_the_structured_load_and_reports_both():
    both = _problems("v2", _Db(load=("running", {}), builds={BUILD: "failed"}))
    assert [p["component"] for p in both] == [STRUCTURED, CHUNKS]
    every = rd.problems(_Db(load=("running", {}), builds={BUILD: "failed"}).connect, PROFILES["v2"], BUILD, ("note_chunks",))
    assert [p["component"] for p in every] == [STRUCTURED, CHUNKS, RETRIEVER]  # a v1-only retriever: all three are reported


def test_v2_profile_without_a_selected_build_refuses_to_start():
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    def start(**env):
        clean = {k: v for k, v in os.environ.items() if k not in ("LUMEN_DATA_PROFILE", "LUMEN_CHUNK_BUILD")}
        return subprocess.run([sys.executable, "-c", "import src.config as c; print(c.DATA_PROFILE, c.CHUNK_BUILD)"],
                              cwd=root, env={**clean, **env}, capture_output=True, text=True)
    refused = start(LUMEN_DATA_PROFILE="v2")
    assert refused.returncode != 0 and "LUMEN_DATA_PROFILE=v2 needs LUMEN_CHUNK_BUILD" in refused.stderr
    assert start(LUMEN_DATA_PROFILE="v2", LUMEN_CHUNK_BUILD=BUILD).stdout.split() == ["v2", BUILD]
    assert start(LUMEN_DATA_PROFILE="structured").stdout.split() == ["structured", "None"]      # no build needed
    assert start().stdout.split() == ["control", "None"]


# --- the retriever must actually be able to query the profile's chunk table -------------------
MISMATCH = "the profile reads note_chunks_v2, but the retriever queries note_chunks only"


def test_v2_is_refused_while_the_retriever_still_reads_note_chunks_even_with_a_perfect_build():
    perfect = _Db()                                                            # completed selected build, populated table
    assert rd.chunk_build_problem(perfect, BUILD) is None and rd.structured_problem(perfect) is None
    only_v1 = ("note_chunks",)                                                 # a retriever that cannot query the v2 table
    assert rd.problems(perfect.connect, PROFILES["v2"], BUILD, only_v1) == [{"component": RETRIEVER, "reason": MISMATCH}]
    with pytest.raises(DataSourceNotReady) as e:
        rd.require(CHUNKS, perfect.connect, PROFILES["v2"], BUILD, only_v1)
    assert (e.value.component, e.value.reason) == (RETRIEVER, MISMATCH)
    # the same check passes by itself once the retriever can query the table: nothing was removed to get here
    assert rd.problems(perfect.connect, PROFILES["v2"], BUILD, V2_CAPABLE) == []
    assert RETRIEVER_CHUNK_TABLES == V2_CAPABLE                                # which is what the retriever is, since E14
    assert rd.problems(perfect.connect, PROFILES["v2"], BUILD) == []


@pytest.mark.parametrize("profile", ["control", "scoped", "structured"])
def test_profiles_on_note_chunks_are_unaffected_by_the_retriever_check(profile):
    assert rd.retriever_problem(PROFILES[profile]) is None
    db = _Db()
    assert rd.problems(db.connect, PROFILES[profile], None) == []
    rd.require(CHUNKS, db.connect, PROFILES[profile], None)                    # the real retriever table list; does not raise
    assert profile == "structured" or db.queries == 0


def test_retriever_table_list_matches_the_tables_its_queries_name():
    import re
    from pathlib import Path

    import src.retrieval.hybrid_retriever_v2 as retriever
    source = Path(retriever.__file__).read_text(encoding="utf-8")
    named = set(re.findall(r"(?:FROM|JOIN)\s+(note_chunks\w*)", source))
    assert named == set(RETRIEVER_CHUNK_TABLES)                                # the list says what the SQL does, no more and no less


# --- a readiness failure is an error, never a quiet fallback ------------------------------------
def _not_ready(component):
    def raiser(*a, **k):
        raise DataSourceNotReady(component, "the latest structured load is running")
    return raiser


def test_lab_lookup_does_not_swallow_a_readiness_failure(monkeypatch):
    import src.agents.graph as g

    class _Resolver:
        labels = []

        def match(self, query):
            return [50912], ["creatinine"]
        fetch = staticmethod(_not_ready(STRUCTURED))
    monkeypatch.setattr(g, "get_lab_resolver", lambda: _Resolver())
    state = {"query": "most recent creatinine", "subject_id": 7, "temporal_mode": "latest"}
    with pytest.raises(DataSourceNotReady):
        g.lab_lookup(state)
    # an ordinary failure of the lookup is still a miss that falls through to retrieval, as before
    _Resolver.fetch = staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert "lab_evidence" not in g.lab_lookup(state)


def test_lab_query_checks_readiness_before_it_reads(monkeypatch):
    from src.generation import lab_query
    monkeypatch.setattr(lab_query.readiness, "require", _not_ready(STRUCTURED))
    resolver = lab_query.LabResolver.__new__(lab_query.LabResolver)           # no database: fetch must stop at the guard
    with pytest.raises(DataSourceNotReady):
        resolver.fetch(7, [50912])


def test_retriever_refuses_an_unready_index_instead_of_returning_no_evidence(monkeypatch):
    from src.retrieval import hybrid_retriever_v2 as r
    monkeypatch.setattr(r.readiness, "require", _not_ready(CHUNKS))
    with pytest.raises(DataSourceNotReady):
        r.HybridRetriever.search(object(), query="chest pain", subject_id=7)  # raised before anything is searched


# --- what the API says --------------------------------------------------------------------------
client = TestClient(api.app)


def _ready(monkeypatch, problems):
    monkeypatch.setattr(api, "_check_database", lambda: {"database": "ok", "schema": "ok", "extension": "ok",
                                                         "corpus": "ok", "ingestion": "ok", "index": "ok"})
    monkeypatch.setattr(api, "_check_ollama", lambda: {"ollama": "ok", "model": "ok"})
    monkeypatch.setattr(api, "_check_retrieval_models", lambda: {"retrieval_models": "ok", "model_problems": []})
    monkeypatch.setattr(api.readiness, "problems", lambda: problems)
    return client.get("/ready")


def test_ready_is_503_and_names_the_failed_dependency(monkeypatch):
    problem = {"component": STRUCTURED, "reason": "the latest structured load is running"}
    r = _ready(monkeypatch, [problem])
    assert r.status_code == 503 and r.json()["status"] == "not_ready"
    assert r.json()["dependencies"]["data_profile"] == "structured_load_not_ready"
    assert r.json()["data_profile_problems"] == [problem]
    ok = _ready(monkeypatch, [])
    assert ok.status_code == 200 and ok.json()["dependencies"]["data_profile"] == "ok" and ok.json()["data_profile_problems"] == []


def test_ask_returns_503_with_the_component_and_no_answer(monkeypatch):
    monkeypatch.setattr(api, "_ensure_subject", lambda sid: None)
    monkeypatch.setattr(api, "_run_ask", _not_ready(STRUCTURED))
    r = client.post("/ask", json={"subject_id": 90000001, "query": "most recent creatinine"})
    body = r.json()
    assert r.status_code == 503 and body["error"] == "data_source_not_ready"
    assert body["detail"] == {"component": STRUCTURED, "reason": "the latest structured load is running"}
    assert "answer" not in body
