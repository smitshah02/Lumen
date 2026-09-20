"""
Demo-plane safety smoke test.

Regression cover for: `demo_smoke_test.py safety` crashed on a demo-only cloud
Pod because cmd_safety() opened the research database unconditionally, and that
database does not exist there by design.

No Postgres: src.storage and the engine are faked, so these tests assert which
databases the safety command *opens* and what it concludes — which is the whole
bug — rather than anything about real data.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import demo_smoke_test as smoke  # noqa: E402


# --- fakes -----------------------------------------------------------------
class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    """Answers by matching a fragment of the SQL, so the tests do not depend on
    the exact statements the command happens to use."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.executed: list[str] = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.executed.append(sql)
        for fragment, value in self.answers.items():
            if fragment in sql:
                return _FakeResult(value)
        return _FakeResult(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self, url: str, answers: dict):
        self.url = make_url(url)
        self.answers = answers
        self.connects = 0

    def connect(self):
        self.connects += 1
        return _FakeConn(self.answers)

    def dispose(self):
        pass


CLEAN_DEMO = {
    "NOT BETWEEN": 0,                 # non-synthetic rows
    "text_original IS NOT NULL": 0,
    "phi_entities": 136,              # notes marked synthetic
    "COUNT(*) FROM clinical_notes": 136,
    "COUNT(*) FROM patients": 36,
}


@pytest.fixture
def demo_env(monkeypatch):
    """A healthy demo deployment: plane=demo, database=lumen_demo, clean corpus."""
    import src.storage as storage
    from src.obs import tracing

    engine = _FakeEngine("postgresql://lumen_demo:x@127.0.0.1:5432/lumen_demo", CLEAN_DEMO)
    monkeypatch.setattr(storage, "engine", engine)
    monkeypatch.setattr(storage, "DATA_PLANE", "demo")
    monkeypatch.setattr(storage, "DEMO_DB_NAME", "lumen_demo")
    monkeypatch.setattr(storage, "RESEARCH_DB_NAME", "lumen")
    monkeypatch.setattr(tracing, "status", lambda: {
        "enabled": True, "provider": "langfuse", "host": "cloud.langfuse.com",
        "state": "pending", "policy": "remote endpoint (demo plane)", "keys_configured": True})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return storage


class _MissingDatabaseEngine:
    """An engine for a database that does not exist. SQLAlchemy builds the engine
    lazily and only raises on connect(), which is exactly where the RunPod
    failure happened, so the fake fails in the same place."""

    def __init__(self, url, **kw):
        self.url = make_url(str(url))

    def connect(self):
        from sqlalchemy.exc import OperationalError

        class _Orig(Exception):
            pgcode = smoke._UNDEFINED_DATABASE
        raise OperationalError("connect", {}, _Orig(f'database "{self.url.database}" does not exist'))

    def dispose(self):
        pass


def _undefined_database(url, **kw):
    return _MissingDatabaseEngine(url, **kw)


# ===========================================================================
# The bug: a demo-only deployment has no research database
# ===========================================================================
def test_safety_passes_when_the_research_database_does_not_exist(demo_env, monkeypatch, capsys):
    """The exact RunPod failure: only lumen_demo exists. This must PASS."""
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)

    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["result"] == "PASS"
    assert out["research"]["state"] == "absent"
    assert out["research"]["result"] == "PASS"
    assert out["demo"]["result"] == "PASS"
    assert out["policy"]["result"] == "PASS"


def test_absent_research_database_is_reported_not_skipped(demo_env, monkeypatch, capsys):
    """Requirement: never silently drop the section."""
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert set(out) >= {"demo", "policy", "research"}
    assert "note" in out["research"] and "demo-only" in out["research"]["note"]
    assert out["research"]["database"] == "lumen"


def test_demo_assertions_run_without_touching_the_research_database(demo_env, monkeypatch, capsys):
    """The demo-plane conclusions must not depend on the research probe at all."""
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert out["demo"]["database"] == "lumen_demo"
    assert out["demo"]["database"] != out["demo"]["research_db_name"]
    assert out["demo"]["non_synthetic_rows"] == 0
    assert out["demo"]["notes_with_text_original"] == 0
    assert out["demo"]["notes_marked_synthetic"] == out["demo"]["notes_total"] == 136
    assert out["demo"]["subjects"] == 36


# ===========================================================================
# The research assertion still bites when a research database is present
# ===========================================================================
def test_clean_research_database_passes(demo_env, monkeypatch, capsys):
    opened = []

    def fake_create_engine(url, **kw):
        opened.append(str(url))
        return _FakeEngine(str(url), {"BETWEEN": 0})

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["research"]["state"] == "checked"
    assert out["research"]["synthetic_rows_in_research_db"] == 0
    assert opened and opened[0].endswith("/lumen")      # probes research, not lumen_demo


def test_synthetic_rows_leaked_into_research_fails(demo_env, monkeypatch, capsys):
    """The guarantee this check exists for must still fail loudly."""
    monkeypatch.setattr("sqlalchemy.create_engine",
                        lambda url, **kw: _FakeEngine(str(url), {"BETWEEN": 7}))
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["result"] == "FAIL"
    assert out["research"]["result"] == "FAIL"
    assert out["research"]["synthetic_rows_in_research_db"] > 0


def test_research_database_present_but_unreadable_fails(demo_env, monkeypatch, capsys):
    """Found a research database and could not prove it clean -> not a PASS."""
    class _Refused:
        def __init__(self, url, **kw): pass
        def connect(self): raise RuntimeError("connection refused")
        def dispose(self): pass
    monkeypatch.setattr("sqlalchemy.create_engine", _Refused)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["research"]["state"] == "unreachable"
    assert out["research"]["result"] == "FAIL"


# ===========================================================================
# Demo-plane assertions must still fail on a genuinely unsafe deployment
# ===========================================================================
@pytest.mark.parametrize("answers, field", [
    ({**CLEAN_DEMO, "NOT BETWEEN": 3}, "non_synthetic_rows"),
    ({**CLEAN_DEMO, "text_original IS NOT NULL": 1}, "notes_with_text_original"),
    ({**CLEAN_DEMO, "phi_entities": 100}, "notes_marked_synthetic"),
])
def test_unsafe_demo_corpus_fails(demo_env, monkeypatch, capsys, answers, field):
    import src.storage as storage
    monkeypatch.setattr(storage, "engine",
                        _FakeEngine("postgresql://lumen_demo:x@127.0.0.1:5432/lumen_demo", answers))
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["demo"]["result"] == "FAIL"


def test_empty_corpus_fails(demo_env, monkeypatch, capsys):
    """0 notes marked synthetic out of 0 notes used to satisfy the equality."""
    import src.storage as storage
    monkeypatch.setattr(storage, "engine", _FakeEngine(
        "postgresql://lumen_demo:x@127.0.0.1:5432/lumen_demo",
        {"NOT BETWEEN": 0, "text_original IS NOT NULL": 0, "phi_entities": 0,
         "COUNT(*) FROM clinical_notes": 0, "COUNT(*) FROM patients": 0}))
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    assert smoke.cmd_safety() == 1


def test_wrong_database_fails(demo_env, monkeypatch, capsys):
    """If the demo plane ever resolved to the research database."""
    import src.storage as storage
    monkeypatch.setattr(storage, "engine",
                        _FakeEngine("postgresql://postgres:x@127.0.0.1:5432/lumen", CLEAN_DEMO))
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["demo"]["result"] == "FAIL"


def test_refused_tracing_policy_fails(demo_env, monkeypatch, capsys):
    """A refused egress policy is a safety failure even with a clean corpus."""
    from src.obs import tracing
    monkeypatch.setattr(tracing, "status", lambda: {
        "enabled": True, "provider": "langfuse", "host": "cloud.langfuse.com",
        "state": "refused", "policy": "remote endpoint refused outside the demo plane"})
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["policy"]["result"] == "FAIL" and out["policy"]["tracing_policy_violation"] is True


@pytest.mark.parametrize("state", ["off", "pending", "unavailable", "misconfigured"])
def test_operational_tracing_states_are_not_safety_failures(demo_env, monkeypatch, capsys, state):
    from src.obs import tracing
    monkeypatch.setattr(tracing, "status", lambda: {"enabled": state != "off", "host": None, "state": state})
    monkeypatch.setattr("sqlalchemy.create_engine", _undefined_database)
    assert smoke.cmd_safety() == 0


# ===========================================================================
# Research URL derivation
# ===========================================================================
def test_unbuildable_research_engine_is_unreachable_not_a_crash(demo_env, monkeypatch, capsys):
    """A malformed research URL must not take the whole safety suite down."""
    def boom(url, **kw):
        raise ValueError("could not parse URL")
    monkeypatch.setattr("sqlalchemy.create_engine", boom)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["research"]["state"] == "unreachable"


def test_missing_table_is_not_mistaken_for_a_missing_database(demo_env, monkeypatch, capsys):
    """'relation "patients" does not exist' means the research database IS there
    but could not be verified — that is unreachable, not vacuously clean."""
    class _NoTables:
        def __init__(self, url, **kw): pass
        def connect(self):
            from sqlalchemy.exc import ProgrammingError

            class _Orig(Exception):
                pgcode = "42P01"
            raise ProgrammingError("SELECT", {}, _Orig('relation "patients" does not exist'))
        def dispose(self): pass
    monkeypatch.setattr("sqlalchemy.create_engine", _NoTables)
    rc = smoke.cmd_safety()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["research"]["state"] == "unreachable"


def test_research_url_defaults_to_the_same_server(demo_env):
    url = smoke._research_url(demo_env)
    assert url.database == "lumen"
    assert url.host == "127.0.0.1" and url.port == 5432


def test_explicit_database_url_wins(demo_env, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@research-host:5555/lumen")
    url = smoke._research_url(demo_env)
    assert url.host == "research-host" and url.port == 5555 and url.database == "lumen"
