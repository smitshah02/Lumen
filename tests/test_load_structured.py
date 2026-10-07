"""Structured-data loader (data-foundation plan, E8). No database: the cursor is a
recorder and the source files are small CSVs written by the test."""
import csv
import gzip
import re
from pathlib import Path

import pytest

from src.storage import load_structured as ls
from src.storage.schema import LAB_FULL_INDEX_SQL, MIGRATIONS, SCHEMA_VERSION, STRUCTURED_SQL

SOURCE = Path(ls.__file__).read_text(encoding="utf-8")
CONTROL_TABLES = ("patients", "admissions", "diagnoses_icd", "labevents", "d_labitems", "prescriptions",
                  "procedures_icd", "clinical_notes", "note_chunks")


def _lab(i, subject_id, hadm_id=""):
    return {"labevent_id": str(i), "subject_id": str(subject_id), "hadm_id": hadm_id, "specimen_id": "77",
            "itemid": "50912", "order_provider_id": "", "charttime": "2180-05-06 22:25:00", "storetime": "",
            "value": "1.1", "valuenum": "1.1", "valueuom": "mg/dL", "ref_range_lower": "0.4",
            "ref_range_upper": "1.1", "flag": "", "priority": "ROUTINE", "comments": ""}


def test_lab_rows_have_no_per_patient_cap_and_keep_only_the_cohort():
    rows = [_lab(i, 1) for i in range(500)] + [_lab(900, 2), _lab(901, 3)]
    out = list(ls.lab_rows(rows, {1, 3}))
    assert len(out) == 501 and {r[1] for r in out} == {"1", "3"}          # all 500 of patient 1, none of patient 2
    assert out[0] == ["0", "1", "", "77", "50912", "", "2180-05-06 22:25:00", "", "1.1", "1.1", "mg/dL",
                      "0.4", "1.1", "", "ROUTINE", ""]                   # column order of the table; empty stays empty
    assert list(ls.lab_rows([_lab(5, 1, hadm_id="22595853.0")], {1}))[0][2] == "22595853"


def test_icd_rows_keep_code_version_and_title():
    assert list(ls.icd_rows([{"icd_code": "4280 ", "icd_version": "9", "long_title": "Heart failure, unspecified"}])) \
        == [["4280", "9", "Heart failure, unspecified"]]


class _Cursor:
    def __init__(self, fail_on=None):
        self.statements, self.copied, self.fail_on = [], {}, fail_on

    def execute(self, sql):
        self.statements.append(sql)

    def copy_expert(self, sql, buffer):
        table = sql.split()[1]
        if table == self.fail_on:
            raise RuntimeError("copy failed")
        self.statements.append(sql)
        self.copied[table] = self.copied.get(table, 0) + sum(1 for _ in csv.reader(buffer))


@pytest.fixture
def hosp(tmp_path):
    def write(name, header, rows, gz=True):
        path = tmp_path / (f"{name}.csv.gz" if gz else f"{name}.csv")
        with (gzip.open(path, "wt", newline="") if gz else open(path, "w", newline="")) as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    write("d_icd_diagnoses", ls.ICD_COLUMNS, [["4280", "9", "Heart failure"], ["I509", "10", "Heart failure, unspecified"]])
    write("d_icd_procedures", ls.ICD_COLUMNS, [["3722", "9", "Left heart cardiac catheterization"]], gz=False)
    write("labevents", ls.LAB_COLUMNS, [list(_lab(i, 1 if i < 450 else 2).values()) for i in range(460)])
    return tmp_path


def test_load_replaces_only_its_own_tables_and_loads_every_cohort_row(hosp, monkeypatch):
    monkeypatch.setattr(ls, "COPY_BATCH", 100)                            # several batches
    cur = _Cursor()
    counts = ls.load_tables(cur, hosp, {1})
    assert counts == {"d_icd_diagnoses": 2, "d_icd_procedures": 1, "labevents_full": 450} == cur.copied
    assert cur.statements[0] == "TRUNCATE labevents_full, d_icd_diagnoses, d_icd_procedures"
    assert LAB_FULL_INDEX_SQL in cur.statements
    assert cur.statements.index(LAB_FULL_INDEX_SQL) > max(i for i, s in enumerate(cur.statements) if s.startswith("COPY"))
    for statement in cur.statements:                                     # nothing names an ingested table
        assert not re.search(r"\b(" + "|".join(CONTROL_TABLES) + r")\b", statement), statement


def test_a_failed_load_rolls_back_and_is_recorded_as_failed(hosp, monkeypatch):
    events = []

    class _Raw:
        def cursor(self): return _Cursor(fail_on="labevents_full")
        def commit(self): events.append("commit")
        def rollback(self): events.append("rollback")
        def close(self): events.append("close")

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "SELECT subject_id FROM patients" in sql:
                return [(1,)]
            if "structured_load_runs" in sql and params:
                events.append(params.get("status", "running"))

    class _Engine:
        def begin(self): return _Conn()
        def connect(self): return _Conn()
        def raw_connection(self): return _Raw()

    monkeypatch.setattr(ls, "DATA_PLANE", "research")
    with pytest.raises(RuntimeError, match="copy failed"):
        ls.run_load(_Engine(), hosp)
    assert events == ["running", "rollback", "failed", "close"]          # never committed, never marked completed


def test_loader_refuses_the_demo_plane_and_a_missing_source_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "DATA_PLANE", "demo")
    with pytest.raises(RuntimeError, match="research plane only"):
        ls.run_load(object(), tmp_path)
    monkeypatch.setattr(ls, "DATA_PLANE", "research")
    with pytest.raises(FileNotFoundError):
        ls.run_load(object(), tmp_path)                                  # object() has no engine methods: nothing was touched


def test_loader_is_separate_from_the_ingestion_pipeline():
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", SOURCE, re.M)
    assert not any(m.startswith("src.storage.ingest") for m in imports)
    code = SOURCE.split('"""', 2)[2]                                      # everything after the module docstring
    assert "run_ingestion" not in code and "CASCADE" not in code and "DELETE FROM" not in code
    assert code.count("TRUNCATE") == 1 and "TRUNCATE {', '.join(OWN_TABLES)}" in code
    assert ls.OWN_TABLES == ("labevents_full", "d_icd_diagnoses", "d_icd_procedures")


def test_structured_tables_are_additive_and_outside_the_versioned_schema():
    created = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", STRUCTURED_SQL)
    assert created == ["labevents_full", "d_icd_diagnoses", "d_icd_procedures", "structured_load_runs"]
    assert "DROP" not in STRUCTURED_SQL and "ALTER" not in STRUCTURED_SQL and "REFERENCES" not in STRUCTURED_SQL
    assert LAB_FULL_INDEX_SQL.endswith("ON labevents_full (subject_id, itemid, charttime)")
    assert "status IN ('running', 'completed', 'failed')" in STRUCTURED_SQL
    assert SCHEMA_VERSION == 5 and not any("labevents_full" in s for m in MIGRATIONS.values() for s in m)
