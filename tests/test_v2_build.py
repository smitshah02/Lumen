"""v2 note-index build (data-foundation plan, E13). No database, no model: a small
fake stands in for Postgres and the embedder, and the tokenizer is a word counter."""
import csv
import gzip
import json
import re
from contextlib import contextmanager
from datetime import datetime

import pytest

from src.retrieval import index_notes as ix
from src.retrieval.chunker import V2_CHUNK_CONFIG, V2_CHUNKER_VERSION
from src.retrieval.index_provenance import configuration_hash
from src.retrieval.section_labels import PARSER_VERSION
from src.storage.schema import MIGRATIONS, NOTE_CHUNKS_V2_SQL, SCHEMA_VERSION


class _Words:
    def __call__(self, text, **_):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]}


DISCHARGE = ("Name: ___ Unit No: ___\n\nAllergies:\nPenicillins\n\nBrief Hospital Course:\n"
             + "She was diuresed with good effect and improved steadily. " * 10 + "\n\nDischarge Medications:\n"
             + "".join(f"{i}. Drug{i} {i} mg PO DAILY with food and plenty of water every morning\n" for i in range(1, 12)))
RADIOLOGY = "FINDINGS:\n" + "The lungs are clear without focal consolidation. " * 8 + "\n\nIMPRESSION:\nNo acute process.\n"
NOTES = [
    {"note_id": 12, "subject_id": 7, "hadm_id": 500, "note_type": "radiology", "charttime": datetime(2180, 5, 6, 20), "text": RADIOLOGY},
    {"note_id": 11, "subject_id": 7, "hadm_id": 500, "note_type": "discharge", "charttime": datetime(2180, 5, 7), "text": DISCHARGE},
]
IDS = {11: "7-DS-21", 12: "7-RR-4"}


def _key(row):
    return row["build_id"], row["mimic_note_id"], row["section_ord"], row["chunk_ord"]


def test_rows_are_deterministic_keyed_and_exact_slices_of_the_note():
    rows = ix.v2_rows(NOTES, IDS, _Words(), V2_CHUNK_CONFIG, "build-1")
    assert rows == ix.v2_rows(list(reversed(NOTES)), IDS, _Words(), V2_CHUNK_CONFIG, "build-1")   # input order does not matter
    assert len({_key(r) for r in rows}) == len(rows) and [r["mimic_note_id"] for r in rows] == sorted(r["mimic_note_id"] for r in rows)
    text = {11: DISCHARGE, 12: RADIOLOGY}
    for r in rows:
        assert text[r["note_id"]][r["char_start"]:r["char_end"]] == r["chunk_text"]
        assert r["embedding"] is None and r["build_id"] == "build-1" and r["subject_id"] == 7 and r["hadm_id"] == 500
        assert (r["note_type"], r["charttime"]) == (("discharge", datetime(2180, 5, 7)) if r["note_id"] == 11
                                                    else ("radiology", datetime(2180, 5, 6, 20)))
    names = [(r["section_name"], r["embed"]) for r in rows if r["note_id"] == 11]
    assert names == [("Header", False), ("Allergies", False), ("Brief Hospital Course", True), ("Discharge Medications", True)]
    assert [(r["section_name"], r["section_ord"], r["chunk_ord"]) for r in rows if r["note_id"] == 12] == [("Report", 0, 0)]
    # a second build of the same source and configuration: identical keys (but for the build id) and text
    again = ix.v2_rows(NOTES, IDS, _Words(), V2_CHUNK_CONFIG, "build-2")
    assert [(_key(r)[1:], r["chunk_text"], r["char_start"], r["char_end"], r["token_count"], r["embed"]) for r in again] \
        == [(_key(r)[1:], r["chunk_text"], r["char_start"], r["char_end"], r["token_count"], r["embed"]) for r in rows]


def test_threshold_changes_embed_flags_only():
    base = ix.v2_rows(NOTES, IDS, _Words(), V2_CHUNK_CONFIG, "b")
    zero = ix.v2_rows(NOTES, IDS, _Words(), {**V2_CHUNK_CONFIG, "min_embed_tokens": 0}, "b")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "embed"} for r in rows]
    assert strip(zero) == strip(base) and [r["section_name"] for r in zero if not r["embed"]] == ["Header"]


def test_configuration_records_everything_that_shapes_a_build():
    cfg = ix.v2_configuration([9, 7], V2_CHUNK_CONFIG)
    assert cfg["provenance"] == "v2" and cfg["table"] == "note_chunks_v2"
    assert cfg["parser"] == {"version": PARSER_VERSION}
    assert cfg["chunker"] == {"version": V2_CHUNKER_VERSION, "target_tokens": 480, "max_tokens": 512, "min_embed_tokens": 40,
                              "embedded_text": "chunk_text, verbatim"}
    assert set(cfg["tokenizer"]) == {"repo", "revision"} and cfg["tokenizer"]["revision"] == cfg["embedding"]["revision"]
    assert cfg["scope"] == {"subject_ids": [7, 9]}
    json.dumps(cfg)                                                       # storable as it is
    other_patients = ix.v2_configuration([1], V2_CHUNK_CONFIG)
    other_threshold = ix.v2_configuration([9, 7], {**V2_CHUNK_CONFIG, "min_embed_tokens": 0})
    h = lambda c: configuration_hash({k: v for k, v in c.items() if k != "scope"})
    assert h(cfg) == h(other_patients) and h(cfg) != h(other_threshold)   # the hash is the recipe, not the patients


# --- recovering MIMIC's own note ids ----------------------------------------------------------------
@pytest.fixture
def note_dir(tmp_path):
    header = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]
    with gzip.open(tmp_path / "discharge.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(["7-DS-21", 7, 500, "DS", 21, "2180-05-07 00:00:00", "", DISCHARGE])
        w.writerow(["8-DS-3", 8, 900, "DS", 3, "2180-05-07 00:00:00", "", DISCHARGE])            # another patient, same text
    with open(tmp_path / "radiology.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(["7-RR-4", 7, 500, "RR", 4, "2180-05-06 20:00:00", "", RADIOLOGY])
        w.writerow(["7-RR-5", 7, 500, "RR", 5, "2180-05-06 20:00:00", "", RADIOLOGY])            # an identical duplicate report
        w.writerow(["7-RR-9", 7, 500, "RR", 9, "2180-05-06 21:00:00", "", RADIOLOGY + "addendum"])
    return tmp_path


def test_mimic_ids_are_recovered_by_patient_time_and_exact_text(note_dir):
    twin = {**NOTES[0], "note_id": 13}
    assert ix.recover_mimic_note_ids(NOTES, note_dir) == {11: "7-DS-21", 12: "7-RR-4"}
    assert ix.recover_mimic_note_ids(NOTES + [twin], note_dir) == {11: "7-DS-21", 12: "7-RR-4", 13: "7-RR-5"}   # duplicates pair off in order


def test_a_note_with_no_source_row_fails_the_build_instead_of_getting_a_made_up_id(note_dir):
    edited = {**NOTES[1], "note_id": 14, "text": DISCHARGE + " edited"}
    with pytest.raises(RuntimeError, match="no matching MIMIC source row"):
        ix.recover_mimic_note_ids([edited], note_dir)
    with pytest.raises(RuntimeError, match="no matching MIMIC source row"):
        ix.recover_mimic_note_ids(NOTES + [{**NOTES[0], "note_id": 13}, {**NOTES[0], "note_id": 15}], note_dir)   # three notes, two source rows


# --- storage definition -------------------------------------------------------------------------------
def test_v2_table_is_additive_keyed_by_build_and_has_no_hnsw():
    statements = [s.strip() for s in NOTE_CHUNKS_V2_SQL.split(";")]
    assert len(statements) == 6 and all(s.startswith((
        "CREATE TABLE IF NOT EXISTS note_chunks_v2", "CREATE INDEX IF NOT EXISTS idx_chunks_v2_",
        "ALTER TABLE note_chunks_v2 ADD COLUMN IF NOT EXISTS chunk_id", "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_v2_chunk_id"))
        for s in statements)
    table = statements[0]
    for column in ("build_id", "mimic_note_id", "section_ord", "chunk_ord", "subject_id", "hadm_id", "note_type", "charttime",
                   "section_name", "char_start", "char_end", "chunk_text", "token_count", "embed", "embedding", "text_search"):
        assert re.search(rf"\n\s+{column}\s", table), column
    assert "PRIMARY KEY (build_id, mimic_note_id, section_ord, chunk_ord)" in table
    assert "GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED" in table
    assert "CHECK (embed = (embedding IS NOT NULL))" in table
    assert "hnsw" not in NOTE_CHUNKS_V2_SQL.lower() and "REFERENCES" not in NOTE_CHUNKS_V2_SQL
    assert not re.search(r"\bnote_chunks\b(?!_v2)", NOTE_CHUNKS_V2_SQL)                      # never names the control table
    assert SCHEMA_VERSION == 5 and not any("note_chunks_v2" in s for m in MIGRATIONS.values() for s in m)


# --- build lifecycle -----------------------------------------------------------------------------------
class _Db:
    def __init__(self, drop_rows=0):
        self.rows, self.run, self.events, self.drop_rows = [], {}, [], drop_rows

    @contextmanager
    def begin(self):
        yield self

    connect = begin

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM clinical_notes" in sql:
            return type("R", (), {"mappings": lambda s: NOTES})()
        if "INSERT INTO note_index_runs" in sql:
            self.run = {"status": "running", **params}
            self.events.append("running")
        elif "UPDATE note_index_runs" in sql:
            self.run.update(params)
            self.events.append(params["status"])
        elif "INSERT INTO note_chunks_v2" in sql:
            assert self.run["status"] == "running"                       # rows are only ever written by a running build
            self.rows += params
        elif "FROM note_chunks_v2" in sql:
            mine = [r for r in self.rows if r["build_id"] == params["b"]]
            stored = len(mine) - self.drop_rows
            return type("R", (), {"first": lambda s: (stored, sum(1 for r in mine if r["embedding"] is not None))})()
        elif "note_chunks" in sql and "note_chunks_v2" not in sql:
            raise AssertionError("the v2 build must not touch note_chunks")


class _Embedder:
    def __init__(self, fail=False, short=False):
        self.fail, self.short, self.texts = fail, short, []

    def embed_documents(self, texts, show_progress=True):
        if self.fail:
            raise RuntimeError("accelerator lost")
        self.texts += texts
        return [[0.5, 0.25]] * (len(texts) - (1 if self.short else 0))


def _build(db, embedder):
    return ix.run_v2_build([7], db=db, tokenizer=_Words(), embedder=embedder, mimic_ids=IDS)


def test_a_build_is_completed_only_after_every_chunk_and_embedding_is_stored():
    db, embedder = _Db(), _Embedder()
    stats = _build(db, embedder)
    assert db.events == ["running", "completed"] and db.run["status"] == "completed"
    assert stats["build_id"] == db.run["run_id"] and len(stats["build_id"]) == 36          # what LUMEN_CHUNK_BUILD takes
    assert (stats["patients"], stats["notes"], stats["chunks"]) == (1, 2, len(db.rows)) and stats["chunks"] == 5
    assert stats["embedded"] == 3 and stats["lexical_only"] == 2 and stats["embedded"] + stats["lexical_only"] == stats["chunks"]
    assert stats["max_embedded_tokens"] <= 480
    assert all((r["embedding"] is not None) == r["embed"] for r in db.rows) and all(r["build_id"] == stats["build_id"] for r in db.rows)
    assert embedder.texts == [r["chunk_text"] for r in db.rows if r["embed"]]              # the exact chunk text is what is embedded
    recorded = json.loads(db.run["configuration"])
    assert recorded == ix.v2_configuration([7], V2_CHUNK_CONFIG) and db.run["expected"] == 2
    assert (db.run["notes"], db.run["chunks"]) == (2, 5) and len(db.run["config_hash"]) == 64


@pytest.mark.parametrize("db,embedder,error", [
    (_Db(), _Embedder(fail=True), "accelerator lost"),
    (_Db(), _Embedder(short=True), "embedder returned"),
    (_Db(drop_rows=1), _Embedder(), "stored 4 chunks"),                  # what is in the table is not what was built
])
def test_a_build_that_does_not_finish_is_failed_and_never_completed(db, embedder, error):
    with pytest.raises(RuntimeError, match=error):
        _build(db, embedder)
    assert db.events == ["running", "failed"] and db.run["status"] == "failed" and db.run["error"] == "RuntimeError"


def test_two_builds_coexist_and_a_failed_one_leaves_the_first_alone():
    db = _Db()
    first = _build(db, _Embedder())
    kept = [dict(r) for r in db.rows]
    with pytest.raises(RuntimeError):
        ix.run_v2_build([7], db=db, tokenizer=_Words(), embedder=_Embedder(fail=True), mimic_ids=IDS)
    assert [r for r in db.rows if r["build_id"] == first["build_id"]] == kept               # untouched
    second = _build(db, _Embedder())
    assert second["build_id"] != first["build_id"]
    a = [(_key(r)[1:], r["chunk_text"]) for r in db.rows if r["build_id"] == first["build_id"]]
    b = [(_key(r)[1:], r["chunk_text"]) for r in db.rows if r["build_id"] == second["build_id"]]
    assert a == b and len(a) == 5                                         # same source, same recipe: same keys and text


def test_control_indexer_and_its_configuration_are_untouched():
    from src.retrieval.index_provenance import CHUNKER_CONFIG, CHUNKER_VERSION, RETRIEVER_CHUNK_TABLES
    assert (CHUNKER_VERSION, CHUNKER_CONFIG) == ("clinical-note-chunker-v2", {"max_tokens": 384, "overlap_tokens": 64, "min_chunk_tokens": 50})
    assert RETRIEVER_CHUNK_TABLES[0] == "note_chunks"                     # the control table is still the default one


def test_patient_id_lists_hold_ids_only_and_a_build_is_always_new(tmp_path):
    import inspect
    from src.retrieval.index_notes import read_subject_ids, run_v2_build
    ids = tmp_path / "ids.txt"
    ids.write_text("# held-out patients\n10000980\n\n10000032  # comment\n10000980\n")
    assert read_subject_ids("", str(ids)) == [10000032, 10000980]
    assert read_subject_ids("7, 5", str(ids)) == [5, 7, 10000032, 10000980]
    ids.write_text("10000032\nWhat was the discharge diagnosis?\n")
    with pytest.raises(ValueError) as err:
        read_subject_ids("", str(ids))
    assert "discharge" not in str(err.value)                    # the offending line is never echoed
    assert not {"build_id", "run_id", "append"} & set(inspect.signature(run_v2_build).parameters)   # no way to write into an existing build
