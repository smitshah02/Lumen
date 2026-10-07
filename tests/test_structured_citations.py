"""The structured-record citation label "R" (data-foundation plan, E9). "[R1]" is
a citation only when an R1 source is in the evidence; otherwise it is the plain
text it always was, so the control path is unchanged."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.agents import citations

UI = Path(__file__).resolve().parents[1] / "src" / "api" / "ui"
NOTE = {"label": "S1", "text": "note text"}
LAB = {"label": "L1", "text": "lab row"}
RECORD = {"label": "R1", "text": "Coded diagnoses, 2 row(s)"}
ANSWER = "Heart failure was coded [R1]. Creatinine was 1.4 [L1]. She improved [S1]."


def test_r1_resolves_when_an_r1_source_exists():
    report = citations.validate(ANSWER, [NOTE, LAB, RECORD])
    assert [c["valid_labels"] for c in report["claims"]] == [["R1"], ["L1"], ["S1"]]
    assert report["bad_labels"] == [] and report["cite_rate"] == 1.0
    assert report["claims"][0]["source_text"] == RECORD["text"]
    assert citations.strip_bad_labels(ANSWER, [NOTE, LAB, RECORD]) == ANSWER
    assert citations.extract_labels("Coded [R1].", [RECORD]) == ["R1"]


def test_r1_is_plain_text_when_no_r1_source_exists():
    report = citations.validate(ANSWER, [NOTE, LAB])
    first = report["claims"][0]
    assert first["labels"] == [] and first["uncited"] is True and first["bad_labels"] == []   # not a citation at all
    assert report["bad_labels"] == []                                    # and not a "hallucinated label" either
    assert citations.strip_bad_labels(ANSWER, [NOTE, LAB]) == ANSWER      # the text is left exactly as written
    assert citations.extract_labels("Coded [R1].") == [] and citations.extract_labels("Coded [R1].", [NOTE]) == []
    assert citations.CITE_RE.findall(ANSWER) == ["L1", "S1"]              # the shared pattern never learned R


def test_only_the_exact_structured_label_resolves():
    report = citations.validate("One [R1]. Two [R2]. Twelve [R12].", [RECORD])
    assert [c["labels"] for c in report["claims"]] == [["R1"], [], []]    # R2 and R12 are not sources: plain text
    assert citations.cite_re([RECORD, {"label": "R12", "text": "x"}]).findall("[R1] [R12] [R2]") == ["R1", "R12"]


@pytest.mark.parametrize("evidence", [[NOTE, LAB], [NOTE, LAB, RECORD]])
def test_existing_labels_behave_the_same_with_or_without_a_structured_source(evidence):
    answer = "Guideline says so [G1]. Trial data [P1]. Three admissions [A1]. Creatinine 1.4 [L1] [S1]. Made up [S9]."
    extra = [{"label": "G1", "text": "g"}, {"label": "P1", "text": "p"}, {"label": "A1", "text": "a"}]
    report = citations.validate(answer, evidence + extra)
    assert [c["labels"] for c in report["claims"]] == [["G1"], ["P1"], ["A1"], ["L1", "S1"], ["S9"]]
    assert report["bad_labels"] == ["S9"] and report["claims"][4]["bad_labels"] == ["S9"]
    assert citations.strip_bad_labels(answer, evidence + extra).endswith("Made up .")
    assert citations.CITE_RE.pattern == r"\[([SLGPA]\d+)\]"               # the pattern verify.py and the evals import
    assert citations.cite_re([NOTE, LAB]) is citations.CITE_RE            # no structured source: the very same object


def test_orphan_citation_repair_is_unchanged():
    assert citations.normalize_orphan_citations("She is on furosemide.\n[S5]") == "She is on furosemide [S5]."
    assert citations.normalize_orphan_citations("Coded.\n[R1]") == "Coded.\n[R1]"   # R was never part of this repair


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_renders_r1_as_a_citation_only_when_the_response_has_that_source():
    script = f"""
      const R = require({json.dumps(str(UI / "review.js"))});
      const labels = (t) => R.citationRuns(t).filter((p) => p.labels).map((p) => p.labels);
      const none = {{ r: labels("Coded [R1]. Note [S1]."), old: labels("A [S1] [S2]. B [L1], C [G1] [P1], D [A1].") }};
      R.structuredLabels = ["S1", "R1"];
      const some = {{ r: labels("Coded [R1] [S1]. Other [R2]."), old: labels("A [S1] [S2]. B [L1], C [G1] [P1], D [A1].") }};
      R.structuredLabels = [];
      console.log(JSON.stringify({{ none, some, after: labels("Coded [R1].") }}));"""
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out["none"]["r"] == [["S1"]]                                   # no R source: "[R1]" stays text
    assert out["some"]["r"] == [["R1", "S1"]]                             # R1 source present; "[R2]" still text
    assert out["none"]["old"] == out["some"]["old"] == [["S1", "S2"], ["L1"], ["G1", "P1"], ["A1"]]
    assert out["after"] == []
    app = (UI / "app.js").read_text()
    assert "LumenReview.structuredLabels = ((msg && msg.resp && msg.resp.sources) || []).map((s) => s.label);" in app
    assert "const CITE_RE = /\\[([SLGPA]\\d+)\\]/g;" in app
