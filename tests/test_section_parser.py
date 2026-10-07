"""Note-type-aware section parser (data-foundation plan, E11). The notes here are
invented; the check against the five real baseline notes reads the git-ignored
benchmark and the research database, and skips when either is missing."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # also runs as a script, for the baseline check

from src.retrieval.section_labels import STANDARD_HEADERS, Section, parse_sections  # noqa: E402

BASELINE = ROOT / "reports" / "data_foundation" / "before_benchmark.json"
SKIP = 77

DISCHARGE = """Name:  ___                 Unit No:   ___
Admission Date:  ___              Discharge Date:   ___
Service: MEDICINE

Allergies:
Penicillins

Chief Complaint:
Chest pain

Major Surgical or Invasive Procedure:
Cardiac catheterization

History of Present Illness:
Appears comfortable on arrival. Planned for catheterization.
Appetite was poor for two days.

Pertinent Results:
___ 07:10AM BLOOD WBC-5.4 Hgb-8.1*
___ 07:10AM BLOOD Glucose-153* Creat-2.3*

Brief Hospital Course:
Admitted with chest pain.
Appointment with cardiology was arranged before discharge.
Plan: continue aspirin.
Medications were reviewed at length with the patient.

Medications on Admission:
1. Aspirin 81 mg PO DAILY

Discharge Medications:
1. Aspirin 81 mg PO DAILY
2. Atorvastatin 80 mg PO QPM

Discharge Disposition:
Home

Discharge Diagnosis:
Unstable angina

Discharge Condition:
Stable

Discharge Instructions:
Take your medications as prescribed.

Followup Instructions:
___
"""


def _names(sections):
    return [s.name for s in sections]


def _check_offsets(text, sections):
    assert [s.order for s in sections] == list(range(len(sections)))
    for s in sections:
        assert text[s.start:s.end] == s.text                       # verbatim, newlines and all
    for a, b in zip(sections, sections[1:]):
        assert a.end == b.start                                    # contiguous, never overlapping
    assert sections[-1].end == len(text)


def test_discharge_summary_is_cut_into_its_template_sections_in_order():
    sections = parse_sections(DISCHARGE, "discharge")
    assert _names(sections) == [
        "Header", "Allergies", "Chief Complaint", "Major Surgical or Invasive Procedure",
        "History of Present Illness", "Pertinent Results", "Brief Hospital Course", "Medications on Admission",
        "Discharge Medications", "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition",
        "Discharge Instructions", "Followup Instructions"]
    _check_offsets(DISCHARGE, sections)
    assert sections[0].start == 0 and "".join(s.text for s in sections) == DISCHARGE
    by = {s.name: s for s in sections}
    assert by["Pertinent Results"].text.startswith("Pertinent Results:\n___ 07:10AM")
    assert "Creat-2.3*\n" in by["Pertinent Results"].text          # table lines keep their line breaks
    assert by["Discharge Medications"].text.count("\n1. Aspirin") == 1
    assert "2. Atorvastatin 80 mg PO QPM" in by["Discharge Medications"].text


def test_ordinary_lines_do_not_open_sections():
    course = {s.name: s for s in parse_sections(DISCHARGE, "discharge")}["Brief Hospital Course"].text
    # "Appointment ...", "Plan: ..." and "Medications were ..." all stay inside the hospital course
    for line in ("Appointment with cardiology", "Plan: continue aspirin.", "Medications were reviewed"):
        assert line in course
    hpi = {s.name: s for s in parse_sections(DISCHARGE, "discharge")}["History of Present Illness"].text
    for word in ("Appears", "Planned", "Appetite"):
        assert word in hpi
    # a header needs its colon, and must start the line
    text = "Allergies\nnone known\nSee Discharge Medications: below\n"
    assert _names(parse_sections(text, "discharge")) == ["Header"]


def test_a_repeated_header_is_a_second_section_not_a_merge():
    text = "Physical Exam:\nADMISSION: alert\n\nPertinent Results:\nWBC-5.4\n\nPhysical Exam:\nDISCHARGE: alert\n"
    sections = parse_sections(text, "discharge")
    assert _names(sections) == ["Physical Exam", "Pertinent Results", "Physical Exam"]
    assert "ADMISSION" in sections[0].text and "DISCHARGE" in sections[2].text and "DISCHARGE" not in sections[0].text
    _check_offsets(text, sections)


def test_parser_uses_the_one_canonical_header_list():
    for _, label in STANDARD_HEADERS:                               # every label is a valid spelling of its own header
        assert _names(parse_sections(f"{label}:\nsome text\n", "discharge")) == [label], label


RADIOLOGY = """EXAMINATION:  CHEST (PA AND LAT)

INDICATION:  ___ year old woman with cough.

TECHNIQUE:  Chest PA and lateral

COMPARISON:  ___

FINDINGS:
Lungs are clear. Appearance of the heart is normal.
Planned follow-up is not needed.

IMPRESSION:
No acute cardiopulmonary process.
"""


def test_radiology_report_is_cut_on_its_six_template_headers():
    sections = parse_sections(RADIOLOGY, "radiology")
    assert _names(sections) == ["Examination", "Indication", "Technique", "Comparison", "Findings", "Impression"]
    _check_offsets(RADIOLOGY, sections)
    assert "Planned follow-up" in sections[4].text and sections[5].text.startswith("IMPRESSION:\nNo acute")


def test_radiology_without_a_template_header_is_one_report_section():
    text = "CHEST RADIOGRAPH\n\nHISTORY: cough.\n\nNo focal consolidation. Impression unchanged from prior.\n"
    (only,) = parse_sections(text, "radiology")
    assert (only.name, only.order, only.start, only.end, only.text) == ("Report", 0, 0, len(text), text)


def test_radiology_title_before_the_first_header_is_kept_and_discharge_headers_are_ignored():
    text = "PORTABLE CHEST\n\nFINDINGS:  Clear.\n\nAllergies:\nnot a radiology header\n\nIMPRESSION:  Normal.\n"
    sections = parse_sections(text, "radiology")
    assert _names(sections) == ["Header", "Findings", "Impression"]
    assert "Allergies:" in sections[1].text
    _check_offsets(text, sections)


def test_empty_and_whitespace_notes_have_no_sections():
    assert parse_sections("", "discharge") == [] and parse_sections("  \n\n", "radiology") == []
    assert parse_sections(None, "discharge") == []
    assert isinstance(parse_sections("free text only", "discharge")[0], Section)


def baseline_check() -> int:
    """Each ground-truth sentence of the 15 baseline questions must fall in its true section."""
    from sqlalchemy import text as sql
    from src.storage import check_connection, engine
    if not check_connection():
        return SKIP
    wrong = 0
    items = json.loads(BASELINE.read_text())["items"]
    with engine.connect() as c:
        for item in items:
            note = c.execute(sql("SELECT text_original FROM clinical_notes WHERE note_id = :n"),
                             {"n": item["note_id"]}).scalar()
            # the sentence can recur elsewhere in the note (a drug is also on the admission
            # list); the benchmark's claim is that it occurs exactly once in its own section
            sections = parse_sections(note, "discharge")
            homes = [next(s.name for s in sections if s.start <= m.start() < s.end)
                     for m in re.finditer(item["provenance"]["regex"], note)]
            if homes.count(item["source_section"]) != item["provenance"]["matches_in_section"]:
                wrong += 1
                print(f"{item['qid']}: expected once in {item['source_section']}, found in {homes}")
    print(f"baseline sections: {len(items) - wrong}/{len(items)} correct")
    return 1 if wrong else 0


def test_baseline_ground_truth_lands_in_its_true_section():
    import pytest
    if not BASELINE.exists():
        pytest.skip("baseline artifact not present (git-ignored)")
    env = {**os.environ, "LUMEN_DATA_PLANE": "research", "LUMEN_TRACING": "0"}
    run = subprocess.run([sys.executable, __file__], env=env, cwd=ROOT, capture_output=True, text=True)
    if run.returncode == SKIP:
        pytest.skip("research database not reachable")
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(baseline_check())
