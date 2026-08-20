"""
De-Identification Pipeline Test Runner
========================================
Runs the pipeline against sample clinical notes and evaluates performance.

Usage:
    cd Lumen
    source .venv/bin/activate
    python -m src.deid.test_pipeline

Outputs:
    - Redacted text for each sample note
    - Per-note entity detection summary
    - Recall measurement against expected PHI
    - Overall pipeline statistics
"""

from __future__ import annotations

import sys
import time
import logging
from collections import defaultdict

import os

from src.deid.pipeline import DeidentificationPipeline, DeidResult
from src.deid.sample_notes import SAMPLE_NOTES
from src.deid.mimic_adapter import MIMICNoteLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def measure_recall(result: DeidResult, expected_phi: list[dict]) -> dict:
    """
    Measure how many expected PHI entities were actually detected.

    Uses substring matching: if the redacted text no longer contains the
    expected PHI value, we count it as successfully caught.
    """
    caught = []
    missed = []

    for phi in expected_phi:
        # If the original value is no longer present in the redacted text,
        # the pipeline successfully caught it
        if phi["value"] not in result.text:
            caught.append(phi)
        else:
            missed.append(phi)

    total = len(expected_phi)
    recall = len(caught) / total if total > 0 else 0.0

    return {
        "total_expected": total,
        "caught": len(caught),
        "missed": len(missed),
        "recall": recall,
        "missed_entities": missed,
    }


def print_divider(char: str = "=", width: int = 80):
    print(char * width)


def run_tests():
    print_divider()
    print("  LUMEN DE-IDENTIFICATION PIPELINE — TEST RUN")
    print_divider()
    print()

    # Initialize pipeline
    print("Initializing pipeline (loading spaCy model + Presidio)...")
    start = time.time()
    pipeline = DeidentificationPipeline(score_threshold=0.35)
    init_time = time.time() - start
    print(f"Pipeline ready in {init_time:.1f}s\n")

    # Aggregate stats
    total_expected = 0
    total_caught = 0
    total_missed = 0
    all_missed: list[dict] = []
    entity_type_counts: dict[str, int] = defaultdict(int)
    note_times: list[float] = []

    for sample in SAMPLE_NOTES:
        note_id = sample["id"]
        note_type = sample["note_type"]
        text = sample["text"]
        expected = sample["expected_phi"]

        print_divider("-")
        print(f"NOTE: {note_id} ({note_type})")
        print(f"Length: {len(text)} chars | Expected PHI: {len(expected)} entities")
        print_divider("-")

        # Run de-identification
        t0 = time.time()
        result = pipeline.deidentify(text)
        elapsed = time.time() - t0
        note_times.append(elapsed)

        # Measure recall
        recall_info = measure_recall(result, expected)
        total_expected += recall_info["total_expected"]
        total_caught += recall_info["caught"]
        total_missed += recall_info["missed"]

        for m in recall_info["missed_entities"]:
            all_missed.append({"note_id": note_id, **m})

        # Count entity types detected
        for ent in result.entities:
            entity_type_counts[ent["type"]] += 1

        # Print redacted text (first 600 chars)
        print("\n📝 REDACTED TEXT (preview):")
        preview = result.text[:600]
        if len(result.text) > 600:
            preview += "\n  [...truncated...]"
        print(preview)

        # Print detected entities
        print(f"\n🔍 DETECTED ENTITIES ({len(result.entities)}):")
        for ent in result.entities:
            print(
                f"  [{ent['type']:<26}] "
                f"score={ent['score']:.2f}  "
                f'"{ent["original_value"]}"'
            )

        # Print recall for this note
        emoji = "✅" if recall_info["recall"] == 1.0 else "⚠️"
        print(
            f"\n{emoji} RECALL: {recall_info['caught']}/{recall_info['total_expected']} "
            f"({recall_info['recall']:.0%})"
        )
        if recall_info["missed_entities"]:
            print("  MISSED:")
            for m in recall_info["missed_entities"]:
                print(f"    - [{m['type']}] \"{m['value']}\"")

        print(f"  ⏱️  Processing time: {elapsed:.3f}s")
        print()

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    print_divider("=")
    print("  OVERALL RESULTS")
    print_divider("=")

    overall_recall = total_caught / total_expected if total_expected > 0 else 0.0
    print(f"\n  Notes tested:       {len(SAMPLE_NOTES)}")
    print(f"  Total expected PHI: {total_expected}")
    print(f"  Total caught:       {total_caught}")
    print(f"  Total missed:       {total_missed}")
    print(f"  Overall recall:     {overall_recall:.1%}")
    print(f"  Avg time/note:      {sum(note_times)/len(note_times):.3f}s")
    print(f"  Total time:         {sum(note_times):.3f}s")

    print("\n  Entity types detected across all notes:")
    for etype, count in sorted(entity_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:<28} {count}")

    if all_missed:
        print(f"\n  ⚠️  ALL MISSED ENTITIES ({len(all_missed)}):")
        for m in all_missed:
            print(f"    [{m['note_id']}] [{m['type']}] \"{m['value']}\"")
        print(
            "\n  💡 TIP: Missed entities often need threshold tuning or a custom"
            "\n  recognizer. Check if lowering score_threshold helps, or add a"
            "\n  PatternRecognizer for the specific format."
        )
    else:
        print("\n  🎉 PERFECT RECALL — all expected PHI entities were caught!")

    print()
    print_divider("=")

    run_mimic_tests(pipeline)

    # Return exit code based on recall
    if overall_recall < 0.80:
        print("❌ FAIL: Recall below 80% — pipeline needs improvement.")
        return 1
    elif overall_recall < 0.95:
        print("⚠️  WARN: Recall between 80-95% — review missed entities above.")
        return 0
    else:
        print("✅ PASS: Recall ≥ 95% — pipeline is ready for MIMIC-IV data.")
        return 0


def run_mimic_tests(pipeline: DeidentificationPipeline, limit: int = 20):
    data_dir = os.environ.get("MIMIC_IV_NOTE_DIR")
    if not data_dir:
        print("\nSkipping MIMIC-IV tests (MIMIC_IV_NOTE_DIR not set)")
        return

    loader = MIMICNoteLoader(data_dir)
    status = loader.check_data_status()
    available = [name for name, info in status.items() if info["found"]]
    if not available:
        print(f"\nMIMIC_IV_NOTE_DIR={data_dir} set but no files found — skipping")
        return

    print_divider("=")
    print("  MIMIC-IV NOTE TESTS")
    print_divider("=")
    print(f"\nData dir: {data_dir}")
    for name, info in status.items():
        icon = "✅" if info["found"] else "❌"
        print(f"  {icon} {name}: {info['path'] or 'NOT FOUND'}")

    notes = loader.load_all_notes(limit_per_type=limit)
    print(f"\nRunning pipeline on {len(notes)} notes (up to {limit} per type)...")

    start = time.time()
    results = pipeline.deidentify_batch([n["text"] for n in notes], show_progress=False)
    elapsed = time.time() - start

    phi_count = sum(len(r.entities) for r in results)
    phi_notes = sum(1 for r in results if r.phi_found)
    print(f"  Notes processed:    {len(results)}")
    print(f"  Notes with PHI:     {phi_notes}")
    print(f"  Total PHI entities: {phi_count}")
    print(f"  Total time:         {elapsed:.2f}s  ({elapsed/len(results):.3f}s/note)")
    print()
    print_divider("=")


if __name__ == "__main__":
    sys.exit(run_tests())
