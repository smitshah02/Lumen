"""
MIMIC-IV Note Adapter
======================
Loads clinical notes from MIMIC-IV-Note discharge summaries and radiology
reports, then runs them through the de-identification pipeline.

This module is ready to use the moment your MIMIC-IV data lands.
Just point MIMIC_IV_NOTE_DIR to the extracted folder.

Usage:
    from src.deid.mimic_adapter import MIMICNoteLoader

    loader = MIMICNoteLoader("data/mimic-iv-note")
    notes = loader.load_discharge_summaries(limit=100)
    # notes is a list of dicts with keys: subject_id, hadm_id, note_type, text

    # To de-identify:
    from src.deid.pipeline import DeidentificationPipeline
    pipeline = DeidentificationPipeline()
    for note in notes:
        result = pipeline.deidentify(note["text"])
        note["deid_text"] = result.text
        note["phi_entities"] = result.entities
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MIMICNoteLoader:
    """
    Loads clinical notes from MIMIC-IV-Note on disk.

    Expected directory structure (after extracting the PhysioNet download):
        mimic-iv-note/
        ├── note/
        │   ├── discharge.csv.gz      (or discharge.csv)
        │   └── radiology.csv.gz      (or radiology.csv)

    MIMIC-IV-Note discharge.csv columns:
        note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text

    MIMIC-IV-Note radiology.csv columns:
        note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            logger.warning(
                f"MIMIC-IV-Note directory not found: {self.base_dir}. "
                "Download from PhysioNet once credentialed."
            )

    def _find_file(self, name: str) -> Optional[Path]:
        """Find a CSV or CSV.GZ file in the note directory."""
        candidates = [
            self.base_dir / "note" / f"{name}.csv.gz",
            self.base_dir / "note" / f"{name}.csv",
            self.base_dir / f"{name}.csv.gz",
            self.base_dir / f"{name}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_csv(
        self, filename: str, limit: Optional[int] = None
    ) -> list[dict]:
        """Load rows from a MIMIC-IV-Note CSV file."""
        path = self._find_file(filename)
        if path is None:
            logger.error(
                f"Could not find {filename}.csv or {filename}.csv.gz "
                f"in {self.base_dir}. Make sure MIMIC-IV-Note is extracted."
            )
            return []

        rows = []
        opener = gzip.open if path.suffix == ".gz" else open
        mode = "rt"

        logger.info(f"Loading {path}...")
        with opener(path, mode, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                rows.append(
                    {
                        "note_id": row.get("note_id", ""),
                        "subject_id": row.get("subject_id", ""),
                        "hadm_id": row.get("hadm_id", ""),
                        "note_type": row.get("note_type", filename),
                        "charttime": row.get("charttime", ""),
                        "text": row.get("text", ""),
                    }
                )

        logger.info(f"Loaded {len(rows)} {filename} notes")
        return rows

    def load_discharge_summaries(
        self, limit: Optional[int] = None
    ) -> list[dict]:
        """Load discharge summary notes."""
        return self._load_csv("discharge", limit=limit)

    def load_radiology_reports(
        self, limit: Optional[int] = None
    ) -> list[dict]:
        """Load radiology report notes."""
        return self._load_csv("radiology", limit=limit)

    def load_all_notes(
        self, limit_per_type: Optional[int] = None
    ) -> list[dict]:
        """Load all available note types."""
        notes = []
        notes.extend(self.load_discharge_summaries(limit=limit_per_type))
        notes.extend(self.load_radiology_reports(limit=limit_per_type))
        return notes

    def check_data_status(self) -> dict:
        """Check which MIMIC-IV-Note files are available."""
        status = {}
        for name in ["discharge", "radiology"]:
            path = self._find_file(name)
            status[name] = {
                "found": path is not None,
                "path": str(path) if path else None,
            }
        return status


# ---------------------------------------------------------------------------
# Quick CLI check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    data_dir = os.environ.get("MIMIC_IV_NOTE_DIR", "data/mimic-iv-note")
    loader = MIMICNoteLoader(data_dir)

    print("MIMIC-IV-Note data status:")
    status = loader.check_data_status()
    for note_type, info in status.items():
        icon = "✅" if info["found"] else "❌"
        print(f"  {icon} {note_type}: {info['path'] or 'NOT FOUND'}")

    if any(s["found"] for s in status.values()):
        print("\nLoading first 3 notes as a sample...")
        notes = loader.load_all_notes(limit_per_type=3)
        for note in notes[:3]:
            print(f"\n  [{note['note_type']}] subject={note['subject_id']}")
            print(f"  Text preview: {note['text'][:200]}...")
    else:
        print(
            "\n⏳ No MIMIC-IV-Note data found yet."
            "\n   Once PhysioNet approves your access, extract the files to:"
            f"\n   {data_dir}/note/discharge.csv.gz"
            f"\n   {data_dir}/note/radiology.csv.gz"
            "\n   Then re-run this script."
        )
