"""
Data Planes
===========
Lumen's MCP server can serve two disjoint corpora:

  demo      synthetic patients, shipped in this file. Safe to expose to any
            MCP host, including cloud-hosted ones.

  research  the real MIMIC-IV cohort. PhysioNet's DUA prohibits sharing this
            data with third parties, so this plane REFUSES TO START on any
            transport that could carry it off the machine. stdio + localhost
            only, no exceptions, enforced below rather than documented.

Rule that applies to both planes: never read clinical_notes.text_original.
That column holds raw MIMIC text. De-identified text lives in text_deid and
note_chunks.chunk_text; those are the only readable sources.
"""

from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)

PLANE = os.environ.get("LUMEN_DATA_PLANE", "research").strip().lower()
VALID_PLANES = {"demo", "research"}


class PlaneViolation(RuntimeError):
    pass


def enforce_transport(transport: str, host: str = "127.0.0.1") -> None:
    """Called before the server binds. Raises rather than degrading."""
    if PLANE not in VALID_PLANES:
        raise PlaneViolation(f"LUMEN_DATA_PLANE={PLANE!r}; expected one of {sorted(VALID_PLANES)}")

    if PLANE == "research":
        if transport != "stdio":
            raise PlaneViolation(
                f"research plane refuses transport={transport!r}. The real MIMIC cohort "
                f"may only be served over stdio on this machine. Use "
                f"LUMEN_DATA_PLANE=demo for anything else."
            )
        if host not in ("127.0.0.1", "localhost", ""):
            raise PlaneViolation(f"research plane refuses host={host!r}")

    logger.info(f"data plane: {PLANE} (transport={transport})")


def is_demo() -> bool:
    return PLANE == "demo"


def banner() -> str:
    return ("SYNTHETIC DEMO DATA — not real patients. "
            if is_demo() else
            "De-identified MIMIC-IV research data. Do not redistribute. ")


# ===========================================================================
# Synthetic demo corpus
# ===========================================================================
# Entirely fabricated. No relationship to any real person or MIMIC record.
# Exists so the server can be demonstrated against a cloud MCP host without
# any DUA-covered data leaving this machine.

DEMO_NOTES = [
    {
        "chunk_id": 9001, "subject_id": 90001, "note_type": "discharge",
        "charttime": "2024-03-14 09:00:00",
        "chunk_text": (
            "SYNTHETIC RECORD. Discharge summary. 64-year-old with type 2 diabetes and "
            "stage 3b chronic kidney disease (eGFR 38). Admitted with volume overload. "
            "Diuresed with IV furosemide. Discharge potassium 5.4 mEq/L. Lisinopril held "
            "on admission, restarted at reduced dose. Follow-up chemistry in one week."
        ),
    },
    {
        "chunk_id": 9002, "subject_id": 90001, "note_type": "discharge",
        "charttime": "2024-03-14 09:00:00",
        "chunk_text": (
            "SYNTHETIC RECORD. Discharge medications: metformin 500 mg PO BID, "
            "lisinopril 5 mg PO daily, furosemide 40 mg PO daily, atorvastatin 40 mg "
            "PO nightly, empagliflozin 10 mg PO daily."
        ),
    },
    {
        "chunk_id": 9003, "subject_id": 90002, "note_type": "radiology",
        "charttime": "2024-05-02 14:20:00",
        "chunk_text": (
            "SYNTHETIC RECORD. Chest radiograph. INDICATION: dyspnea. FINDINGS: mild "
            "pulmonary vascular congestion, small bilateral pleural effusions. No focal "
            "consolidation. IMPRESSION: findings consistent with volume overload."
        ),
    },
]

DEMO_LABS = {
    (90001, "potassium"): [
        ("2024-03-10 06:00:00", 5.9, "mEq/L", "abnormal"),
        ("2024-03-12 06:00:00", 5.6, "mEq/L", "abnormal"),
        ("2024-03-14 06:00:00", 5.4, "mEq/L", "abnormal"),
    ],
    (90001, "creatinine"): [
        ("2024-03-10 06:00:00", 1.9, "mg/dL", "abnormal"),
        ("2024-03-14 06:00:00", 1.7, "mg/dL", "abnormal"),
    ],
}

DEMO_TIMELINE = {
    90001: [
        {"when": "2024-03-09 22:15:00", "kind": "admission", "detail": "Admitted: volume overload"},
        {"when": "2024-03-10 06:00:00", "kind": "lab", "detail": "Potassium 5.9 mEq/L (abnormal)"},
        {"when": "2024-03-14 09:00:00", "kind": "note", "detail": "discharge summary"},
    ],
    90002: [
        {"when": "2024-05-02 14:20:00", "kind": "note", "detail": "radiology report"},
    ],
}