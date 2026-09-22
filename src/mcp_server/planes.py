"""
Data Planes
===========
Lumen's MCP server can serve two disjoint database planes:

  demo      synthetic patients loaded by scripts/load_synthetic_demo.py.
            Safe to expose to any MCP host, including cloud-hosted ones.

  research  the real MIMIC-IV cohort. PhysioNet's DUA prohibits sharing this
            data with third parties, so this plane REFUSES TO START on any
            transport that could carry it off the machine. stdio + localhost
            only, no exceptions, enforced below rather than documented.

Rule that applies to both planes: never read clinical_notes.text_original.
That column holds raw MIMIC text. De-identified text lives in text_deid and
note_chunks.chunk_text; those are the only readable sources.
"""

from __future__ import annotations

import logging

from src.config import DATA_PLANE as PLANE, VALID_PLANES

logger = logging.getLogger(__name__)

class PlaneViolation(RuntimeError):
    pass


def enforce_transport(transport: str, host: str = "127.0.0.1") -> None:
    """Called before the server binds. Raises rather than degrading."""
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
