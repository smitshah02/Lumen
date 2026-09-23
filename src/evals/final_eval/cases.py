"""
The evaluation case set
=======================
Reads the FROZEN 40-case gold set at src/demo_data/golden_qa.json and
normalizes it into typed cases. It never writes to that file, never invents a
case, and never invents a label the source does not support.

Source of truth
---------------
  cases          src/demo_data/golden_qa.json          (40 cases)
  integrity      src/demo_data/manifest.json           (sha256 per file)
  legacy subset  demo_q01..demo_q15                    (mirrors
                 scripts/performance_eval.py LEGACY_IDS, which is the ORIGINAL
                 15-question demo set from before the corpus grew to 40 — it
                 is a historical subset, not a designed calibration sample)

Fields the source does NOT carry, and which are therefore NOT modelled:
  * expected_route       — inferable for the deterministic-lab path, but not
                           gold, so routing is recorded and never scored
  * severity/criticality — absent from the source
  * per-fact typing      — derived here from the fact's own content, below
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from src.evals.final_eval import normalize as N

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = ROOT / "src" / "demo_data" / "golden_qa.json"
DEMO_MANIFEST_PATH = ROOT / "src" / "demo_data" / "manifest.json"

# The original 15-question demo set. Mirrors scripts/performance_eval.py:LEGACY_IDS
# so the legacy comparison and the calibration subset address the same cases.
LEGACY_IDS = tuple(f"demo_q{i:02d}" for i in range(1, 16))

# The 3-case smoke triple, one per execution path:
#   demo_q01  simple/direct   -> deterministic lab_lookup, zero LLM calls
#   demo_q02  temporal        -> creatinine trend, three chronological values
#   demo_q19  complex         -> multi-note synthesis on the MAIN tier
SMOKE_IDS = ("demo_q01", "demo_q02", "demo_q19")


@dataclass(frozen=True)
class GoldFact:
    """One expected fact, with its deterministic anchors pre-parsed.

    `kind` decides how the fact can fail:
      structured  carries a quantity, date or bare number. A wrong value for
                  the same measurement is a deterministic CONTRADICTION.
      text_only   prose. It can be confirmed by exact/normalized match, but a
                  non-match only means "not deterministically confirmed" — it
                  is never, on its own, evidence that the answer is wrong.
    """
    text: str
    kind: str                                    # "structured" | "text_only"
    quantities: tuple = ()
    dates: tuple = ()
    bare_numbers: tuple = ()
    terms: tuple = ()

    @staticmethod
    def build(text: str) -> "GoldFact":
        qty = tuple(N.parse_quantities(text))
        dates = tuple(N.parse_dates(text))
        nums = tuple(N.parse_bare_numbers(text))
        terms = tuple(N.content_tokens(text))
        kind = "structured" if (qty or dates or nums) else "text_only"
        return GoldFact(text=text, kind=kind, quantities=qty, dates=dates,
                        bare_numbers=nums, terms=terms)


@dataclass(frozen=True)
class EvalCase:
    query_id: str
    subject_id: int
    query: str
    category: str
    answer_type: str
    difficulty: str
    temporal: str                       # "latest" | "trend" | "all"
    expected_facts: tuple               # tuple[GoldFact, ...]
    min_facts: int
    expected_answer: str
    unsupported: bool
    must_not_contain: tuple
    evidence_hadm_ids: tuple
    evidence_note_types: tuple
    evidence_note_ids: tuple = ()
    evidence_provenance: tuple = ()
    gold_profile: str | None = None

    # --- derived expectations -------------------------------------------
    @property
    def expects_abstention(self) -> bool:
        """The record genuinely does not contain the answer; the system must
        decline without fabricating a value or a citation."""
        return bool(self.unsupported)

    @property
    def expects_ambiguity(self) -> bool:
        """The record is conflicting or unresolved. A correct answer states the
        uncertainty; it does NOT have to use the hard refusal sentence."""
        return self.answer_type == "ambiguous"

    @property
    def temporal_applicable(self) -> bool:
        return self.temporal in ("latest", "earliest", "trend")

    @property
    def admission_scope_applicable(self) -> bool:
        return bool(self.evidence_hadm_ids)

    @property
    def n_structured_facts(self) -> int:
        return sum(1 for f in self.expected_facts if f.kind == "structured")


def load_cases(path: Path | str | None = None) -> list[EvalCase]:
    """Every case in the gold set, in file order."""
    p = Path(path or GOLDEN_PATH)
    raw = json.loads(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a list of cases, got {type(raw).__name__}")
    cases = [_to_case(r, p) for r in raw]
    ids = [c.query_id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{p}: duplicate query_id(s): {', '.join(dupes)}")
    return cases


def _to_case(r: dict, p: Path) -> EvalCase:
    missing = [k for k in ("id", "subject_id", "query") if k not in r]
    if missing:
        raise ValueError(f"{p}: case is missing {missing}: {str(r)[:120]}")
    return EvalCase(
        query_id=r["id"],
        subject_id=int(r["subject_id"]),
        query=r["query"],
        category=r.get("category", "unknown"),
        answer_type=r.get("answer_type", "unknown"),
        difficulty=r.get("difficulty", "unknown"),
        temporal=r.get("temporal", "all"),
        expected_facts=tuple(GoldFact.build(f) for f in (r.get("expected_facts") or [])),
        min_facts=int(r.get("min_facts", 0)),
        expected_answer=r.get("expected_answer", ""),
        unsupported=bool(r.get("unsupported", False)),
        must_not_contain=tuple(r.get("must_not_contain") or []),
        evidence_hadm_ids=tuple(r.get("evidence_hadm_ids") or []),
        evidence_note_types=tuple(r.get("evidence_note_types") or []),
        evidence_note_ids=tuple(r.get("evidence_note_ids") or []),
        evidence_provenance=tuple(r.get("evidence_provenance") or []),
        gold_profile=r.get("gold_profile"),
    )


def select(cases: list[EvalCase], ids=None, subset: str | None = None) -> list[EvalCase]:
    """The cases to evaluate, in the order asked for.

    An unknown id is a hard failure: silently running a different subset than
    the one requested makes every number in the report unattributable. This
    mirrors scripts/performance_eval.py:_select deliberately.
    """
    by_id = {c.query_id: c for c in cases}
    if ids:
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise SystemExit(
                f"unknown query id(s): {', '.join(unknown)}. "
                f"Known: {len(by_id)} ids from {min(by_id)} to {max(by_id)}")
        return [by_id[i] for i in ids]
    if subset in (None, "all", "full"):
        return list(cases)
    if subset in ("legacy15", "legacy", "calibration"):
        return select(cases, ids=[i for i in LEGACY_IDS if i in by_id])
    if subset == "smoke":
        return select(cases, ids=list(SMOKE_IDS))
    raise SystemExit(f"unknown subset {subset!r}; use all | legacy15 | smoke")


def dataset_fingerprint(path: Path | str | None = None) -> dict:
    """Version + hash of the evaluation set, so a run can never be attributed
    to a dataset it did not use. `manifest_sha256_matches` is False the moment
    the gold file is edited without regenerating the demo manifest."""
    p = Path(path or GOLDEN_PATH)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    recorded, version, seed = None, None, None
    try:
        dm = json.loads(DEMO_MANIFEST_PATH.read_text())
        recorded = (dm.get("sha256") or {}).get(p.name)
        version, seed = dm.get("version"), dm.get("seed")
    except Exception:
        pass
    cases = json.loads(data)
    return {
        "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
        "sha256": sha,
        "recorded_sha256": recorded,
        "manifest_sha256_matches": (recorded == sha) if recorded else None,
        "dataset_version": version,
        "generator_seed": seed,
        "n_cases": len(cases),
        "case_ids": [c.get("id") for c in cases],
    }


def demo_data_fingerprint() -> dict:
    """Row counts and per-file hashes of the synthetic corpus behind the cases —
    the closest practical stand-in for a database/index fingerprint."""
    try:
        dm = json.loads(DEMO_MANIFEST_PATH.read_text())
    except Exception as e:
        return {"available": False, "error": type(e).__name__}
    return {"available": True, "version": dm.get("version"), "seed": dm.get("seed"),
            "counts": dm.get("counts"), "sha256": dm.get("sha256")}
