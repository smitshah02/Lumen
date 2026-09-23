"""Case-source boundary for the answer-level evaluator.

The frozen demo and profile-bound Synthea providers implement one protocol.
Research is intentionally not registered. Collection, scoring, judging,
aggregation, and artifact logic remain provider-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.evals.final_eval import cases as case_mod


@runtime_checkable
class CaseProvider(Protocol):
    name: str
    data_plane: str
    profile: str | None
    expected_database: str

    def load_cases(self) -> list[case_mod.EvalCase]: ...

    def fingerprint(self) -> dict: ...


@dataclass(frozen=True)
class DemoCaseProvider:
    name: str = "demo"
    data_plane: str = "demo"
    profile: str | None = None
    expected_database: str = "lumen_demo"

    def load_cases(self) -> list[case_mod.EvalCase]:
        return case_mod.load_cases()

    def fingerprint(self) -> dict:
        return {
            **case_mod.dataset_fingerprint(),
            "provider": self.name,
            "provider_data_plane": self.data_plane,
        }


@dataclass(frozen=True)
class SyntheaCaseProvider:
    profile: str
    name: str = "synthea"
    data_plane: str = "synthea"
    expected_database: str = "lumen_synthea"

    def __post_init__(self):
        if self.profile not in ("dev", "eval"):
            raise ValueError("Synthea provider profile must be dev or eval")

    def _root(self) -> Path:
        return case_mod.ROOT / "data" / "synthea" / self.profile

    def load_cases(self) -> list[case_mod.EvalCase]:
        from src.evals.final_eval.synthea_cases import artifact_paths, validate_artifacts
        validate_artifacts(self.profile, self._root())
        golden, _ = artifact_paths(self.profile, self._root())
        return case_mod.load_cases(golden)

    def fingerprint(self) -> dict:
        from src.evals.final_eval.synthea_cases import validate_artifacts
        return {**validate_artifacts(self.profile, self._root()),
                "provider": self.name, "provider_profile": self.profile,
                "provider_data_plane": self.data_plane,
                "expected_database": self.expected_database}


_PROVIDERS: dict[str, CaseProvider] = {"demo": DemoCaseProvider()}


def provider_names() -> tuple[str, ...]:
    return ("demo", "synthea")


def get_provider(name: str = "demo", profile: str | None = None) -> CaseProvider:
    if name == "synthea":
        return SyntheaCaseProvider(profile or "dev")
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown case provider {name!r}; available: {', '.join(provider_names())}"
        ) from None
