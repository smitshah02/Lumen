"""Case-source boundary for the answer-level evaluator.

The frozen demo provider implements the protocol. Research is intentionally
not registered. Collection, scoring, judging,
aggregation, and artifact logic remain provider-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
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


_PROVIDERS: dict[str, CaseProvider] = {"demo": DemoCaseProvider()}


def provider_names() -> tuple[str, ...]:
    return tuple(_PROVIDERS)


def get_provider(name: str = "demo") -> CaseProvider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown case provider {name!r}; available: {', '.join(provider_names())}"
        ) from None
