"""Deployment examples must agree with the one authoritative model registry."""

import json
from pathlib import Path

from src import config


ROOT = Path(__file__).resolve().parents[1]


def _example_env() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_python_and_deployment_model_tags_match_registry():
    registry = json.loads((ROOT / "configs/models.json").read_text())["ollama"]
    example = _example_env()
    assert config.MAIN_MODEL_DEFAULT == registry["runtime_main"] == example["LUMEN_LLM_MAIN"]
    assert config.FAST_MODEL_DEFAULT == registry["runtime_fast"] == example["LUMEN_LLM_FAST"]
    assert config.JUDGE_MODEL_DEFAULT == registry["independent_judge"]


def test_demo_compose_has_no_copied_model_fallbacks():
    compose = (ROOT / "docker-compose.demo.yml").read_text()
    assert "LUMEN_LLM_MAIN:?" in compose
    assert "LUMEN_LLM_FAST:?" in compose
    assert config.MAIN_MODEL_DEFAULT not in compose
    assert config.FAST_MODEL_DEFAULT not in compose
