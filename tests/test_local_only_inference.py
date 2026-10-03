"""Clinical text is only ever sent to a local model.

No tracked Python file imports a hosted LLM SDK, local model calls ignore proxy
settings, every judge endpoint is validated, and the research facade pins the
local-only settings on every command."""

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOSTED = {"groq", "openai", "anthropic", "cohere", "mistralai", "litellm", "together",
          "google.generativeai", "google.genai", "langchain_openai", "langchain_groq",
          "langchain_anthropic", "langchain_google_genai", "langchain_mistralai", "langchain_cohere"}


def _imported_modules(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _is_hosted(module: str) -> bool:
    return any(module == h or module.startswith(h + ".") for h in HOSTED)


@pytest.mark.parametrize("line", ["from langchain_openai import ChatOpenAI", "import os, openai",
                                  "from google import generativeai", "import litellm",
                                  "from groq import Groq"])
def test_detector_catches_direct_and_indirect_sdk_imports(line):
    assert any(_is_hosted(m) for m in _imported_modules(line))


def test_no_tracked_python_file_imports_a_hosted_llm_sdk():
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    offenders = [f for f in files if (ROOT / f).is_file()
                 and any(_is_hosted(m) for m in _imported_modules((ROOT / f).read_text(encoding="utf-8")))]
    assert offenders == []


def test_no_hosted_llm_package_is_required():
    pattern = re.compile(r"^(" + "|".join(re.escape(h).replace("_", "[-_]").replace(r"\.", "[-_.]")
                                          for h in HOSTED) + r")\b", re.M | re.I)
    for name in ("requirements.txt", "requirements-dev.txt"):
        assert not pattern.search((ROOT / name).read_text()), name


def test_retrieval_judge_has_no_default_backend(tmp_path):
    from src.evals.llm_judge import LLMJudge
    judge = LLMJudge(cache_path=str(tmp_path / "cache.json"))
    with pytest.raises(RuntimeError, match="no call_fn"):
        judge._call([{"role": "user", "content": "x"}])
    assert not hasattr(judge, "api_key")


def test_local_model_calls_ignore_proxy_environment():
    from src.llm import local_client
    assert local_client.HTTP.trust_env is False


def test_ollama_judge_refuses_a_remote_host_on_the_research_plane(monkeypatch):
    from src import config
    from src.evals import ollama_backend
    monkeypatch.setenv("LUMEN_DATA_PLANE", "research")
    monkeypatch.delenv("LUMEN_RESEARCH_ALLOW_REMOTE_MODELS", raising=False)
    with pytest.raises(config.ConfigurationError):
        ollama_backend.make_ollama_call_fn(host="https://api.example.com")


def test_research_facade_pins_local_only_settings_on_every_command():
    text = (ROOT / "scripts" / "lumen").read_text()
    research = text.split('if [ "$plane" = "research" ]; then')[1].split("fi")[0]
    for pin in ("LUMEN_TRACING=0", "LUMEN_RESEARCH_ALLOW_REMOTE_MODELS=0", "HF_HUB_OFFLINE=1"):
        assert pin in research


def test_research_start_refuses_to_override_the_loopback_bind():
    result = subprocess.run([str(ROOT / "scripts" / "lumen"), "research", "start", "--host", "0.0.0.0"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "loopback" in result.stderr


def test_research_facade_drops_uvicorn_socket_overrides():
    text = (ROOT / "scripts" / "lumen").read_text()
    research = text.split('if [ "$plane" = "research" ]; then')[1].split("fi")[0]
    assert "unset UVICORN_FD UVICORN_UDS UVICORN_HOST" in research


def test_research_tracing_refuses_to_run_through_a_proxy(monkeypatch):
    from src.obs import tracing
    monkeypatch.setattr(tracing, "PLANE", "research")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    assert tracing._policy()[0] is False
