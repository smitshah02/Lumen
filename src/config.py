"""Small, side-effect-free configuration boundary for Lumen.

Environment loading happens in :mod:`src`; this module only validates values
and resolves repository-relative paths.  Keep operational policy here, while
feature-specific knobs stay beside the feature that owns them.
"""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(RuntimeError):
    """A configured value would make the selected runtime unsafe or invalid."""


VALID_PLANES = frozenset({"demo", "research"})

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LUMEN_DATA_DIR", REPO_ROOT / "data")).expanduser()
MIMIC_IV_DIR = Path(os.environ.get("MIMIC_IV_DIR", DATA_DIR / "mimiciv")).expanduser()
MIMIC_NOTE_DIR = Path(
    os.environ.get("MIMIC_IV_NOTE_DIR", DATA_DIR / "mimic-iv-note")
).expanduser()
MODELS_DIR = Path(os.environ.get("LUMEN_MODELS_DIR", REPO_ROOT / "models")).expanduser()
GUIDELINES_DIR = Path(
    os.environ.get("LUMEN_GUIDELINES_DIR", DATA_DIR / "guidelines")
).expanduser()

MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.json"


def _load_models_config() -> dict:
    try:
        config = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
        if config.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        return config
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid model manifest {MODELS_CONFIG_PATH}: {exc}") from exc


MODELS_CONFIG = _load_models_config()
LLM_HOST_DEFAULT = "http://localhost:11434"
MAIN_MODEL_DEFAULT = MODELS_CONFIG["ollama"]["runtime_main"]
FAST_MODEL_DEFAULT = MODELS_CONFIG["ollama"]["runtime_fast"]
JUDGE_MODEL_DEFAULT = MODELS_CONFIG["ollama"]["independent_judge"]
PGVECTOR_VERSION = "0.8.6"

_TRUE = frozenset({"1", "true", "yes", "on"})
_LOCAL_NAMES = frozenset({"localhost", "host.docker.internal"})


def get_data_plane(value: str | None = None) -> str:
    """Return a validated data plane, reading the environment when omitted."""
    plane = (value if value is not None else os.environ.get("LUMEN_DATA_PLANE", "research"))
    plane = plane.strip().lower()
    if plane not in VALID_PLANES:
        raise ConfigurationError(
            f"LUMEN_DATA_PLANE={plane!r}; expected one of {sorted(VALID_PLANES)}"
        )
    return plane


DATA_PLANE = get_data_plane()


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def is_local_endpoint(endpoint: str) -> bool:
    """Whether an HTTP endpoint resolves by an explicitly local host name/IP."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_model_endpoint(
    endpoint: str,
    *,
    plane: str | None = None,
    allow_remote: bool | None = None,
) -> str:
    """Validate a model endpoint against the research-plane egress policy.

    Research inference is local by default. A deliberately named override is
    available for controlled environments, but there is no implicit exception
    for generic private-network or container host names.
    """
    value = endpoint.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError(f"invalid model endpoint {endpoint!r}; expected an http(s) URL")
    selected_plane = get_data_plane(plane)
    remote_allowed = (
        _env_true("LUMEN_RESEARCH_ALLOW_REMOTE_MODELS")
        if allow_remote is None
        else allow_remote
    )
    if selected_plane == "research" and not is_local_endpoint(value) and not remote_allowed:
        raise ConfigurationError(
            f"research plane refuses remote model endpoint {parsed.hostname!r}; "
            "use a loopback/host.docker.internal endpoint or explicitly set "
            "LUMEN_RESEARCH_ALLOW_REMOTE_MODELS=1"
        )
    return value
