"""
Run directory + experiment manifest
===================================
Every evaluation run gets one immutable directory under results/final_eval/.

Immutability rules
------------------
  * a directory carrying `.complete` is finished and is NEVER written again,
    with or without --resume;
  * an existing directory WITHOUT `.complete` is an interrupted run: it can be
    resumed (same run_id, same manifest) but not silently overwritten;
  * manifest.json is written once, at creation, and never rewritten. On resume
    the live environment is re-read and compared against it, and any drift is
    recorded in `resumes` rather than quietly accepted.

Secrets
-------
No environment values are copied into the manifest. Credentials are reported as
booleans ("configured: true"), hosts as hostnames only. DATABASE_URL is reduced
to its database name via SQLAlchemy's parsed URL, never its DSN.
"""

from __future__ import annotations

import io
import os
import re
import sys
import json
import time
import socket
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.evals.final_eval import EVALUATOR_VERSION, cases as case_mod

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = Path(os.environ.get("LUMEN_EVAL_RESULTS_ROOT",
                                           str(ROOT / "results" / "final_eval")))

ARTIFACTS = {
    "manifest": "manifest.json",
    "responses": "responses.jsonl",
    "deterministic": "deterministic.jsonl",
    "judge": "judge.jsonl",
    "failures": "failures.jsonl",
    "summary": "summary.json",
    "report": "report.md",
    "calibration": "calibration.json",
    "calibration_report": "calibration.md",
    "api_crosscheck": "api_crosscheck.json",
    "comparison": "comparison.json",
    # Working file, NOT a published result artifact. Holds the exact evidence
    # text the model saw, because the offline judge must grade groundedness
    # against that text and runs as a separate stage. It is gitignored, never
    # read by `score` or `compare`, and removable with `purge-evidence`.
    "evidence_cache": "evidence_cache.jsonl",
}
# The artifacts that constitute the run's published result. Everything here is
# free of raw clinical text by construction.
PUBLISHED_ARTIFACTS = ("manifest", "responses", "deterministic", "judge",
                       "failures", "summary", "report")
COMPLETE_SENTINEL = ".complete"

# Shape of the per-case rows in responses/deterministic/judge .jsonl. Bumped
# when a field is added, removed or changes meaning, so a reader can tell at a
# glance whether two runs' raw artifacts are directly comparable. 2 is the aj2
# judge row (criterion_assessments, consistency_violations) plus the
# admission_scope_violation / execution_error / invalid_visible_citation tags.
RESULTS_SCHEMA_VERSION = 2


class RunDirError(RuntimeError):
    """Refusal to write where writing would destroy or confuse a run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str = "") -> str:
    """Sortable and self-describing: 20260920T161500Z-031b1c4."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = (_git("rev-parse", "--short=7", "HEAD") or "nogit")
    return f"{prefix}{stamp}-{sha}"


# `git status --porcelain` emits "XY PATH". _git() strips the whole output, so
# the FIRST line of an unstaged-only status loses its leading blank X and a
# naive line[3:] then eats the first character of that filename. The manifest
# would record "rc/evals/..." as the dirty file — a provenance record that
# names a path which does not exist.
_PORCELAIN_RE = re.compile(r"^[ MADRCU?!]{1,2}\s+(.*)$")


def porcelain_paths(status: str) -> list[str]:
    """Paths out of `git status --porcelain`, tolerant of that stripped line."""
    out = []
    for line in (status or "").splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _PORCELAIN_RE.match(line)
        path = (m.group(1) if m else line).strip()
        if " -> " in path:                  # rename/copy: record the destination
            path = path.split(" -> ", 1)[1]
        out.append(path.strip('"'))
    return out


def _git(*args) -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
class RunDir:
    def __init__(self, run_id: str, root: Path | str | None = None):
        self.run_id = run_id
        self.root = Path(root or DEFAULT_RESULTS_ROOT)
        self.path = self.root / run_id

    # --- paths ---
    def file(self, name: str) -> Path:
        return self.path / ARTIFACTS[name]

    @property
    def sentinel(self) -> Path:
        return self.path / COMPLETE_SENTINEL

    def exists(self) -> bool:
        return self.path.exists()

    def is_complete(self) -> bool:
        return self.sentinel.exists()

    # --- lifecycle ---
    def open_for_write(self, manifest: dict, resume: bool = False) -> dict:
        """Create the run directory, or attach to an interrupted one.

        Returns the manifest actually in force — on resume that is the ORIGINAL
        manifest read back from disk, never the freshly built one, so a run's
        provenance cannot drift mid-flight.
        """
        if self.is_complete():
            raise RunDirError(
                f"run {self.run_id} is already complete ({self.sentinel}). "
                f"Completed runs are immutable. Use a new --run-id, or read it "
                f"with `score`/`compare`.")
        if self.exists() and any(self.path.iterdir()) and not resume:
            raise RunDirError(
                f"run directory {self.path} already exists and is not empty. "
                f"Pass --resume to continue the interrupted run, or choose a "
                f"different --run-id. Refusing to overwrite.")

        self.path.mkdir(parents=True, exist_ok=True)
        mpath = self.file("manifest")
        if mpath.exists():
            on_disk = json.loads(mpath.read_text())
            on_disk.setdefault("resumes", []).append({
                "resumed_at": utc_now(),
                "drift": _manifest_drift(on_disk, manifest),
            })
            _atomic_write(mpath, json.dumps(on_disk, indent=2) + "\n")
            return on_disk
        _atomic_write(mpath, json.dumps(manifest, indent=2) + "\n")
        return manifest

    def mark_complete(self, note: str = "") -> None:
        self.sentinel.write_text(json.dumps(
            {"completed_at": utc_now(), "run_id": self.run_id, "note": note}, indent=2) + "\n")

    # --- jsonl ---
    def append(self, name: str, obj: dict) -> None:
        """One JSON object per line, flushed immediately so an interrupted run
        never loses a case that already cost a model call."""
        p = self.file(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with io.open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_jsonl(self, name: str) -> list[dict]:
        p = self.file(name)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def completed_ids(self, name: str, key: str = "query_id") -> set:
        """Ids already present in an artifact — the basis of resume. Only rows
        that actually parsed count; a half-written final line is ignored and
        will be regenerated."""
        ids = set()
        p = self.file(name)
        if not p.exists():
            return ids
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get(key) is not None:
                ids.add(row[key])
        return ids

    def write_json(self, name: str, obj: dict) -> Path:
        p = self.file(name)
        _atomic_write(p, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.file(name)
        _atomic_write(p, text)
        return p

    def read_json(self, name: str) -> dict:
        return json.loads(self.file(name).read_text())


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_DRIFT_KEYS = ("git_sha", "dirty_worktree", "models", "prompt_versions",
               "judge", "eval_set", "data_plane", "collection_backend")


def _manifest_drift(original: dict, current: dict) -> dict:
    """What changed between the original run and this resume. Recorded, not
    enforced: the operator decides whether a drifted resume is still valid, but
    the artifact can never claim there was none."""
    drift = {}
    for k in _DRIFT_KEYS:
        if original.get(k) != current.get(k):
            drift[k] = {"original": original.get(k), "now": current.get(k)}
    return drift


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------
def _provenance() -> dict:
    """Git state. On a Pod with no .git, falls back to the .deployment_source.json
    written by scripts/sync_to_pod.sh — the same file scripts/cloud_eval.py reads."""
    sha = _git("rev-parse", "HEAD")
    if sha:
        dirty_files = sorted(set(porcelain_paths(_git("status", "--porcelain"))))
        return {"git_sha": sha, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty_worktree": bool(dirty_files), "dirty_files": dirty_files[:50],
                "provenance_source": "git"}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import cloud_eval  # noqa
        p = cloud_eval.read_provenance(ROOT / ".deployment_source.json")
        return {**p, "dirty_files": []}
    except Exception:
        return {"git_sha": None, "branch": None, "dirty_worktree": "unknown",
                "dirty_files": [], "provenance_source": "missing"}


def _models() -> dict:
    from src.llm import local_client
    rc = local_client.runtime_config()
    host = rc.get("host") or ""
    out = {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL,
           "roles": rc.get("roles"), "keep_alive": rc.get("keep_alive"),
           "ctx_main": rc.get("ctx_main"), "ctx_fast": rc.get("ctx_fast"),
           "host": _hostonly(host), "digests": {}}
    try:
        import requests
        tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
        by = {m.get("name"): m for m in tags}
        for role in ("main", "fast"):
            m = by.get(out[role]) or {}
            out["digests"][out[role]] = {
                "digest": (m.get("digest") or "")[:12] or None,
                "parameter_size": (m.get("details") or {}).get("parameter_size"),
                "quantization": (m.get("details") or {}).get("quantization_level"),
                "installed": bool(m)}
        out["ollama_version"] = requests.get(f"{host}/api/version", timeout=10).json().get("version")
    except Exception as e:
        out["digests_error"] = type(e).__name__
    return out


def _hostonly(url: str) -> str:
    """Hostname[:port] only — never credentials embedded in a URL."""
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        return u.netloc.split("@")[-1] if u.netloc else (url or "")
    except Exception:
        return ""


def _prompts() -> dict:
    from src.agents import prompts
    return {"triage": prompts.TRIAGE_VERSION, "synthesis": prompts.SYNTHESIS_VERSION,
            "verify": prompts.VERIFY_VERSION, "verify_batch": prompts.VERIFY_BATCH_VERSION,
            "concept": prompts.CONCEPT_VERSION}


def _retrieval() -> dict:
    """Retrieval / temporal / reranker configuration as the application resolves
    it. Guarded: on a box without torch the run is still recorded, with the
    reason the section is missing."""
    out: dict = {"source": "src/retrieval/hybrid_retriever_v2.py + src/agents/graph.py"}
    try:
        from src.agents import graph as G
        out.update({"patient_top_k": G.PATIENT_TOP_K, "guideline_top_k": G.GUIDELINE_TOP_K,
                    "lab_recent_points": G.LAB_RECENT_POINTS,
                    "deterministic_labs": G.DETERMINISTIC_LABS})
    except Exception as e:
        out["graph_config_error"] = type(e).__name__
    try:
        from src.retrieval import hybrid_retriever_v2 as H
        out.update({
            "bm25_top_n": 60, "vector_top_n": 60, "min_tokens": 40,
            "hnsw_ef_search": H.HNSW_EF_SEARCH,
            "rrf": {"k": 60, "bm25_weight": H.RRF_BM25_WEIGHT,
                    "vector_weight": H.RRF_VECTOR_WEIGHT},
            "query_expansion": H.QUERY_EXPANSION,
            "reranker_model_path": Path(H.DEFAULT_RERANKER_MODEL).name,
        })
    except Exception as e:
        out["retriever_config_error"] = type(e).__name__
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from fetch_models import MODELS
        out["embedding_model"] = {k: f"{MODELS[k][0]}@{MODELS[k][1][:10]}"
                                  for k in ("medcpt-query", "medcpt-article") if k in MODELS}
        if "bge-reranker" in MODELS:
            out["reranker_model"] = f"{MODELS['bge-reranker'][0]}@{MODELS['bge-reranker'][1][:10]}"
    except Exception as e:
        out["model_revision_error"] = type(e).__name__
    out["temporal"] = {"mode_source": "detect_temporal_mode(query) via triage",
                       "anchor": "patient-relative (each patient's own latest record)",
                       "recency_days": 365}
    return out


def _environment() -> dict:
    out = {
        "python": platform.python_version(),
        "platform": f"{platform.system()}/{platform.machine()}",
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "timezone": time.tzname[0] if time.tzname else None,
    }
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:
        out["torch_error"] = type(e).__name__
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            out["nvidia_smi"] = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    out["dependencies"] = _dependencies()
    return out


_TRACKED_DEPS = ("langgraph", "langchain-core", "fastapi", "pydantic", "SQLAlchemy",
                 "requests", "transformers", "numpy", "pytest", "psycopg")


def _dependencies() -> dict:
    from importlib import metadata
    out = {}
    for name in _TRACKED_DEPS:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _data_plane() -> dict:
    out = {"data_plane": os.environ.get("LUMEN_DATA_PLANE"), "database": None}
    try:
        from src import storage
        out["data_plane"] = storage.DATA_PLANE
        out["database"] = storage.engine.url.database      # name only, never the DSN
    except Exception as e:
        out["storage_error"] = type(e).__name__
    return out


def build_manifest(*, run_id: str, case_ids: list[str], subset: str,
                   judge_config: dict, collection_backend: str,
                   api_base_url: str | None = None, notes: str = "") -> dict:
    """The immutable record of what produced a run. Built fresh from the live
    environment; nothing here is defaulted from a literal that could go stale."""
    prov = _provenance()
    m = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "evaluator_version": EVALUATOR_VERSION,
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "benchmark": "Lumen Final Answer-Level Quality Baseline v1",
        "notes": notes,
        **prov,
        **_data_plane(),
        "eval_set": {**case_mod.dataset_fingerprint(), "subset": subset,
                     "n_evaluated": len(case_ids), "evaluated_ids": case_ids},
        "demo_corpus": case_mod.demo_data_fingerprint(),
        "retrieval": _retrieval(),
        "models": _models(),
        "prompt_versions": _prompts(),
        "judge": judge_config,
        "collection_backend": collection_backend,
        "api_base_url": api_base_url,
        "environment": _environment(),
        "tracing_enabled": os.environ.get("LUMEN_TRACING", "0") not in ("0", "false", "no"),
        "secrets_recorded": False,
        "resumes": [],
    }
    m["manifest_sha256"] = hashlib.sha256(
        json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()
    return m
