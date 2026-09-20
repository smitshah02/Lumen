"""
RunPod preflight — `final_eval.py doctor`
=========================================
Everything that must be true BEFORE a benchmark is worth starting, checked in
one command that costs no GPU time and changes nothing.

It is strictly diagnostic. It starts no model, pulls nothing, creates no run
directory, writes no artifact and never touches the environment. The most it
does is open a read-only database connection and ask an Ollama host what it has
installed.

Profiles
--------
`final` (default) is the pre-benchmark contract: every prerequisite for a real
final run is REQUIRED, and a missing one exits non-zero.

`local` is for a development box. The environment-dependent checks — the 30B
main model, the GPU, the database, the reranker weights — are demoted to
advisory, because a Mac not having the cloud configuration is a fact about the
Mac, not a defect in the evaluator. Everything the repository itself controls
(dataset hash, case counts, judge independence, output writability) stays
required in both profiles.
"""

from __future__ import annotations

import os
import sys
import platform
from dataclasses import dataclass, field
from pathlib import Path

from src.evals.final_eval import EVALUATOR_VERSION, cases as case_mod
from src.evals.final_eval import manifest as man

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

# Checks whose outcome depends on the machine rather than on the repository.
# Demoted to advisory under --profile local.
ENVIRONMENT_CHECKS = frozenset({
    "database_connectivity", "demo_data_plane_rows", "llm_host_reachable",
    "runtime_model_main", "runtime_model_fast", "judge_model_installed",
    "embedding_models", "reranker_model", "gpu",
})
# `judge_independence` is deliberately NOT in that set. Judging the system with
# itself is a design error, not a property of the machine, so it blocks in
# every profile.

EXPECTED_CASES = 40
EXPECTED_LEGACY = 15
TARGET_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    value: object = None
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN, SKIP)


@dataclass
class Report:
    profile: str
    checks: list = field(default_factory=list)

    def add(self, *checks) -> None:
        self.checks.extend(c for c in checks if c is not None)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def blocking(self) -> list:
        return [c for c in self.failures if c.required]

    @property
    def exit_code(self) -> int:
        return 1 if self.blocking else 0

    def to_dict(self) -> dict:
        return {"profile": self.profile,
                "ready": not self.blocking,
                "exit_code": self.exit_code,
                "checks": [{"name": c.name, "status": c.status, "required": c.required,
                            "detail": c.detail, "value": c.value} for c in self.checks]}


def _c(name, ok, detail="", value=None, required=True, warn_only=False) -> Check:
    status = PASS if ok else (WARN if warn_only else FAIL)
    return Check(name, status, detail, value, required)


# ---------------------------------------------------------------------------
# Repository and interpreter
# ---------------------------------------------------------------------------
def check_repository() -> list:
    sha = man._git("rev-parse", "HEAD")
    branch = man._git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = sorted(set(man.porcelain_paths(man._git("status", "--porcelain"))))
    return [
        _c("git_sha", bool(sha),
           sha or "not a git checkout — provenance cannot be recorded", sha),
        Check("git_branch", PASS, f"on {branch or 'unknown'}", branch, required=False),
        # A dirty tree is not fatal (it is recorded in the manifest either way),
        # but a benchmark run from uncommitted code cannot be reproduced from a
        # SHA, so it is surfaced loudly.
        _c("clean_worktree", not dirty,
           "clean" if not dirty else f"{len(dirty)} modified file(s): {', '.join(dirty[:5])}"
                                     + (" …" if len(dirty) > 5 else ""),
           dirty, required=False, warn_only=True),
    ]


def check_python() -> list:
    v = sys.version_info
    matches = (v.major, v.minor) == TARGET_PYTHON
    return [
        Check("python_version",
              PASS if matches else WARN,
              f"{platform.python_version()}"
              + ("" if matches else f" — the cloud target is "
                                    f"{TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}"),
              platform.python_version(), required=False),
    ]


def check_versions(judge_prompt_version: str) -> list:
    return [
        Check("evaluator_version", PASS, EVALUATOR_VERSION, EVALUATOR_VERSION, required=False),
        Check("results_schema_version", PASS, str(man.RESULTS_SCHEMA_VERSION),
              man.RESULTS_SCHEMA_VERSION, required=False),
        Check("judge_prompt_version", PASS, judge_prompt_version,
              judge_prompt_version, required=False),
    ]


# ---------------------------------------------------------------------------
# The evaluation set
# ---------------------------------------------------------------------------
def check_dataset() -> list:
    out = []
    path = case_mod.GOLDEN_PATH
    if not path.exists():
        return [_c("eval_set_present", False, f"missing {path}", str(path))]
    fp = case_mod.dataset_fingerprint()
    out.append(_c("eval_set_present", True, str(path), str(path)))
    out.append(_c("eval_set_sha256", bool(fp["sha256"]), fp["sha256"][:16], fp["sha256"]))
    out.append(_c(
        "demo_manifest_hash_matches", fp["manifest_sha256_matches"] is True,
        ("golden_qa.json matches the hash recorded in src/demo_data/manifest.json"
         if fp["manifest_sha256_matches"] else
         f"recorded {str(fp['recorded_sha256'])[:16]} != actual {fp['sha256'][:16]} — the "
         f"gold set was edited without regenerating the demo manifest"),
        fp["manifest_sha256_matches"]))
    out.append(_c("eval_set_case_count", fp["n_cases"] == EXPECTED_CASES,
                  f"{fp['n_cases']} cases (expected {EXPECTED_CASES})", fp["n_cases"]))

    try:
        cases = case_mod.load_cases()
        legacy = case_mod.select(cases, subset="legacy15")
        out.append(_c("legacy_subset_count", len(legacy) == EXPECTED_LEGACY,
                      f"{len(legacy)} of {EXPECTED_LEGACY} calibration cases "
                      f"({case_mod.LEGACY_IDS[0]}..{case_mod.LEGACY_IDS[-1]}); this is the "
                      f"ORIGINAL demo subset, not a designed statistical sample",
                      [c.query_id for c in legacy]))
        out.append(_c("eval_set_parses", True,
                      f"{sum(len(c.expected_facts) for c in cases)} expected facts, "
                      f"{sum(1 for c in cases if c.expects_abstention)} unsupported, "
                      f"{sum(1 for c in cases if c.expects_ambiguity)} ambiguous",
                      None))
    except Exception as e:
        out.append(_c("eval_set_parses", False, f"{type(e).__name__}: {e}", None))
    return out


# ---------------------------------------------------------------------------
# Data plane
# ---------------------------------------------------------------------------
def check_data_plane(probe=None) -> list:
    """The evaluation runs against the synthetic demo plane only."""
    plane = os.environ.get("LUMEN_DATA_PLANE")
    out = [_c("data_plane_is_demo", plane == "demo",
              f"LUMEN_DATA_PLANE={plane!r}" +
              ("" if plane == "demo" else " — the final evaluation refuses any other plane"),
              plane)]
    info = (probe or _probe_database)()
    out.append(_c("database_connectivity", bool(info.get("connected")),
                  info.get("detail") or "", info.get("database")))
    rows = info.get("counts") or {}
    out.append(_c("demo_data_plane_rows", bool(rows.get("note_chunks")),
                  f"note_chunks={rows.get('note_chunks')} labevents={rows.get('labevents')}"
                  if rows else (info.get("counts_error") or "no row counts available"),
                  rows))
    return out


def _probe_database() -> dict:
    """Read-only. Returns a description instead of raising, so one unreachable
    database never masks the rest of the report."""
    try:
        from sqlalchemy import text as sa_text
        from src import storage
        with storage.engine.connect() as c:
            c.execute(sa_text("SELECT 1"))
            counts = {}
            for table in ("note_chunks", "labevents"):
                try:
                    counts[table] = c.execute(
                        sa_text(f"SELECT count(*) FROM {table}")).scalar()
                except Exception as e:
                    counts[table] = f"unavailable ({type(e).__name__})"
        return {"connected": True, "database": storage.engine.url.database,
                "detail": f"connected to {storage.engine.url.database} "
                          f"(plane {storage.DATA_PLANE})",
                "counts": counts}
    except Exception as e:
        return {"connected": False, "detail": f"{type(e).__name__}: {str(e)[:160]}",
                "database": None, "counts": {}}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _installed(tags: list, model: str) -> dict:
    from src.evals.final_eval.judge_backend import _norm_tag
    for m in tags or []:
        if _norm_tag(m.get("name")) == _norm_tag(model):
            return {"installed": True, "digest": (m.get("digest") or "")[:12],
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                    "quantization": (m.get("details") or {}).get("quantization_level")}
    return {"installed": False}


def probe_ollama(host: str) -> dict:
    try:
        import requests
        h = (host or "").rstrip("/")
        tags = requests.get(f"{h}/api/tags", timeout=10).json().get("models", [])
        return {"reachable": True, "models": tags}
    except Exception as e:
        return {"reachable": False, "models": [], "error": f"{type(e).__name__}"}


def check_models(judge_model: str | None = None, judge_host: str | None = None,
                 probe=None) -> list:
    """Runtime tiers, the independent judge, and the independence rule itself.

    Model names are resolved live from the application and the host — never
    asserted from a literal in this file, which would go stale the moment a tier
    is re-pointed.
    """
    from src.evals.final_eval import judge_backend as jb
    probe = probe or probe_ollama
    out = []

    try:
        from src.llm import local_client
        rc = local_client.runtime_config()
        runtime = {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL}
        host = rc.get("host") or ""
    except Exception as e:
        return [_c("runtime_models_resolvable", False,
                   f"could not read the runtime model configuration: {type(e).__name__}: {e}")]
    out.append(_c("runtime_models_resolvable", True,
                  f"main={runtime['main']} fast={runtime['fast']}", runtime))

    info = probe(host)
    out.append(_c("llm_host_reachable", info.get("reachable"),
                  man._hostonly(host) + ("" if info.get("reachable")
                                         else f" — {info.get('error')}"),
                  man._hostonly(host)))
    for role in ("main", "fast"):
        d = _installed(info.get("models"), runtime[role])
        out.append(_c(f"runtime_model_{role}", d["installed"],
                      f"{runtime[role]}" + (f" digest {d['digest']} "
                                            f"{d.get('parameter_size') or ''} "
                                            f"{d.get('quantization') or ''}".rstrip()
                                            if d["installed"] else " NOT INSTALLED"),
                      {**d, "model": runtime[role]}))

    jmodel = judge_model or jb.DEFAULT_JUDGE_MODEL
    jhost = judge_host or jb.DEFAULT_JUDGE_HOST
    try:
        jb.assert_independent(jmodel, runtime)
        out.append(_c("judge_independence", True,
                      f"{jmodel} is neither the main nor the fast runtime tier", jmodel))
    except jb.JudgeNotIndependent as e:
        out.append(_c("judge_independence", False, str(e)[:200], jmodel))

    jinfo = info if jhost.rstrip("/") == (host or "").rstrip("/") else probe(jhost)
    d = _installed(jinfo.get("models"), jmodel)
    out.append(_c("judge_model_installed", d["installed"],
                  f"{jmodel} on {man._hostonly(jhost)}"
                  + ("" if d["installed"] else
                     " NOT INSTALLED — the framework never falls back to a runtime model"),
                  {**d, "model": jmodel}))
    return out


def check_retrieval_assets() -> list:
    """Embedding and reranker availability, as the application resolves them."""
    out = []
    try:
        from src.retrieval import hybrid_retriever_v2 as H
        p = Path(H.DEFAULT_RERANKER_MODEL)
        out.append(_c("reranker_model", p.exists(),
                      f"{p.name} {'present' if p.exists() else 'MISSING'} at {p.parent}",
                      str(p)))
    except Exception as e:
        out.append(_c("reranker_model", False,
                      f"retriever configuration unreadable: {type(e).__name__}", None))
    try:
        sys.path.insert(0, str(man.ROOT / "scripts"))
        import fetch_models
        # Same resolution the application uses, without importing torch.
        model_dir = fetch_models._models_dir()
        wanted = [k for k in ("medcpt-query", "medcpt-article") if k in fetch_models.MODELS]
        present = [k for k in wanted if (model_dir / k).exists()]
        out.append(_c("embedding_models", bool(wanted) and len(present) == len(wanted),
                      (f"{len(present)}/{len(wanted)} present under {model_dir}"
                       + (f": {', '.join(present)}" if present else "")) if wanted
                      else "no embedding models configured",
                      {"dir": str(model_dir), "present": present, "wanted": wanted}))
    except Exception as e:
        out.append(_c("embedding_models", False,
                      f"model registry unreadable: {type(e).__name__}", None))
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        out.append(Check("gpu", PASS if cuda else WARN,
                         (torch.cuda.get_device_name(0) if cuda else
                          "no CUDA device — expected on a Mac, not on the Pod"),
                         cuda, required=True))
    except Exception as e:
        out.append(Check("gpu", WARN, f"torch unavailable ({type(e).__name__})",
                         None, required=True))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def check_output(results_root=None, run_id: str | None = None) -> list:
    root = Path(results_root or man.DEFAULT_RESULTS_ROOT)
    out = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".doctor_write_probe"
        probe.write_text("ok")
        probe.unlink()
        out.append(_c("results_root_writable", True, str(root), str(root)))
    except Exception as e:
        out.append(_c("results_root_writable", False,
                      f"{str(root)}: {type(e).__name__}: {e}", str(root)))
    if run_id:
        rd = man.RunDir(run_id, results_root)
        if rd.is_complete():
            out.append(_c("run_id_available", False,
                          f"run {run_id} already exists and is SEALED; completed runs are "
                          f"immutable. Choose another --run-id.", run_id))
        elif rd.exists() and any(rd.path.iterdir()):
            out.append(Check("run_id_available", WARN,
                             f"run {run_id} exists but is not sealed — it can only be "
                             f"continued with --resume", run_id, required=True))
        else:
            out.append(_c("run_id_available", True, f"{run_id} is free", run_id))
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def run_checks(*, profile: str = "final", judge_model: str | None = None,
               judge_host: str | None = None, results_root=None,
               run_id: str | None = None, db_probe=None, ollama_probe=None) -> Report:
    from src.evals.final_eval.judge import JUDGE_PROMPT_VERSION
    rep = Report(profile=profile)
    rep.add(*check_repository())
    rep.add(*check_python())
    rep.add(*check_versions(JUDGE_PROMPT_VERSION))
    rep.add(*check_dataset())
    rep.add(*check_data_plane(probe=db_probe))
    rep.add(*check_models(judge_model, judge_host, probe=ollama_probe))
    rep.add(*check_retrieval_assets())
    rep.add(*check_output(results_root, run_id))
    if profile == "local":
        for c in rep.checks:
            if c.name in ENVIRONMENT_CHECKS and c.status == FAIL:
                c.status, c.required = WARN, False
                c.detail += "  [advisory under --profile local]"
    return rep


_ICON = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN", SKIP: "SKIP"}


def render(rep: Report) -> str:
    L = [f"=== final_eval doctor — profile: {rep.profile} ===", ""]
    for c in rep.checks:
        req = "" if c.required else "  (advisory)"
        L.append(f"  [{_ICON[c.status]}] {c.name:<28} {c.detail}{req}")
    L.append("")
    if rep.blocking:
        L.append(f"NOT READY — {len(rep.blocking)} required check(s) failed:")
        L += [f"    - {c.name}: {c.detail}" for c in rep.blocking]
        L.append("")
        L.append("Nothing was changed. Fix the above and run doctor again.")
    else:
        warns = [c for c in rep.checks if c.status == WARN]
        L.append("READY" + (f" — {len(warns)} advisory warning(s)" if warns else ""))
    return "\n".join(L)
