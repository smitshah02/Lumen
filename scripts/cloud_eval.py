"""
Cloud deployment evidence for the SYNTHETIC demo (runs ON the Pod)
==================================================================
Writes aggregate-only JSON — no note text, no answers, no secrets:

    results/cloud_run/deployment_manifest.json   environment, versions, corpus counts
    results/cloud_run/smoke_test.json            health/ready/retrieve/ask/human-review + GPU checks
    results/cloud_run/performance.json           1 warm-up (excluded) + golden-QA /ask timings
    results/cloud_run/tracing_smoke.json         optional: one /ask + tracing status (mode `tracing`)

    cd /workspace/lumen/repo && set -a && . /root/lumen-runtime/lumen.env && set +a
    /root/lumen-runtime/venv/bin/python scripts/cloud_eval.py all --out /workspace/lumen/results/cloud_run
"""

from __future__ import annotations

import os
import re
import math
import sys
import json
import time
import uuid
import argparse
import time
import hashlib
import platform
import statistics
from collections import Counter
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
API = "http://127.0.0.1:8000"
OLLAMA = os.environ.get("LUMEN_LLM_HOST", "http://127.0.0.1:11434")
LUMEN_ROOT = Path(os.environ.get("LUMEN_ROOT", "/workspace/lumen"))                    # persistent: results
RUNTIME_ROOT = Path(os.environ.get("LUMEN_RUNTIME_ROOT", "/root/lumen-runtime"))      # Pod-local: logs, pid
GOLDEN = json.loads((ROOT / "src/demo_data/golden_qa.json").read_text())
CREAT = {"subject_id": 90000001, "query": "What was the most recent creatinine?"}
BREATH = {"subject_id": 90000003, "query": "Why was the patient having trouble breathing?"}

# The verified A40 run BEFORE the latency work (qwen3:8b serving both tiers,
# LLM triage on every request, one MAIN verification call per citation).
# Kept here so `perf` prints the comparison rather than leaving it to memory.
# LEGACY_IDS is the original 15-question demo set; the corpus has since grown
# to 40 questions, so the legacy subset is summarised separately to keep an
# apples-to-apples before/after row.
A40_BASELINE = {
    "models": {"main": "qwen3:8b", "fast": "qwen3:8b"},
    "n": 15, "successful": 15, "mean_ms": 15240, "p50_ms": 15710, "p95_ms": 19380,
    "mean_retrieval_ms": 503, "mean_llm_ms": 14600, "mean_llm_calls": 3.6,
    "ollama_decode_tokens_per_s": 39,
    "note": "measured on RunPod A40 before main/fast split, deterministic triage, "
            "deterministic + batched verification",
}
LEGACY_IDS = [f"demo_q{i:02d}" for i in range(1, 16)]

# Earlier measurements supplied by the user (single warm requests, not re-measured here).
REFERENCE = {
    "native_mac": {"total_ms": 30952, "retrieval_ms": 2935, "llm_ms": 19589, "llm_calls": 3,
                   "runtime": "macOS, Apple Silicon (MPS retrieval, Metal Ollama)"},
    "docker_desktop": {"total_ms": 45728, "retrieval_ms": 15985, "llm_ms": 17955, "llm_calls": 3,
                       "runtime": "Docker Desktop VM, CPU retrieval, host Metal Ollama; first request after restart"},
}


def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _tokens(text: str) -> set:
    return {t.strip(".,;:()") for t in re.sub(r"[\[\]]", " ", (text or "").lower()).split()}


def _ask(payload: dict, rid: str) -> tuple[int, dict, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{API}/ask", json=payload, headers={"X-Request-ID": rid}, timeout=900)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {}), \
        round((time.perf_counter() - t0) * 1000, 1)


def _write(out: Path, name: str, obj: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {out / name}")


# ---------------------------------------------------------------------------
_SHA_RE = re.compile(r"[0-9a-f]{40}")


def read_provenance(path: Path) -> dict:
    """Source provenance written by scripts/sync_to_pod.sh from the LOCAL git repo.
    The Pod has no .git, so nothing is inferred remotely: anything missing or
    malformed is reported as null / "unknown", never guessed."""
    out = {"git_sha": None, "branch": None, "dirty_worktree": "unknown", "generated_at": None,
           "provenance_source": "missing"}
    try:
        d = json.loads(path.read_text())
    except FileNotFoundError:
        return out
    except Exception:
        return {**out, "provenance_source": "unreadable"}
    sha, branch, dirty = d.get("git_sha"), d.get("branch"), d.get("dirty_worktree")
    return {"git_sha": sha if isinstance(sha, str) and _SHA_RE.fullmatch(sha) else None,
            "branch": branch if isinstance(branch, str) and branch else None,
            "dirty_worktree": dirty if isinstance(dirty, bool) else "unknown",
            "generated_at": d.get("generated_at") if isinstance(d.get("generated_at"), str) else None,
            "provenance_source": "sync_metadata"}


def manifest(out: Path) -> None:
    import torch
    from sqlalchemy import text
    from src import storage
    from src.llm import local_client
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_models import MODELS

    prov = read_provenance(ROOT / ".deployment_source.json")
    gpu = [g.strip() for g in _sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader").split(",")]
    mem_kb = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")), 0)
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=10).json().get("models", [])
    def _tag(name):
        m = next((m for m in tags if m["name"] == name), {})
        return {"tag": name, "digest": (m.get("digest") or "")[:12],
                "parameter_size": m.get("details", {}).get("parameter_size"),
                "quantization": m.get("details", {}).get("quantization_level"),
                "installed": bool(m)}
    with storage.engine.connect() as c:
        q = lambda s: c.execute(text(s)).scalar()
        counts = {"patients": q("SELECT COUNT(*) FROM patients"), "notes": q("SELECT COUNT(*) FROM clinical_notes"),
                  "chunks": q("SELECT COUNT(*) FROM note_chunks"), "chunks_embedded": q("SELECT COUNT(embedding) FROM note_chunks"),
                  "non_synthetic_subjects": q("SELECT COUNT(*) FROM patients WHERE subject_id NOT BETWEEN 90000000 AND 90999999")}
    _write(out, "deployment_manifest.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": prov["git_sha"], "branch": prov["branch"], "dirty_worktree": prov["dirty_worktree"],
        "source_generated_at": prov["generated_at"], "provenance_source": prov["provenance_source"],
        "platform": f"{platform.system()}/{platform.machine()}",
        "gpu": {"name": gpu[0] if gpu else None, "vram": gpu[1] if len(gpu) > 1 else None,
                "driver": gpu[2] if len(gpu) > 2 else None},
        "cpu_count": os.cpu_count(), "ram_gb": round(mem_kb / 1048576, 1),
        "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "ollama_version": requests.get(f"{OLLAMA}/api/version", timeout=10).json().get("version"),
        "llm": {"main": _tag(local_client.MAIN_MODEL), "fast": _tag(local_client.FAST_MODEL),
                "roles": local_client.runtime_config()["roles"],
                "keep_alive": local_client.KEEP_ALIVE,
                "deterministic_labs": os.environ.get("LUMEN_DETERMINISTIC_LABS", "1")},
        "embedding_model": {k: f"{MODELS[k][0]}@{MODELS[k][1][:10]}" for k in ("medcpt-query", "medcpt-article")},
        "reranker_model": f"{MODELS['bge-reranker'][0]}@{MODELS['bge-reranker'][1][:10]}",
        "data_plane": storage.DATA_PLANE, "database": storage.engine.url.database,
        "synthetic_counts": counts,
        "api_bind": os.environ.get("LUMEN_API_BIND", "127.0.0.1"),
        "tracing": _tracing_view(),
    })


def _tracing_view() -> dict:
    """The API process's own tracing status (/ready), else this process's config. No keys."""
    try:
        tr = requests.get(f"{API}/ready", timeout=30).json().get("tracing") or {}
    except Exception:
        from src.obs import tracing
        tr = tracing.status()
    return {"tracing_enabled": bool(tr.get("enabled")), "tracing_provider": "langfuse",
            "tracing_host": tr.get("host"), "tracing_state": tr.get("state")}


# ---------------------------------------------------------------------------
def _gpu_evidence() -> dict:
    ps = requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
    from src.llm import local_client
    # Both tiers may be resident at once now; report the worst offloading ratio,
    # because one tier spilling to CPU is what a latency regression looks like.
    wanted = {local_client.MAIN_MODEL, local_client.FAST_MODEL}
    loaded = [m for m in ps if m.get("name") in wanted] or [m for m in ps if m.get("size")]
    frac = min(((m.get("size_vram", 0) / m["size"]) for m in loaded if m.get("size")), default=0.0)
    resident = [m.get("name") for m in loaded]
    log = RUNTIME_ROOT / "logs" / "api.log"
    reranker_cuda = log.exists() and any("Reranker loaded" in l and "on cuda" in l for l in log.read_text().splitlines())
    apps = _sh("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader")
    api_pid = (RUNTIME_ROOT / "api.pid").read_text().strip() if (RUNTIME_ROOT / "api.pid").exists() else ""
    api_on_gpu = any(l.split(",")[0].strip() == api_pid for l in apps.splitlines()) if api_pid else False
    return {
        "ollama": {"model_loaded": bool(resident), "resident": resident, "expected": sorted(wanted),
                   "vram_fraction": round(frac, 3),
                   "result": "PASS" if resident and frac >= 0.99 else "FAIL"},
        "pytorch": {"reranker_logged_on_cuda": reranker_cuda, "api_pid_in_nvidia_smi": api_on_gpu,
                    "result": "PASS" if (reranker_cuda or api_on_gpu) else "FAIL"},
        "gpu_memory_used": _sh("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"),
    }


# Controlled post-verification states. Synthetic, fixed text; no LLM involved.
_REVIEW_EVIDENCE = [{"label": "S1", "chunk_id": -1, "source_type": "note", "note_type": "discharge",
                     "charttime": None, "score": 0.0, "text": "SYNTHETIC controlled evidence for routing check."}]
_REVIEW_CASES = {
    # name: (state after `verification`, expected route, expected pause, expected review_status)
    "unsupported_claim": ({"needs_human_review": True, "review_status": "pending",
                           "verification": {"checked": 1, "unsupported": 1, "synthesis_failed": False},
                           "citations": [{"claim": "Claim [S1].", "label": "S1", "chunk_id": -1, "verified": False,
                                          "verification_note": "unsupported: controlled routing check"}]},
                          "human_review", True, "pending"),
    "all_verified": ({"needs_human_review": False, "review_status": "auto_approved",
                      "verification": {"checked": 1, "unsupported": 0, "synthesis_failed": False},
                      "citations": [{"claim": "Claim [S1].", "label": "S1", "chunk_id": -1, "verified": True,
                                     "verification_note": "supported: controlled routing check"}]},
                     "finalize", False, "auto_approved"),
    "synthesis_failed": ({"needs_human_review": True, "review_status": "failed", "draft_answer": "",
                          "verification": {"checked": 0, "unsupported": 0, "synthesis_failed": True},
                          "citations": []},
                         "human_review", False, "failed"),
}


def review_routing_check(graph) -> dict:
    """Deterministic human-review routing check on the REAL compiled graph.

    For each case, a fresh thread is seeded with a controlled state as if the
    production `verification` node had just produced it (update_state(...,
    as_node="verification")), then resumed with invoke(None). Execution goes
    through the production conditional edge (route_after_verification) and the
    real human_review/finalize nodes — neither calls an LLM. PASS requires, per
    case, the expected route, pause (interrupt) and review_status. A pause is
    what the API reports as status "human_review_required"."""
    from src.agents.graph import route_after_verification
    cases, ok_all = {}, True
    for name, (seed, want_route, want_pause, want_status) in _REVIEW_CASES.items():
        tid = f"smoke-routing-{name}-{uuid.uuid4().hex[:8]}"
        cfg = {"configurable": {"thread_id": tid}}
        state = {"query": "controlled routing check", "subject_id": 90000001, "thread_id": tid,
                 "draft_answer": "Claim [S1].", "patient_evidence": _REVIEW_EVIDENCE, **seed}
        route = route_after_verification(state)
        graph.update_state(cfg, state, as_node="verification")
        next_node = list(graph.get_state(cfg).next)
        out = graph.invoke(None, cfg)
        after = graph.get_state(cfg).values
        paused = "__interrupt__" in out
        got = {"route": route, "next_node": next_node[0] if next_node else None, "paused": paused,
               "review_status": after.get("review_status"),
               "flagged": len(out["__interrupt__"][0].value.get("flagged", [])) if paused else 0,
               "api_status_equivalent": "human_review_required" if paused else
                                        ("failed" if after.get("review_status") == "failed" else "completed")}
        ok = (route == want_route and got["next_node"] == want_route and paused == want_pause
              and got["review_status"] == want_status)
        ok_all &= ok
        cases[name] = {**got, "expected": {"route": want_route, "paused": want_pause, "review_status": want_status},
                       "result": "PASS" if ok else "FAIL"}
    return {"method": "controlled post-verification states resumed through the production graph (no LLM)",
            "cases": cases, "result": "PASS" if ok_all else "FAIL"}


def smoke(out: Path) -> None:
    res = {}
    r = requests.get(f"{API}/health", timeout=10)
    res["health"] = {"http": r.status_code, "result": "PASS" if r.status_code == 200 and r.json() == {"status": "ok"} else "FAIL"}
    r = requests.get(f"{API}/ready", timeout=30)
    rd = r.json()
    ok = (r.status_code == 200 and rd.get("data_plane") == "demo" and rd.get("database") == "lumen_demo"
          and all(v == "ok" for v in rd.get("dependencies", {}).values()))
    res["ready"] = {"http": r.status_code, "data_plane": rd.get("data_plane"), "database": rd.get("database"),
                    "dependencies": rd.get("dependencies"), "result": "PASS" if ok else "FAIL"}
    r = requests.post(f"{API}/retrieve", json={**CREAT, "top_k": 5}, timeout=600)
    rr = r.json()
    hits = [x["rank"] for x in rr.get("results", []) if "creatinine 1.4 mg/dl" in " ".join(x["text"].lower().split())]
    res["retrieve"] = {"http": r.status_code, "first_rank_with_1.4_mg_dL": hits[0] if hits else None,
                       "latency_ms": rr.get("latency_ms"), "result": "PASS" if r.status_code == 200 and hits else "FAIL"}
    code, a, _ = _ask(CREAT, "cloud-smoke-creat")
    labels = {s["label"] for s in a.get("sources", [])}
    cites_ok = any(c["label"] in labels and c["verified"] for c in a.get("citations", [])) and \
        all(c["label"] in labels for c in a.get("citations", []) if c["label"])
    fact = {"creatinine", "1.4", "mg/dl"} <= _tokens(a.get("answer", ""))
    res["ask"] = {"http": code, "status": a.get("status"), "fact_1.4_mg_dL": fact, "valid_citation": cites_ok,
                  "timings": a.get("timings"), "result": "PASS" if code == 200 and a.get("status") == "completed" and fact and cites_ok else "FAIL"}
    from src.agents.graph import build_graph, close_pools
    graph, _ = build_graph()                  # the production graph + its Postgres checkpointer
    try:
        res["human_review_routing"] = review_routing_check(graph)
    finally:
        close_pools()
    # Informational only: whether a live LLM answer gets flagged is stochastic.
    code, b, _ = _ask(BREATH, "cloud-smoke-review")
    res["human_review_live_observation"] = {
        "http": code, "status": b.get("status"), "flagged_claims": b.get("flagged_claims"),
        "observed_human_review": b.get("status") == "human_review_required",
        "note": "stochastic LLM output; informational, not a pass/fail criterion"}
    res["gpu"] = _gpu_evidence()
    _write(out, "smoke_test.json", {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **res})


# ---------------------------------------------------------------------------
def _pct(xs: list, p: float):
    """Nearest-rank percentile."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(1, math.ceil(p * len(s))) - 1]


def _main_model() -> str:
    """The MAIN tag as the application resolves it. Never a literal default here:
    a stale fallback would silently benchmark a different model than /ask uses."""
    from src.llm import local_client
    return local_client.MAIN_MODEL


def _ollama_speed(runs: int = 3) -> dict:
    """Decode/prefill speed from Ollama's own counters, on a fixed non-clinical prompt.
    Measured on MAIN, which is the tier the latency work put the most weight on."""
    rates, prefill = [], []
    for _ in range(runs):
        b = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
            "model": _main_model(), "stream": False, "think": False,
            "prompt": "List the integers from 1 to 80, separated by commas.",
            "options": {"temperature": 0, "num_predict": 200}}).json()
        if b.get("eval_duration"):
            rates.append(b["eval_count"] / (b["eval_duration"] / 1e9))
        if b.get("prompt_eval_duration"):
            prefill.append(b["prompt_eval_count"] / (b["prompt_eval_duration"] / 1e9))
    return {"decode_tokens_per_s": round(statistics.mean(rates), 1) if rates else None,
            "prefill_tokens_per_s": round(statistics.mean(prefill), 1) if prefill else None,
            "runs": runs, "prompt": "fixed non-clinical counting prompt"}


def _summarise(rows: list[dict]) -> dict:
    """One latency + call-budget summary over a set of /ask results."""
    ok = [r for r in rows if r["http"] == 200 and r["status"] in ("completed", "human_review_required", "refused")]
    tot = [r["total_ms"] for r in ok if r["total_ms"] is not None]
    mean = lambda k: (round(statistics.mean([r[k] for r in ok if r.get(k) is not None]), 2)
                      if any(r.get(k) is not None for r in ok) else None)
    total = lambda k: sum(r.get(k) or 0 for r in ok)
    return {
        "n": len(rows), "successful": len(ok), "failed": len(rows) - len(ok),
        "success_rate": round(len(ok) / len(rows), 3) if rows else None,
        "mean_ms": round(statistics.mean(tot), 1) if tot else None,
        "p50_ms": _pct(tot, 0.50), "p95_ms": _pct(tot, 0.95),
        "mean_retrieval_ms": mean("retrieval_ms"), "mean_llm_ms": mean("llm_ms"),
        # Call budget — the thing this optimization is actually trying to move.
        "mean_llm_calls": mean("llm_calls"),
        "mean_main_calls": mean("llm_main_calls"), "mean_fast_calls": mean("llm_fast_calls"),
        "total_main_calls": total("llm_main_calls"), "total_fast_calls": total("llm_fast_calls"),
        "requests_with_zero_llm_calls": sum(1 for r in ok if (r.get("llm_calls") or 0) == 0),
        "deterministic_answers": total("deterministic_answer"),
        "claims_verified_deterministically": total("deterministic_verified"),
        "claims_verified_by_model": total("llm_verified"),
        "query_class": {c: sum(1 for r in ok if r.get("query_complexity") == c) for c in ("simple", "complex")},
        "human_review_required": sum(1 for r in ok if r["status"] == "human_review_required"),
        "errors": sum(1 for r in rows if r["http"] != 200),
    }


def perf(out: Path) -> None:
    code, _, warm_ms = _ask({"subject_id": GOLDEN[0]["subject_id"], "query": GOLDEN[0]["query"]}, "cloud-perf-warmup")
    rows = []
    for g in GOLDEN:
        code, a, client_ms = _ask({"subject_id": g["subject_id"], "query": g["query"]}, f"cloud-perf-{g['id']}")
        t = a.get("timings") or {}
        rows.append({"query_id": g["id"], "http": code, "status": a.get("status"),
                     "total_ms": t.get("total_ms"), "client_ms": client_ms, "retrieval_ms": t.get("retrieval_ms"),
                     "llm_ms": t.get("llm_ms"), "llm_calls": t.get("llm_calls"),
                     "llm_main_calls": t.get("llm_main_calls"), "llm_fast_calls": t.get("llm_fast_calls"),
                     "llm_main_ms": t.get("llm_main_ms"), "llm_fast_ms": t.get("llm_fast_ms"),
                     "deterministic_answer": t.get("deterministic_answer"),
                     "deterministic_verified": t.get("deterministic_verified"),
                     "llm_verified": t.get("llm_verified"),
                     "query_complexity": t.get("query_complexity"), "classified_by": t.get("classified_by"),
                     "node_trail": a.get("node_trail"),
                     "needs_human_review": a.get("needs_human_review")})
        print(f"  {g['id']} {code} {a.get('status')} {t.get('total_ms')} ms "
              f"calls={t.get('llm_calls')} (main={t.get('llm_main_calls')} fast={t.get('llm_fast_calls')})", flush=True)

    from src.llm import local_client
    legacy = [r for r in rows if r["query_id"] in LEGACY_IDS]
    _write(out, "performance.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "sequential /ask over src/demo_data/golden_qa.json after 1 excluded warm-up; server-side timings; "
                  "p50/p95 by nearest rank",
        "models": {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL,
                   "roles": local_client.runtime_config()["roles"]},
        "warmup": {"http": code, "client_ms": warm_ms, "excluded": True},
        "requests": rows,
        "summary": _summarise(rows),
        # demo_q01-demo_q15 are the questions the pre-optimization A40 run used.
        # Compare THIS row against a40_baseline_before; the full set is larger now.
        "legacy_subset": {"ids": LEGACY_IDS, "summary": _summarise(legacy)},
        "a40_baseline_before": A40_BASELINE,
        "ollama_speed": _ollama_speed(),
        "reference_local_single_requests": REFERENCE,
        "comparability": (f"different hardware and runtimes; the cloud row is a {len(rows)}-request mean, "
                          f"the local rows are single requests. Compare legacy_subset.summary against "
                          f"a40_baseline_before — the full golden set grew from 15 to {len(GOLDEN)} questions."),
    })


def tracing_smoke(out: Path) -> None:
    """One synthetic /ask with tracing status. PASS/FAIL reflects the application only;
    an unreachable or unverifiable observability backend is reported, never fatal."""
    rid = f"cloud-trace-{uuid.uuid4().hex[:8]}"
    code, a, client_ms = _ask(CREAT, rid)
    app_ok = code == 200 and a.get("status") in ("completed", "human_review_required")
    view = _tracing_view()                     # after the /ask, so the API has tried to connect
    backend = {"checked": False, "trace_found": None}
    if view["tracing_state"] == "active" and os.environ.get("LANGFUSE_PUBLIC_KEY") and a.get("thread_id"):
        backend["checked"] = True
        try:
            from langfuse import get_client
            lf = get_client()
            for _ in range(10):                # the API exports spans in the background
                time.sleep(3)
                if lf.api.trace.list(session_id=a["thread_id"], limit=5).data:
                    backend["trace_found"] = True
                    break
            else:
                backend["trace_found"] = False
        except Exception as e:
            backend["error_type"] = type(e).__name__
    tracing_result = ("disabled" if not view["tracing_enabled"] else
                      "verified" if backend.get("trace_found") else
                      view["tracing_state"] if view["tracing_state"] != "active" else "enabled_unverified")
    _write(out, "tracing_smoke.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": rid, "thread_id": a.get("thread_id"), "http": code, "status": a.get("status"),
        "client_ms": client_ms, "app_result": "PASS" if app_ok else "FAIL",
        "tracing": view, "backend": backend, "tracing_result": tracing_result,
    })


# ---------------------------------------------------------------------------
# Human-review audit: WHY each legacy query escalated
# ---------------------------------------------------------------------------
CLAIM_PREVIEW_CHARS = 100


def _claim_id(text: str) -> str:
    """Stable short id for a claim, so two runs can be diffed without storing
    the sentence twice."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


def _escalation_reason(st: dict, trace: list[dict]) -> str:
    """One label per query, for the grouped summary."""
    v = st.get("verification") or {}
    if v.get("synthesis_failed"):
        return "synthesis_failed"
    bad = [t for t in trace if t.get("final") == "unsupported"]
    if not bad:
        return "none"
    reasons = {t.get("reason", "") for t in bad}
    stages = {t.get("stage") for t in bad}
    if any(r == "no valid citation" for r in reasons):
        return "uncited_claim"
    if any(r == "no verdict returned" for r in reasons):
        return "missing_batch_verdict"
    if stages == {"fast_model"}:
        verdicts = {t.get("verdict") for t in bad}
        if verdicts == {"partial"}:
            return "fast_partial"
        return "fast_unsupported"
    return "other"


def audit(out: Path, ids: list[str] | None = None) -> None:
    """Run the legacy golden subset through the PRODUCTION graph in-process and
    record, per claim, how each verdict was reached and why a run escalated.

    In-process rather than over /ask because the per-claim verifier trace is
    graph state, not part of the API contract — this adds no endpoint and
    changes no response shape.

    Privacy: no note text, no retrieved chunks, no prompts, no secrets. Claims
    are generated sentences over synthetic demo data; they are stored truncated
    and alongside a hash so runs can be compared without the full text.
    """
    import uuid
    from src.agents.graph import build_graph, close_pools
    from src.llm import local_client

    wanted = ids or LEGACY_IDS
    golden = {g["id"]: g for g in GOLDEN}
    missing = [i for i in wanted if i not in golden]
    if missing:
        print(f"warning: not in the golden set, skipping: {missing}", file=sys.stderr)
    rows, graph = [], build_graph()[0]
    try:
        for qid in [i for i in wanted if i in golden]:
            g = golden[qid]
            tid = f"audit-{qid}-{uuid.uuid4().hex[:6]}"
            t0 = time.perf_counter()
            res = graph.invoke({"query": g["query"], "subject_id": g["subject_id"], "thread_id": tid},
                               config={"configurable": {"thread_id": tid}})
            st = graph.get_state({"configurable": {"thread_id": tid}}).values
            ms = round((time.perf_counter() - t0) * 1000, 1)

            v = st.get("verification") or {}
            trace = v.get("claims") or []
            cites = st.get("citations") or []
            interrupted = "__interrupt__" in res
            by_i = {t["i"]: t for t in trace}

            claims = []
            for i, c in enumerate(cites):
                t = by_i.get(i, {})
                claims.append({
                    "index": i,
                    "claim_id": _claim_id(c.get("claim", "")),
                    "claim_preview": (c.get("claim", "") or "")[:CLAIM_PREVIEW_CHARS],
                    "citation_labels": t.get("labels", c.get("labels") or
                                             ([c["label"]] if c.get("label") else [])),
                    "stage": t.get("stage"),
                    "deterministic_verdict": ("supported" if t.get("stage") == "deterministic"
                                              and t.get("final") == "supported" else "unresolved"),
                    "deterministic_reason": t.get("det_reason") or (
                        t.get("reason") if t.get("stage") == "deterministic" else None),
                    "sent_to_fast_verifier": t.get("stage") == "fast_model",
                    "fast_verdict": t.get("verdict") if t.get("stage") == "fast_model" else None,
                    "fast_verdict_parsed": (t.get("stage") == "fast_model"
                                            and t.get("reason") != "no verdict returned"),
                    "fast_reason": t.get("reason") if t.get("stage") == "fast_model" else None,
                    "final": t.get("final", "unsupported" if not c.get("verified") else "supported"),
                })

            rows.append({
                "query_id": qid,
                "status": ("human_review_required" if interrupted
                           else "refused" if st.get("query_type") == "unsupported" else "completed"),
                "review_status": st.get("review_status"),
                "needs_human_review": bool(interrupted or st.get("needs_human_review")),
                "escalation_reason": ("none" if not (interrupted or st.get("needs_human_review"))
                                      else _escalation_reason(st, trace)),
                "query_complexity": st.get("query_complexity"),
                "classified_by": st.get("classified_by"),
                "query_type": st.get("query_type"),
                "synthesis_model": (local_client.role_spec(v["synthesis_role"])["model"]
                                    if v.get("synthesis_role") else None),
                "synthesis_role": v.get("synthesis_role"),
                "node_trail": st.get("node_trail") or [],
                "n_claims": len(cites),
                "n_flagged_claims": sum(1 for c in cites if not c.get("verified")),
                "deterministic_supported": v.get("deterministic"),
                "sent_to_fast_verifier": v.get("llm_checked"),
                "unsupported": v.get("unsupported"),
                "declined_answer": bool(v.get("refusal")),
                "synthesis_failed": bool(v.get("synthesis_failed")),
                "total_ms": ms,
                "claims": claims,
            })
            print(f"  {qid} {rows[-1]['review_status']:<16} claims={len(cites)} "
                  f"flagged={rows[-1]['n_flagged_claims']} reason={rows[-1]['escalation_reason']}", flush=True)
    finally:
        close_pools()

    escalated = [r for r in rows if r["needs_human_review"]]
    _write(out, "human_review_audit.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subset": wanted,
        "method": "production graph in-process; per-claim verifier trace from graph state. "
                  "No note text, chunks, prompts or secrets are recorded.",
        "models": {"main": local_client.MAIN_MODEL, "fast": local_client.FAST_MODEL},
        "prompt_versions": {"synthesis": _prompt_versions()[0], "verify_batch": _prompt_versions()[1]},
        "summary": {
            "queries": len(rows),
            "human_review": len(escalated),
            "by_escalation_reason": dict(Counter(r["escalation_reason"] for r in rows)),
            "claims_total": sum(r["n_claims"] for r in rows),
            "claims_supported_deterministically": sum(r["deterministic_supported"] or 0 for r in rows),
            "claims_sent_to_fast_verifier": sum(r["sent_to_fast_verifier"] or 0 for r in rows),
            "claims_unsupported": sum(r["unsupported"] or 0 for r in rows),
            "declined_answers": sum(1 for r in rows if r["declined_answer"]),
            "synthesis_failures": sum(1 for r in rows if r["synthesis_failed"]),
            "by_final_claim_state": dict(Counter(c["final"] for r in rows for c in r["claims"])),
            "by_fast_verdict": dict(Counter(c["fast_verdict"] for r in rows for c in r["claims"]
                                            if c["fast_verdict"])),
            "by_synthesis_role": dict(Counter(r["synthesis_role"] for r in rows if r["synthesis_role"])),
        },
        "queries": rows,
    })


def _prompt_versions() -> tuple[str, str]:
    from src.agents import prompts
    return prompts.SYNTHESIS_VERSION, prompts.VERIFY_BATCH_VERSION


def main() -> int:
    if os.environ.get("LUMEN_DATA_PLANE") != "demo":
        print("refusing: LUMEN_DATA_PLANE must be demo", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "smoke", "perf", "all", "tracing", "audit"])
    ap.add_argument("--out", default=str(LUMEN_ROOT / "results" / "cloud_run"))
    ap.add_argument("--ids", nargs="+", default=None,
                    help="audit only: query ids to run (default: the legacy demo_q01-demo_q15 subset)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.mode in ("manifest", "all"):
        manifest(out)
    if a.mode in ("smoke", "all"):
        smoke(out)
    if a.mode in ("perf", "all"):
        perf(out)
    if a.mode == "tracing":
        tracing_smoke(out)
    if a.mode == "audit":
        audit(out, a.ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
