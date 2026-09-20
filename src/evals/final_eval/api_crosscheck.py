"""
API contract cross-check
========================
The authoritative quality evaluation runs the production graph in-process,
because /ask does not expose what Phases 3 and 4 need: evidence TEXT for
groundedness, the per-claim verifier trace for routing, and the detail of a
synthesis failure that the API turns into a 5xx.

That leaves one thing unmeasured — whether the DEPLOYED HTTP surface behaves
like the graph the benchmark scored. This module checks exactly that, and
nothing else. It is a post-evaluation utility:

  * it is never a prerequisite for the judge or for any score;
  * it compares CONTRACT, not prose. Two runs of a temperature-0 LLM over a
    fresh thread can differ in wording, and requiring byte-identical text would
    produce a permanent red light that means nothing. What must match is the
    status semantics, the schema, the citation label set and the review
    routing;
  * it re-executes cases against a live server, so it is opt-in and is skipped
    outright when no API is reachable.

Answer consistency is reported on three axes a contract can actually hold:
identical citation labels, identical numeric anchors, and content-word overlap.
A divergence is a finding to read, not an automatic failure.
"""

from __future__ import annotations

import time
from collections import Counter

from src.evals.final_eval import EVALUATOR_VERSION
from src.evals.final_eval import normalize as N
from src.evals.final_eval.manifest import utc_now

REQUIRED_RESPONSE_KEYS = (
    "request_id", "thread_id", "data_plane", "status", "review_status", "answer",
    "answer_is_draft", "citations", "sources", "flagged_claims", "needs_human_review",
    "query_type", "temporal_mode", "node_trail", "models", "latency_ms", "timings",
)

# src/api/app.py turns a failed synthesis into a 5xx, so the graph status
# `failed` has no 200 response to compare against. Every other terminal status
# is returned as 200 with that status in the body.
EXPECTED_HTTP = {"completed": 200, "human_review_required": 200, "refused": 200}

OVERLAP_FLOOR = 0.6
# Jaccard over content words is only meaningful with enough of them. A one-line
# lab answer carries two or three identifying words once units, dates and
# stopwords are removed, so "creatinine was 1.4 mg/dL" and "the creatinine
# measured 1.4 mg/dL" score 0.5 while saying exactly the same thing. Below this
# many tokens the numeric anchors and the citation labels are the signal, and
# the overlap figure is reported without being treated as a finding.
MIN_TOKENS_FOR_OVERLAP = 8


def _labels(payload: dict) -> list:
    out = []
    for c in payload.get("citations") or []:
        for l in ([c.get("label")] if c.get("label") else []) + list(c.get("labels") or []):
            if l and l not in out:
                out.append(l)
    return sorted(out)


def _anchors(text: str) -> set:
    """The numeric content of an answer: quantities and dates. Two renderings
    of the same clinical answer agree here even when the prose differs."""
    return {str(q) for q in N.parse_quantities(text or "")} | set(N.parse_dates(text or ""))


def _overlap(a: str, b: str):
    """(jaccard, union size) over content words, or (None, 0) when there are none."""
    ta, tb = set(N.content_tokens(a or "")), set(N.content_tokens(b or ""))
    union = ta | tb
    if not union:
        return None, 0
    return round(len(ta & tb) / len(union), 3), len(union)


def compare_one(row: dict, http_status: int, payload: dict) -> dict:
    """One collected graph response against one /ask response."""
    graph_status = row.get("status")
    findings = []

    expected_http = EXPECTED_HTTP.get(graph_status)
    if expected_http is None:
        http_ok = http_status >= 500
        if not http_ok:
            findings.append(f"graph status {graph_status!r} should surface as 5xx, "
                            f"got HTTP {http_status}")
    else:
        http_ok = http_status == expected_http
        if not http_ok:
            findings.append(f"expected HTTP {expected_http} for {graph_status!r}, "
                            f"got {http_status}")

    api_status = payload.get("status")
    status_ok = (api_status == graph_status) if http_status == 200 else None
    if status_ok is False:
        findings.append(f"status {api_status!r} != graph status {graph_status!r}")

    missing = [k for k in REQUIRED_RESPONSE_KEYS if k not in payload] if http_status == 200 else []
    if missing:
        findings.append(f"response is missing {missing}")

    graph_labels, api_labels = _labels(row), _labels(payload)
    labels_ok = graph_labels == api_labels
    if not labels_ok and http_status == 200:
        findings.append(f"citation labels differ: graph {graph_labels} vs api {api_labels}")

    review_ok = None
    if http_status == 200 and "needs_human_review" in payload:
        review_ok = bool(payload["needs_human_review"]) == bool(row.get("needs_human_review"))
        if not review_ok:
            findings.append(f"needs_human_review {payload['needs_human_review']} != "
                            f"graph {row.get('needs_human_review')}")

    ga, aa = row.get("answer") or "", payload.get("answer") or ""
    anchors_g, anchors_a = _anchors(ga), _anchors(aa)
    overlap, n_tokens = _overlap(ga, aa)
    anchors_ok = anchors_g == anchors_a
    if http_status == 200 and not anchors_ok:
        findings.append(f"numeric anchors differ: only in graph {sorted(anchors_g - anchors_a)}, "
                        f"only in api {sorted(anchors_a - anchors_g)}")
    if http_status == 200 and overlap is not None and overlap < OVERLAP_FLOOR \
            and n_tokens >= MIN_TOKENS_FOR_OVERLAP:
        findings.append(f"content-word overlap {overlap} below {OVERLAP_FLOOR} "
                        f"over {n_tokens} content words")

    return {
        "query_id": row["query_id"],
        "http_status": http_status,
        "graph_status": graph_status,
        "api_status": api_status,
        "checks": {
            "http_semantics": http_ok,
            "status_matches": status_ok,
            "schema_complete": (not missing) if http_status == 200 else None,
            "citation_labels_match": labels_ok if http_status == 200 else None,
            "review_status_matches": review_ok,
            "numeric_anchors_match": anchors_ok if http_status == 200 else None,
        },
        "answer_consistency": {
            "content_word_overlap": overlap,
            "content_word_union": n_tokens,
            "overlap_was_assessed": n_tokens >= MIN_TOKENS_FOR_OVERLAP,
            "graph_anchors": sorted(anchors_g),
            "api_anchors": sorted(anchors_a),
            "note": "byte-identical text is NOT required and is never asserted",
        },
        "missing_response_keys": missing,
        "findings": findings,
        "contract_ok": not findings,
    }


def default_caller(base_url: str, timeout: int = 300):
    """A POST /ask caller. Injected in tests so the logic needs no server."""
    import requests
    base = (base_url or "").rstrip("/")

    def call(subject_id: int, query: str, request_id: str):
        r = requests.post(f"{base}/ask",
                          json={"subject_id": subject_id, "query": query},
                          headers={"X-Request-ID": request_id}, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {}
    return call


def probe(base_url: str, timeout: int = 10) -> dict:
    """Is the API reachable? A negative answer is a skip, never a failure."""
    try:
        import requests
        r = requests.get(f"{(base_url or '').rstrip('/')}/health", timeout=timeout)
        return {"reachable": r.status_code == 200, "http": r.status_code}
    except Exception as e:
        return {"reachable": False, "error": type(e).__name__}


def run(run_dir, base_url: str, caller=None, progress=None, limit: int | None = None) -> dict:
    """Cross-check every collected case against the deployed API."""
    say = progress or (lambda *_a, **_k: None)
    call = caller or default_caller(base_url)
    rows = sorted(run_dir.read_jsonl("responses"), key=lambda r: r["query_id"])
    rows = [r for r in rows if r.get("status") != "evaluator_error"]
    if limit:
        rows = rows[:limit]

    results, errors = [], []
    for row in rows:
        rid = f"xcheck-{run_dir.run_id}-{row['query_id']}"
        t0 = time.perf_counter()
        try:
            code, payload = call(row["subject_id"], row["query"], rid)
        except Exception as e:                     # a transport failure is a finding
            errors.append({"query_id": row["query_id"], "error": f"{type(e).__name__}: {e}"[:200]})
            say(f"  {row['query_id']:<10} TRANSPORT ERROR {type(e).__name__}")
            continue
        out = compare_one(row, code, payload or {})
        out["api_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(out)
        say(f"  {out['query_id']:<10} http={code} {'OK' if out['contract_ok'] else 'DIVERGENT'} "
            f"{out['findings'] or ''}")

    n = len(results)
    report = {
        "generated_at": utc_now(),
        "evaluator_version": EVALUATOR_VERSION,
        "run_id": run_dir.run_id,
        "base_url": base_url,
        "n_cases": n,
        "n_contract_ok": sum(1 for r in results if r["contract_ok"]),
        "n_divergent": sum(1 for r in results if not r["contract_ok"]),
        "transport_errors": errors,
        "finding_counts": dict(Counter(
            f.split(":")[0] for r in results for f in r["findings"]).most_common()),
        "cases": results,
        "scope": ("HTTP contract only. This is NOT a quality measurement and is not a "
                  "prerequisite for the judge or for any score in summary.json. "
                  "Generated text is compared on citation labels, numeric anchors and "
                  "content-word overlap; byte-identical prose is never required."),
    }
    run_dir.write_json("api_crosscheck", report)
    return report
