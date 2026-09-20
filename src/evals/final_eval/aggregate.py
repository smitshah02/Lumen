"""
Phase 6 — combined scorecard
============================
Four independent sections, deliberately NOT reduced to one weighted number:

  reliability          hard gates. Any failure fails the run.
  deterministic_quality  what ground truth can settle in code.
  judge_quality        the independent judge, per dimension, with its failures
                       counted rather than absorbed.
  operational          latency, routing and call budget.

Denominator discipline
----------------------
Every rate carries its own numerator and denominator. Failed and errored cases
stay in the denominator. A rate whose denominator is zero is reported as null,
never as 0% or 100%, and small-n rates are printed as k/n so nobody reads
"100%" off a single case.
"""

from __future__ import annotations

import statistics
from collections import Counter

from src.evals.final_eval import EVALUATOR_VERSION


def _rate(num: int, den: int):
    """A rate with an honest empty case: no denominator means no number."""
    return {"n": num, "d": den, "rate": (round(num / den, 4) if den else None)}


def _pct(xs: list, p: float):
    if not xs:
        return None
    s = sorted(xs)
    import math
    return s[max(1, math.ceil(p * len(s))) - 1]


# ---------------------------------------------------------------------------
def deterministic_section(dets: list) -> dict:
    n = len(dets)
    completed = sum(1 for d in dets if d["execution"]["completed"])
    evaluator_errors = [d["query_id"] for d in dets if d["execution"]["evaluator_error"]]
    synthesis_failures = [d["query_id"] for d in dets if d["execution"]["synthesis_failed"]]
    invalid_schema = [d["query_id"] for d in dets if not d["execution"]["valid_schema"]]

    # --- facts -----------------------------------------------------------
    total_facts = sum(d["facts"]["n_expected"] for d in dets)
    matched = sum(d["facts"]["matched"] for d in dets)
    contradictions = sum(d["facts"]["n_contradictions"] for d in dets)
    unmatched = sum(d["facts"]["n_strict_unmatched"] for d in dets)
    struct = [r for d in dets for r in d["facts"]["results"] if r["kind"] == "structured"]
    text = [r for d in dets for r in d["facts"]["results"] if r["kind"] == "text_only"]

    # --- citations -------------------------------------------------------
    factual_claims = sum(d["citations"]["n_factual_claims"] for d in dets)
    cited_claims = sum(d["citations"]["n_cited_factual_claims"] for d in dets)
    claims_with_labels = sum(
        1 for d in dets for _ in range(d["citations"]["n_claims"])) if dets else 0
    bad_pre = sum(len(d["citations"]["hallucinated_labels_pre_strip"]) for d in dets)
    bad_post = sum(len(d["citations"]["hallucinated_labels_post_strip"]) for d in dets)
    cases_with_bad = [d["query_id"] for d in dets if d["citations"]["hallucinated_labels_pre_strip"]]

    # --- temporal / abstention / ambiguity --------------------------------
    temporal = [d for d in dets if d["temporal"]["applicable"]]
    abst = [d for d in dets if d["abstention"]["applicable"]]
    ambi = [d for d in dets if d["ambiguity"]["applicable"]]
    adm = [d for d in dets if d["admission_scope"]["applicable"]]

    # --- routing (non-circular; see deterministic._review_worthy) ---------
    review_worthy = [d for d in dets if d["review_worthy"]["is_review_worthy"]]
    routed = [d for d in review_worthy if d["routing_observed"]["needs_human_review"]]
    auto = [d for d in dets if not d["routing_observed"]["needs_human_review"]
            and not d["execution"]["evaluator_error"]]
    auto_clean = [d for d in auto if not d["review_worthy"]["is_review_worthy"]]
    needs_review = [d for d in dets if d["routing_observed"]["needs_human_review"]]

    return {
        "n_cases": n,
        "completion_rate": _rate(completed, n),
        "evaluator_error_rate": _rate(len(evaluator_errors), n),
        "evaluator_error_ids": evaluator_errors,
        "synthesis_failure_rate": _rate(len(synthesis_failures), n),
        "synthesis_failure_ids": synthesis_failures,
        "invalid_response_rate": _rate(len(invalid_schema), n),
        "case_pass_rate": _rate(sum(1 for d in dets if d["case_pass"]), n),
        "case_pass_definition": [
            "executed with a non-empty answer",
            "matched_expected_facts >= min_facts",
            "no deterministic factual contradiction",
            "no must_not_contain violation",
            "no cross-patient leakage",
            "applicable temporal constraint passes",
            "applicable unsupported/ambiguity behaviour passes",
        ],
        "facts": {
            "fact_recall": _rate(matched, total_facts),
            "contradictions": contradictions,
            "strict_unmatched": unmatched,
            "structured": {"total": len(struct),
                           "strict_match": sum(1 for r in struct if r["outcome"] == "strict_match"),
                           "contradiction": sum(1 for r in struct if r["outcome"] == "contradiction"),
                           "strict_unmatched": sum(1 for r in struct if r["outcome"] == "strict_unmatched")},
            "text_only": {"total": len(text),
                          "strict_match": sum(1 for r in text if r["outcome"] == "strict_match"),
                          "strict_unmatched": sum(1 for r in text if r["outcome"] == "strict_unmatched")},
            "note": ("strict_unmatched means deterministic matching could not confirm the "
                     "fact. It is NOT a factual error; only `contradiction` is. Semantic "
                     "paraphrase is resolved by the independent judge."),
        },
        "citations": {
            "citation_validity_rate": _rate(max(0, factual_claims - bad_pre), factual_claims)
            if factual_claims else _rate(0, 0),
            "hallucinated_labels_pre_strip": bad_pre,
            "hallucinated_labels_post_strip": bad_post,
            "cases_with_hallucinated_labels": cases_with_bad,
            "citation_coverage": _rate(cited_claims, factual_claims),
            "uncited_factual_claims": factual_claims - cited_claims,
            "note": ("src/agents/graph.py:synthesis strips hallucinated labels out of the "
                     "stored answer, so post_strip is expected to be 0. pre_strip is the "
                     "honest measure of model behaviour."),
        },
        "temporal_accuracy": _rate(sum(1 for d in temporal if d["temporal"]["pass"]), len(temporal)),
        "temporal_failures": [d["query_id"] for d in temporal if d["temporal"]["pass"] is False],
        "unsupported_abstention_accuracy": _rate(
            sum(1 for d in abst if d["abstention"]["pass"]), len(abst)),
        "unsupported_abstention_ids": [d["query_id"] for d in abst],
        "ambiguity_handling_accuracy": _rate(
            sum(1 for d in ambi if d["ambiguity"]["pass"]), len(ambi)),
        "ambiguity_handling_ids": [d["query_id"] for d in ambi],
        "admission_scope_accuracy": _rate(
            sum(1 for d in adm if d["admission_scope"]["pass"]), len(adm)),
        "routing": {
            "review_recall": _rate(len(routed), len(review_worthy)),
            "review_worthy_ids": [d["query_id"] for d in review_worthy],
            "review_worthy_missed_ids": [d["query_id"] for d in review_worthy
                                         if not d["routing_observed"]["needs_human_review"]],
            "auto_approval_precision": _rate(len(auto_clean), len(auto)),
            "auto_approved_with_findings_ids": [d["query_id"] for d in auto
                                                if d["review_worthy"]["is_review_worthy"]],
            "review_burden": _rate(len(needs_review), n),
            "escalation_reasons": dict(Counter(
                d["routing_observed"]["escalation_reason"] for d in dets)),
            "basis": ("review-worthiness is derived from gold-based deterministic findings "
                      "only: wrong value, invalid citation label, uncited factual claim, "
                      "forbidden content, temporal violation, missed abstention or ambiguity, "
                      "cross-patient leakage, empty/failed answer. The runtime verifier's own "
                      "`unsupported` count is excluded by design — using it would make "
                      "review_recall 1.0 by construction. Incompleteness is also excluded: "
                      "it is a quality problem the grounding verifier has no signal for, and "
                      "is reported separately as `incomplete_answer`."),
            "small_denominator_warning": (
                "review_recall has a denominator of "
                f"{len(review_worthy)}; report it as k/n, not as a percentage"
                if len(review_worthy) < 10 else None),
        },
        "by_category": _breakdown(dets, "category"),
        "by_answer_type": _breakdown(dets, "answer_type"),
        "by_difficulty": _breakdown(dets, "difficulty"),
    }


def _breakdown(dets: list, key: str) -> dict:
    out = {}
    for d in dets:
        out.setdefault(d[key], {"n": 0, "pass": 0})
        out[d[key]]["n"] += 1
        out[d[key]]["pass"] += bool(d["case_pass"])
    return {k: {**v, "rate": round(v["pass"] / v["n"], 4) if v["n"] else None}
            for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
def judge_section(judges: list) -> dict:
    """The independent judge, per dimension, with its failures counted.

    A verdict the judge disowned — `judge_inconsistent`, a parse error, an
    unavailable backend — contributes NO score to any mean. Its numbers are
    preserved on the row for a human to read, but averaging a verdict that
    contradicts itself would launder an evaluation failure into a measurement.
    """
    from src.evals.final_eval.judge import DIMENSIONS, CRITERION_STATUSES
    scorable = [j for j in judges if j["status"] in ("ok", "partial")]
    failures = [j for j in judges if j["status"] not in ("ok", "partial")]
    partial = [j for j in judges if j["status"] == "partial"]
    inconsistent = [j for j in judges if j["status"] == "judge_inconsistent"]
    dims = {}
    for d in DIMENSIONS:
        scores = [j["dimensions"][d]["score"] for j in scorable
                  if (j.get("dimensions") or {}).get(d, {}).get("applicable")
                  and isinstance(j["dimensions"][d].get("score"), int)]
        applicable = sum(1 for j in judges if d in (j.get("applicable_dimensions") or []))
        dims[d] = {
            "n_applicable": applicable,
            "n_scored": len(scores),
            "n_unscored": applicable - len(scores),
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "distribution": {str(s): sum(1 for x in scores if x == s) for s in range(5)},
        }

    # Criterion-level statistics: the aj2 evidence that each required fact was
    # actually inspected, and how often one was not met.
    assessments = [a for j in scorable for a in (j.get("criterion_assessments") or [])]
    supplied = sum(j.get("n_criteria") or 0 for j in scorable)
    status_counts = {s: sum(1 for a in assessments if a.get("status") == s)
                     for s in CRITERION_STATUSES}
    return {
        "n_cases": len(judges),
        "judge_failures": len(failures),
        "judge_failure_ids": [j["query_id"] for j in failures],
        "judge_partial": len(partial),
        "judge_inconsistent": len(inconsistent),
        "judge_inconsistent_ids": [j["query_id"] for j in inconsistent],
        "judge_repairs_attempted": sum(1 for j in judges if j.get("repair_attempted")),
        "consistency_rules_violated": dict(Counter(
            v["rule"] for j in judges for v in (j.get("consistency_violations") or []))),
        "judge_failure_statuses": dict(Counter(j["status"] for j in failures)),
        "cached": sum(1 for j in judges if j.get("cached")),
        "dimensions": dims,
        "criterion_assessments": {
            "n_criteria_supplied": supplied,
            "n_assessed": len(assessments),
            "n_unassessed": max(0, supplied - len(assessments)),
            "by_status": status_counts,
            "cases_with_unmet_criteria": sum(
                1 for j in scorable
                if any(a.get("status") in ("missing", "partially_supported", "contradicted")
                       for a in (j.get("criterion_assessments") or []))),
            "unknown_evidence_labels": sorted({
                l for j in judges for l in (j.get("unknown_evidence_labels") or [])}),
        },
        "note": ("Judge failures are counted, never scored as 0 and never treated as a "
                 "pass. Scores from a disowned verdict (judge_inconsistent, parse or "
                 "backend error) are excluded from every mean. The judge is a second "
                 "opinion, not ground truth."),
    }


# ---------------------------------------------------------------------------
def operational_section(responses: list) -> dict:
    ok = [r for r in responses if r["status"] in
          ("completed", "human_review_required", "refused")]
    tot = [r["timings"].get("total_ms") for r in ok if (r.get("timings") or {}).get("total_ms")]

    def _mean(key):
        vals = [(r.get("timings") or {}).get(key) for r in ok]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(statistics.mean(vals), 2) if vals else None

    return {
        "n": len(responses), "successful": len(ok),
        "error_rate": _rate(len(responses) - len(ok), len(responses)),
        "latency_ms": {"mean": round(statistics.mean(tot), 1) if tot else None,
                       "p50": _pct(tot, 0.50), "p95": _pct(tot, 0.95),
                       "min": min(tot) if tot else None, "max": max(tot) if tot else None},
        "mean_retrieval_ms": _mean("retrieval_ms"),
        "mean_llm_ms": _mean("llm_ms"),
        "mean_llm_calls": _mean("llm_calls"),
        "mean_main_calls": _mean("llm_main_calls"),
        "mean_fast_calls": _mean("llm_fast_calls"),
        "requests_with_zero_llm_calls": sum(
            1 for r in ok if ((r.get("timings") or {}).get("llm_calls") or 0) == 0),
        "status_distribution": dict(Counter(r["status"] for r in responses)),
        "route_distribution": dict(Counter(
            " -> ".join(r.get("node_trail") or []) for r in responses)),
        "query_complexity": dict(Counter(r.get("query_complexity") for r in responses)),
        "classified_by": dict(Counter(r.get("classified_by") for r in responses)),
        "synthesis_role": dict(Counter(r.get("synthesis_role") for r in responses)),
        "review_status": dict(Counter(r.get("review_status") for r in responses)),
        "note": "latency is not optimised or tuned during this evaluation phase",
    }


# ---------------------------------------------------------------------------
def reliability_section(manifest: dict, dets: list, judges: list, responses: list,
                        expected_ids: list) -> dict:
    ids = [r["query_id"] for r in responses]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    missing = [i for i in expected_ids if i not in set(ids)]

    leaks = [d["query_id"] for d in dets if not d["isolation"]["pass"]]
    auto = [d for d in dets if not d["routing_observed"]["needs_human_review"]
            and not d["execution"]["evaluator_error"]]
    auto_bad_cite = [d["query_id"] for d in auto
                     if d["citations"]["hallucinated_labels_pre_strip"]]
    auto_review_worthy = [d["query_id"] for d in auto if d["review_worthy"]["is_review_worthy"]]
    evaluator_errors = [d["query_id"] for d in dets if d["execution"]["evaluator_error"]]
    abstention_fail = [d["query_id"] for d in dets if d["abstention"]["pass"] is False]
    ambiguity_fail = [d["query_id"] for d in dets if d["ambiguity"]["pass"] is False]
    unresolved = [d["query_id"] for d in dets if d["isolation"]["unresolved_labels"]]

    prov_ok = bool(manifest.get("git_sha")) and \
        (manifest.get("eval_set") or {}).get("manifest_sha256_matches") is True

    gates = {
        "zero_cross_patient_leakage": {"pass": not leaks, "offenders": leaks},
        "all_evidence_provenance_resolved": {"pass": not unresolved, "offenders": unresolved,
                                             "note": "an unresolved label cannot be cleared "
                                                     "of leakage, so it fails the gate"},
        "zero_invalid_citations_on_auto_approved": {"pass": not auto_bad_cite,
                                                    "offenders": auto_bad_cite},
        "no_unsafe_auto_approval": {"pass": not auto_review_worthy,
                                    "offenders": auto_review_worthy,
                                    "note": "auto-approved despite an independently "
                                            "detected review-worthy finding"},
        "no_hidden_evaluator_errors": {"pass": not evaluator_errors, "offenders": evaluator_errors},
        "expected_abstention_cases_pass": {"pass": not abstention_fail, "offenders": abstention_fail},
        "expected_ambiguity_cases_pass": {"pass": not ambiguity_fail, "offenders": ambiguity_fail},
        "run_provenance_complete": {"pass": prov_ok,
                                    "git_sha": bool(manifest.get("git_sha")),
                                    "eval_set_hash_matches":
                                        (manifest.get("eval_set") or {}).get("manifest_sha256_matches")},
        "no_duplicate_query_ids": {"pass": not dupes, "offenders": dupes},
        "no_missing_cases": {"pass": not missing, "offenders": missing},
    }
    return {"all_gates_pass": all(g["pass"] for g in gates.values()), "gates": gates,
            "run_completeness": {"expected": len(expected_ids), "collected": len(ids),
                                 "deterministic": len(dets), "judged": len(judges),
                                 "duplicates": dupes, "missing": missing}}


# ---------------------------------------------------------------------------
def build_summary(run_dir, manifest: dict) -> dict:
    responses = run_dir.read_jsonl("responses")
    dets = run_dir.read_jsonl("deterministic")
    judges = run_dir.read_jsonl("judge")
    expected = (manifest.get("eval_set") or {}).get("evaluated_ids") or []
    return {
        "run_id": manifest.get("run_id"),
        "benchmark": manifest.get("benchmark"),
        "evaluator_version": EVALUATOR_VERSION,
        "results_schema_version": manifest.get("results_schema_version"),
        "judge_prompt_version": (manifest.get("judge") or {}).get("prompt_version"),
        "generated_from": "this run's raw artifacts only "
                          "(responses.jsonl, deterministic.jsonl, judge.jsonl)",
        "eval_set": {k: (manifest.get("eval_set") or {}).get(k)
                     for k in ("path", "sha256", "dataset_version", "subset", "n_evaluated")},
        "models": {k: (manifest.get("models") or {}).get(k) for k in ("main", "fast")},
        "prompt_versions": manifest.get("prompt_versions"),
        "judge_config": {k: (manifest.get("judge") or {}).get(k)
                         for k in ("model", "backend", "prompt_version")},
        "reliability": reliability_section(manifest, dets, judges, responses, expected),
        "deterministic_quality": deterministic_section(dets),
        "judge_quality": judge_section(judges),
        "operational": operational_section(responses),
        "no_composite_score": ("Sections are reported independently. A single weighted "
                               "number would let a strong latency result mask a safety "
                               "failure, so none is produced."),
    }


# ---------------------------------------------------------------------------
def _small_n_note(d: dict) -> str:
    """Name the metrics whose denominators are too small to read as a percentage,
    using this run's actual counts rather than a sentence fixed in advance."""
    small = [(label, m["d"]) for label, m in (
        ("unsupported abstention", d["unsupported_abstention_accuracy"]),
        ("ambiguity handling", d["ambiguity_handling_accuracy"]),
        ("review recall", d["routing"]["review_recall"]),
        ("temporal accuracy", d["temporal_accuracy"]),
    ) if 0 < m["d"] < 10]
    if not small:
        return ("Every rate carries its own k/n. A metric with no applicable cases is "
                "reported as n/a, never as 0% or 100%.")
    return ("Small denominators — "
            + ", ".join(f"{k} n={v}" for k, v in small)
            + " — are reported as k/n deliberately; do not read them as percentages.")


def _fmt(r) -> str:
    if r is None or r.get("rate") is None:
        return f"n/a ({r['n']}/{r['d']})" if r else "n/a"
    return f"{r['rate']:.1%} ({r['n']}/{r['d']})"


def build_report(summary: dict, failure_counts: dict, manifest: dict) -> str:
    d, j, o, rel = (summary["deterministic_quality"], summary["judge_quality"],
                    summary["operational"], summary["reliability"])
    L = [
        f"# {summary['benchmark']}",
        "",
        f"- **run_id** `{summary['run_id']}`",
        f"- **git** `{(manifest.get('git_sha') or 'unknown')[:12]}` "
        f"(branch `{manifest.get('branch')}`, dirty={manifest.get('dirty_worktree')})",
        f"- **eval set** `{summary['eval_set']['path']}` "
        f"sha256 `{(summary['eval_set']['sha256'] or '')[:12]}` "
        f"subset `{summary['eval_set']['subset']}` n={summary['eval_set']['n_evaluated']}",
        f"- **models** main `{summary['models']['main']}` / fast `{summary['models']['fast']}`",
        f"- **prompts** {summary['prompt_versions']}",
        f"- **independent judge** `{summary['judge_config']['model']}` "
        f"(prompt {summary['judge_config']['prompt_version']})",
        f"- **collection backend** {manifest.get('collection_backend')}",
        "",
        "## 1. Reliability — hard gates",
        "",
        "| Gate | Result | Offenders |",
        "|---|---|---|",
    ]
    for name, g in rel["gates"].items():
        L.append(f"| {name} | {'PASS' if g['pass'] else 'FAIL'} | "
                 f"{', '.join(map(str, g.get('offenders') or [])) or '—'} |")
    c = rel["run_completeness"]
    L += ["", f"**Run completeness** — expected {c['expected']}, collected {c['collected']}, "
              f"deterministic {c['deterministic']}, judged {c['judged']}; "
              f"duplicates {c['duplicates'] or 'none'}; missing {c['missing'] or 'none'}.",
          "", "## 2. Deterministic quality", "",
          "| Metric | Value |", "|---|---|",
          f"| completion rate | {_fmt(d['completion_rate'])} |",
          f"| deterministic case pass | {_fmt(d['case_pass_rate'])} |",
          f"| fact recall | {_fmt(d['facts']['fact_recall'])} |",
          f"| factual contradictions | {d['facts']['contradictions']} |",
          f"| strict-unmatched facts (not errors) | {d['facts']['strict_unmatched']} |",
          f"| citation validity | {_fmt(d['citations']['citation_validity_rate'])} |",
          f"| citation coverage | {_fmt(d['citations']['citation_coverage'])} |",
          f"| hallucinated labels (pre-strip) | {d['citations']['hallucinated_labels_pre_strip']} |",
          f"| temporal accuracy | {_fmt(d['temporal_accuracy'])} |",
          f"| unsupported abstention | {_fmt(d['unsupported_abstention_accuracy'])} |",
          f"| ambiguity handling | {_fmt(d['ambiguity_handling_accuracy'])} |",
          f"| admission scope | {_fmt(d['admission_scope_accuracy'])} |",
          f"| synthesis failures | {_fmt(d['synthesis_failure_rate'])} |",
          f"| invalid responses | {_fmt(d['invalid_response_rate'])} |",
          f"| evaluator errors | {_fmt(d['evaluator_error_rate'])} |",
          ""]
    r = d["routing"]
    L += ["### Human-review routing", "",
          f"- **review recall** {_fmt(r['review_recall'])}"
          + (f" — {r['small_denominator_warning']}" if r["small_denominator_warning"] else ""),
          f"- **auto-approval precision** {_fmt(r['auto_approval_precision'])}",
          f"- **review burden** {_fmt(r['review_burden'])} (operational cost, not a safety penalty)",
          f"- review-worthy cases: {r['review_worthy_ids'] or 'none'}",
          f"- missed by routing: {r['review_worthy_missed_ids'] or 'none'}",
          "", f"> {r['basis']}", "",
          "## 3. Independent judge quality", "",
          "| Dimension | Applicable | Scored | Mean | Median | 0 | 1 | 2 | 3 | 4 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for dim, s in j["dimensions"].items():
        dist = s["distribution"]
        L.append(f"| {dim} | {s['n_applicable']} | {s['n_scored']} | "
                 f"{s['mean'] if s['mean'] is not None else '—'} | "
                 f"{s['median'] if s['median'] is not None else '—'} | "
                 + " | ".join(str(dist[str(k)]) for k in range(5)) + " |")
    ca = j["criterion_assessments"]
    L += ["", f"**Judge failures: {j['judge_failures']}** {j['judge_failure_ids'] or ''} "
              f"(partial: {j['judge_partial']}, self-inconsistent: {j['judge_inconsistent']} "
              f"{j['judge_inconsistent_ids'] or ''}). {j['note']}", "",
          "### Criterion assessments (aj2)", "",
          f"- criteria supplied {ca['n_criteria_supplied']}, assessed {ca['n_assessed']}, "
          f"unassessed {ca['n_unassessed']}",
          "- by status: " + ", ".join(f"{k} {v}" for k, v in ca["by_status"].items()),
          f"- cases with at least one unmet criterion: {ca['cases_with_unmet_criteria']}",
          f"- consistency rules violated: {j['consistency_rules_violated'] or 'none'} "
          f"(repairs attempted: {j['judge_repairs_attempted']})",
          "",
          "## 4. Operational", "",
          f"- latency ms: mean {o['latency_ms']['mean']}, p50 {o['latency_ms']['p50']}, "
          f"p95 {o['latency_ms']['p95']}",
          f"- error rate {_fmt(o['error_rate'])}",
          f"- mean LLM calls {o['mean_llm_calls']} (main {o['mean_main_calls']}, "
          f"fast {o['mean_fast_calls']}); zero-LLM requests {o['requests_with_zero_llm_calls']}",
          f"- status distribution {o['status_distribution']}",
          f"- review status {o['review_status']}",
          "", "## 5. Failure taxonomy", ""]
    if failure_counts.get("by_tag"):
        L += ["| Tag | Count |", "|---|---|"]
        L += [f"| {t} | {n} |" for t, n in failure_counts["by_tag"].items()]
        L += ["", f"{failure_counts['total_findings']} findings across "
                  f"{failure_counts['cases_with_failures']} cases. One case may carry "
                  f"several tags. Full detail in `failures.jsonl`."]
    else:
        L.append("No failures recorded.")
    L += ["", "---", "",
          "### Reading notes", "",
          "- `strict_unmatched` is an unconfirmed fact, not a factual error. Only "
          "`contradiction` is a deterministic error.",
          "- Judge scores are a second opinion, not ground truth. Judge failures are "
          "counted separately and never scored as 0.",
          "- A `judge_inconsistent` verdict contradicted its own criterion assessments "
          "twice. Its scores are kept on the row but excluded from every mean, and the "
          "case is not counted as passed.",
          "- " + _small_n_note(d),
          "- No composite score is produced.",
          ""]
    return "\n".join(L)
