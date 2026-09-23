"""
Phase 3 — deterministic answer-level evaluation
===============================================
Checks that ground truth can settle in code, with no model in the loop. Every
check records WHY it reached its verdict, so a per-case row can be read on its
own without re-running anything.

Three principles hold throughout:

1. A strict text mismatch is NOT a factual error. A structured fact (number,
   unit, date, typed dose) can be deterministically CONTRADICTED. A text-only
   fact can only be confirmed or left unconfirmed — `strict_unmatched` means
   "deterministic matching could not settle this", and the independent judge
   resolves semantic paraphrase. Both values are preserved per case.

2. Unsupported abstention and ambiguity handling are different behaviours and
   are scored separately. A correct answer to an ambiguous case states the
   conflict; it does not have to emit the hard refusal sentence.

3. Review-worthiness is derived ONLY from gold-based findings here, never from
   the runtime verifier's own `unsupported` count. Deriving it from the
   verifier would make review_recall 1.0 by construction and measure nothing.
"""

from __future__ import annotations

from decimal import Decimal

from src.agents import citations as cite_mod
from src.evals.final_eval import normalize as N

# ---------------------------------------------------------------------------
# Lexical vocabularies. Small, explicit and reviewed — never generated.
# ---------------------------------------------------------------------------
# The exact sentence src/agents/prompts.py instructs synthesis to emit when the
# evidence does not answer the question.
HARD_REFUSAL = cite_mod._REFUSAL          # "the available records do not contain enough information"

# Markers that state uncertainty without being the hard refusal. Used ONLY for
# the ambiguity cases, where a correct answer explains the conflict rather than
# declining. Lexical by design; the judge's abstention_quality is the semantic
# counterpart and the two are reported side by side.
UNCERTAINTY_MARKERS = (
    "could not be clarified", "not clarified", "unclear", "not clear",
    "conflicting", "inconsistent", "discrepan", "contradict",
    "pending", "on hold", "unresolved", "not documented", "not recorded",
    "uncertain", "ambiguous", "not confirmed", "cannot be confirmed",
    "requires clarification", "should be clarified", "not established",
    "no documentation", "does not document", "not specified",
)

FACT_MATCH = "strict_match"
FACT_CONTRADICTION = "contradiction"
FACT_UNMATCHED = "strict_unmatched"


# ---------------------------------------------------------------------------
# Fact matching
# ---------------------------------------------------------------------------
def _multi_valued(fact, siblings) -> bool:
    """Does the GOLD itself assert several values for this measurand and unit?

    A trend case lists a sequence of time points ("creatinine 1.8 / 2.1 / 1.4
    mg/dL"). There, rival values in the answer are the EXPECTED state, so a
    missing gold value means the answer omitted a time point — incompleteness,
    not a wrong number. Contradiction detection is therefore restricted to
    measurands the gold pins to exactly one value.
    """
    if not fact.quantities:
        return False

    def key(f):
        return (tuple(sorted(f.terms)), tuple(sorted({q.unit for q in f.quantities})))

    k = key(fact)
    return sum(1 for f in siblings
               if f.kind == "structured" and f.quantities and key(f) == k) > 1


def match_fact(fact, answer: str, siblings=()) -> dict:
    """One gold fact against the answer.

    structured -> strict_match | contradiction | strict_unmatched
    text_only  -> strict_match | strict_unmatched      (never contradiction)

    `siblings` is the case's full fact list, used only to tell a single-valued
    measurand from a multi-valued trend (see _multi_valued).
    """
    if fact.kind == "text_only":
        hit = N.contains_phrase(fact.text, answer)
        found = [t for t in fact.terms if N.contains_phrase(t, answer)]
        return {
            "fact": fact.text, "kind": fact.kind,
            "outcome": FACT_MATCH if hit else FACT_UNMATCHED,
            "detail": ("exact phrase present" if hit else
                       f"phrase absent; terms present {found or []} of {list(fact.terms)}"),
        }

    missing_dates = [d for d in fact.dates if not N.contains_date(d, answer)]
    missing_qty = [str(q) for q in fact.quantities if not N.contains_quantity(q, answer)]
    missing_nums = [N._plain(n) for n in fact.bare_numbers if not N.contains_number(n, answer)]
    term_ok = (not fact.terms) or any(N.contains_phrase(t, answer) for t in fact.terms)

    if not missing_dates and not missing_qty and not missing_nums and term_ok:
        return {"fact": fact.text, "kind": fact.kind, "outcome": FACT_MATCH,
                "detail": "all anchors present"}

    contradiction = None if _multi_valued(fact, siblings) else _contradiction(fact, answer)
    if contradiction:
        return {"fact": fact.text, "kind": fact.kind, "outcome": FACT_CONTRADICTION,
                "detail": contradiction}

    bits = []
    if missing_qty:
        bits.append(f"quantities absent {missing_qty}")
    if missing_dates:
        bits.append(f"dates absent {missing_dates}")
    if missing_nums:
        bits.append(f"numbers absent {missing_nums}")
    if not term_ok:
        bits.append(f"none of the terms {list(fact.terms)} appear")
    return {"fact": fact.text, "kind": fact.kind, "outcome": FACT_UNMATCHED,
            "detail": "; ".join(bits) or "not confirmed"}


def _contradiction(fact, answer: str):
    """A deterministic factual contradiction, or None.

    Scoped to sentences that mention one of the fact's own identifying terms,
    so an unrelated number elsewhere in the answer can never manufacture one.
    A contradiction requires BOTH that the gold value is absent AND that a
    different value for the same measurement, in the same unit, is asserted.
    A fact with no identifying term (e.g. "4 liters") cannot be scoped and is
    therefore never reported as a contradiction, and neither is a measurand the
    gold itself gives several values for (see _multi_valued).

    Dates are deliberately excluded: answers legitimately cite several dates,
    and a missing gold date is not evidence that a stated one is wrong.
    """
    if not fact.terms or not fact.quantities:
        return None
    scoped = [s for s in N.split_sentences(answer)
              if any(N.contains_phrase(t, s) for t in fact.terms)]
    if not scoped:
        return None
    text = " ".join(scoped)
    for q in fact.quantities:
        if N.contains_quantity(q, text):
            continue
        rival = sorted({N._plain(x.value) for x in N.parse_quantities(text) if x.unit == q.unit})
        if rival:
            return (f"expected {q}, answer states {', '.join(rival)} {q.unit} "
                    f"for '{' / '.join(fact.terms)}'")
    return None


# ---------------------------------------------------------------------------
# Citation integrity
# ---------------------------------------------------------------------------
def _is_factual_claim(claim: str) -> bool:
    """A sentence that asserts something and therefore needs a citation.

    Excludes the hard refusal sentence (declining is not a claim about the
    record) and citation-only fragments (`[S1]` on a line of its own), which
    src/agents/citations.py already treats as formatting, not content.
    """
    text = (claim or "").strip()
    if not text or cite_mod._CITE_ONLY_RE.match(text):
        return False
    if HARD_REFUSAL in text.lower():
        return False
    return bool(N.content_tokens(text))


def check_citations(row: dict) -> dict:
    """Citation integrity, measured on BOTH sides of the runtime repair.

    src/agents/graph.py:synthesis strips hallucinated labels out of the stored
    answer, so re-validating that answer always shows zero bad labels. The
    honest measure of model behaviour is the pre-strip list the graph preserves
    in verification.citation_report.bad_labels; the post-strip view is what a
    reader actually sees. Both are reported.
    """
    answer = row.get("answer") or ""
    labels_available = {s.get("label") for s in (row.get("sources") or []) if s.get("label")}
    stub_evidence = [{"label": l, "text": ""} for l in sorted(labels_available)]

    normalized = cite_mod.normalize_orphan_citations(answer)
    report = cite_mod.validate(normalized, stub_evidence)

    factual = [c for c in report["claims"] if _is_factual_claim(c["claim"])]
    cited = [c for c in factual if c["valid_labels"]]
    uncited = [c["claim"][:120] for c in factual if not c["valid_labels"]]

    pre_strip_bad = list((row.get("citation_report") or {}).get("bad_labels") or [])
    post_strip_bad = list(report["bad_labels"])

    return {
        "n_claims": report["n_claims"],
        "n_factual_claims": len(factual),
        "n_cited_factual_claims": len(cited),
        "citation_coverage": (len(cited) / len(factual)) if factual else None,
        "uncited_factual_claims": uncited,
        "hallucinated_labels_pre_strip": pre_strip_bad,
        "hallucinated_labels_post_strip": post_strip_bad,
        "labels_available": sorted(labels_available),
        "is_refusal": report["is_refusal"],
        "citation_syntax_valid": True,          # CITE_RE only matches well-formed markers
        "empty_citation_only_claims": sum(
            1 for c in report["claims"] if cite_mod._CITE_ONLY_RE.match(c["claim"].strip())),
    }


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
def check_temporal(case, row: dict) -> dict:
    """Patient-relative temporal correctness.

    latest  the gold latest value must be asserted, with no contradiction.
            Mentioning older values as context is legitimate (the gold answers
            themselves do it) and is not penalised here.
    trend   every gold value must be present AND in the gold list's order,
            which the source authors chronologically (verified against the
            expected_answer text of every trend case).
    """
    structured = [f for f in case.expected_facts if f.kind == "structured" and f.quantities]
    if not case.temporal_applicable or case.unsupported or not structured:
        return {"applicable": False, "mode": case.temporal, "pass": None,
                "detail": "no structured temporal anchor in the gold case"}

    answer = row.get("answer") or ""
    results = [match_fact(f, answer, case.expected_facts) for f in structured]

    if case.temporal in ("latest", "earliest"):
        ok = all(r["outcome"] == FACT_MATCH for r in results)
        return {"applicable": True, "mode": case.temporal, "pass": ok,
                "detail": (f"{case.temporal} gold value asserted" if ok else
                           "; ".join(f"{r['fact']}: {r['outcome']}" for r in results))}

    # trend
    if any(r["outcome"] == FACT_CONTRADICTION for r in results):
        return {"applicable": True, "mode": "trend", "pass": False,
                "detail": "; ".join(f"{r['fact']}: {r['detail']}" for r in results
                                    if r["outcome"] == FACT_CONTRADICTION)}
    if any(r["outcome"] != FACT_MATCH for r in results):
        return {"applicable": True, "mode": "trend", "pass": False,
                "detail": "value(s) absent: " + "; ".join(
                    r["fact"] for r in results if r["outcome"] != FACT_MATCH)}
    order_ok, detail = _chronological(structured, answer)
    return {"applicable": True, "mode": "trend", "pass": order_ok, "detail": detail}


def _chronological(structured, answer: str) -> tuple:
    """Do the gold values appear in the answer in gold (chronological) order?

    Uses first occurrence of each value, matched numerically and by unit. A
    repeated value is anchored to its first mention, which is the conservative
    reading: it cannot rescue an out-of-order sequence.
    """
    positions = []
    quantities = [q for f in structured for q in f.quantities]
    for q in quantities:
        pos = _first_position(q, answer)
        if pos is None:
            return False, f"value {q} not locatable for ordering"
        positions.append((str(q), pos))
    seq = [p for _, p in positions]
    if seq == sorted(seq):
        return True, "values appear in chronological (gold) order: " + " -> ".join(
            n for n, _ in positions)
    return False, ("values out of chronological order; gold "
                   + " -> ".join(n for n, _ in positions)
                   + f", answer positions {seq}")


def _first_position(q, answer: str):
    """Character offset of the first mention of this quantity, or None."""
    src = N.strip_citations(answer)
    for m in N._QTY_RE.finditer(src):
        val = N._dec(m.group(1))
        if val is not None and val == q.value and N.canon_unit(m.group(2)) == q.unit:
            return m.start()
    return None


# ---------------------------------------------------------------------------
# Abstention and ambiguity
# ---------------------------------------------------------------------------
def _uncertainty_markers(answer: str) -> list:
    low = N.norm_text(answer)
    return [m for m in UNCERTAINTY_MARKERS if N.norm_text(m) in low]


def check_abstention(case, row: dict, cites: dict) -> dict:
    """Unsupported-question handling: the record genuinely lacks the answer."""
    if not case.expects_abstention:
        return {"applicable": False, "pass": None, "detail": "case is answerable"}
    answer = row.get("answer") or ""
    refused = HARD_REFUSAL in answer.lower() or bool(_uncertainty_markers(answer))
    violations = _mnc_violations(case, answer)
    fabricated = cites["hallucinated_labels_pre_strip"]
    ok = refused and not violations and not fabricated
    return {"applicable": True, "pass": ok,
            "refusal_detected": refused,
            "must_not_contain_violations": violations,
            "fabricated_citations": fabricated,
            "detail": ("declined without fabricating evidence" if ok else
                       f"refused={refused} violations={violations} fabricated={fabricated}")}


def check_ambiguity(case, row: dict) -> dict:
    """Ambiguity handling: the record is conflicting or unresolved.

    A correct answer states the uncertainty. It does NOT have to use the hard
    refusal sentence, so a hard refusal and an explained conflict both pass.
    """
    if not case.expects_ambiguity:
        return {"applicable": False, "pass": None, "detail": "case is not ambiguous"}
    answer = row.get("answer") or ""
    markers = _uncertainty_markers(answer)
    hard = HARD_REFUSAL in answer.lower()
    violations = _mnc_violations(case, answer)
    ok = bool(markers or hard) and not violations
    return {"applicable": True, "pass": ok, "markers": markers, "hard_refusal": hard,
            "must_not_contain_violations": violations,
            "detail": ("uncertainty stated explicitly" if ok else
                       "no uncertainty marker found" if not (markers or hard) else
                       f"must_not_contain violated {violations}")}


def _mnc_violations(case, answer: str) -> list:
    return [t for t in case.must_not_contain if N.contains_phrase(t, answer)]


# ---------------------------------------------------------------------------
# Isolation and admission scope
# ---------------------------------------------------------------------------
def check_isolation(case, row: dict) -> dict:
    """Cross-patient leakage. A HARD failure: any cited or retrieved evidence
    belonging to another subject fails the whole run.

    An unresolved label is reported, never assumed clean — the check fails
    open (visible) rather than closed (silent pass).
    """
    leaked, unresolved = [], []
    for s in row.get("sources") or []:
        sid, res = s.get("subject_id"), s.get("resolution")
        if res == "unresolved" or sid is None:
            unresolved.append(s.get("label"))
        elif int(sid) != int(case.subject_id):
            leaked.append({"label": s.get("label"), "chunk_id": s.get("chunk_id"),
                           "subject_id": sid})
    return {"pass": not leaked, "leaked": leaked, "unresolved_labels": unresolved,
            "n_sources": len(row.get("sources") or [])}


def check_admission_scope(case, row: dict) -> dict:
    """Are cited chunks drawn from the gold admissions?

    Reported, never folded into the case pass: the gold field lists the
    admissions where the evidence LIVES, and a correct answer may legitimately
    cite context from an adjacent admission.
    """
    if not case.admission_scope_applicable:
        return {"applicable": False, "pass": None, "detail": "no gold admissions"}
    want = {int(h) for h in case.evidence_hadm_ids}
    cited_labels = {c.get("label") for c in (row.get("citations") or []) if c.get("label")}
    for c in row.get("citations") or []:
        cited_labels.update(c.get("labels") or [])
    out_of_scope, unknown = [], []
    for s in row.get("sources") or []:
        if s.get("label") not in cited_labels:
            continue
        h = s.get("hadm_id")
        if h is None:
            unknown.append(s.get("label"))
        elif int(h) not in want:
            out_of_scope.append({"label": s.get("label"), "hadm_id": h})
    return {"applicable": True, "pass": not out_of_scope, "expected_hadm_ids": sorted(want),
            "out_of_scope": out_of_scope, "unknown_admission_labels": unknown}


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------
def evaluate_case(case, row: dict) -> dict:
    """All deterministic checks for one collected response."""
    status = row.get("status")
    answer = row.get("answer") or ""
    executed = status in ("completed", "human_review_required", "refused")
    execution = {
        "status": status,
        "completed": bool(executed and answer.strip()),
        "evaluator_error": status == "evaluator_error",
        "error_type": row.get("error_type"),
        "synthesis_failed": bool((row.get("verification_summary") or {}).get("synthesis_failed")
                                 or status == "failed"),
        "empty_answer": not answer.strip(),
        "valid_schema": _schema_ok(row),
        "graph_errors": row.get("graph_errors") or [],
    }

    cites = check_citations(row)
    fact_results = [match_fact(f, answer, case.expected_facts) for f in case.expected_facts]
    matched = sum(1 for r in fact_results if r["outcome"] == FACT_MATCH)
    contradictions = [r for r in fact_results if r["outcome"] == FACT_CONTRADICTION]
    unmatched = [r for r in fact_results if r["outcome"] == FACT_UNMATCHED]
    facts = {
        "results": fact_results,
        "n_expected": len(fact_results),
        "n_structured": case.n_structured_facts,
        "matched": matched,
        "n_contradictions": len(contradictions),
        "n_strict_unmatched": len(unmatched),
        "min_facts": case.min_facts,
        "threshold_met": matched >= case.min_facts,
    }
    mnc = {"terms": list(case.must_not_contain), "violations": _mnc_violations(case, answer)}
    temporal = check_temporal(case, row)
    abstention = check_abstention(case, row, cites)
    ambiguity = check_ambiguity(case, row)
    isolation = check_isolation(case, row)
    admission = check_admission_scope(case, row)

    # --- case pass (approved definition) --------------------------------
    # Citation metrics stay OUT of this on purpose and are reported separately.
    components = {
        "executed": execution["completed"],
        "fact_threshold_met": facts["threshold_met"],
        "no_factual_contradiction": not contradictions,
        "no_must_not_contain_violation": not mnc["violations"],
        "no_cross_patient_leakage": isolation["pass"],
        "temporal_ok": temporal["pass"] is not False,
        "abstention_ok": abstention["pass"] is not False,
        "ambiguity_ok": ambiguity["pass"] is not False,
    }
    case_pass = all(components.values())

    review = _review_worthy(execution, cites, facts, mnc, temporal, abstention,
                            ambiguity, isolation)
    return {
        "query_id": case.query_id,
        "subject_id": case.subject_id,
        "category": case.category,
        "answer_type": case.answer_type,
        "temporal_mode_gold": case.temporal,
        "difficulty": case.difficulty,
        "execution": execution,
        "citations": cites,
        "facts": facts,
        "must_not_contain": mnc,
        "temporal": temporal,
        "abstention": abstention,
        "ambiguity": ambiguity,
        "isolation": isolation,
        "admission_scope": admission,
        "case_pass": case_pass,
        "case_pass_components": components,
        "review_worthy": review,
        "routing_observed": {
            "needs_human_review": bool(row.get("needs_human_review")),
            "review_status": row.get("review_status"),
            "escalation_reason": row.get("escalation_reason"),
            "deterministic_lab_path": bool(row.get("deterministic_lab_path")),
        },
        "failure_tags": _failure_tags(execution, cites, facts, mnc, temporal,
                                      abstention, ambiguity, isolation, admission,
                                      review, row),
    }


_REQUIRED_ROW_KEYS = ("query_id", "status", "answer", "citations", "sources", "node_trail")


def _schema_ok(row: dict) -> bool:
    if any(k not in row for k in _REQUIRED_ROW_KEYS):
        return False
    if not isinstance(row.get("citations"), list) or not isinstance(row.get("sources"), list):
        return False
    return all(isinstance(c, dict) and "claim" in c and "verified" in c
               for c in row["citations"])


def _review_worthy(execution, cites, facts, mnc, temporal, abstention, ambiguity, isolation) -> dict:
    """Independent, gold-derived grounds for a human to look at this answer.

    Deliberately excludes the runtime verifier's own `unsupported` count and
    `needs_human_review` flag: review_recall must measure whether routing
    CAUGHT independently-detectable problems, not whether the verifier agrees
    with itself.

    Also deliberately EXCLUDES incompleteness (matched facts below min_facts).
    An answer that is correct as far as it goes is a quality problem, not a
    safety one, and the runtime verifier checks grounding — it has no signal
    for "you should also have mentioned X". Counting incompleteness here would
    inflate the denominator with findings routing cannot detect and would make
    review_recall look worse for the wrong reason. Incompleteness is reported
    in its own right as `incomplete_answer` and by the judge's completeness
    dimension.
    """
    reasons = []
    if execution["empty_answer"] or execution["synthesis_failed"]:
        reasons.append("empty_or_failed_answer")
    if cites["hallucinated_labels_pre_strip"]:
        reasons.append("invalid_citation_label")
    if cites["uncited_factual_claims"]:
        reasons.append("uncited_factual_claim")
    if facts["n_contradictions"]:
        reasons.append("factual_contradiction")
    if mnc["violations"]:
        reasons.append("must_not_contain_violation")
    if temporal["pass"] is False:
        reasons.append("temporal_violation")
    if abstention["pass"] is False:
        reasons.append("missed_abstention")
    if ambiguity["pass"] is False:
        reasons.append("missed_ambiguity")
    if not isolation["pass"]:
        reasons.append("cross_patient_leakage")
    return {"is_review_worthy": bool(reasons), "reasons": reasons,
            "basis": "gold-derived deterministic findings only; runtime verifier "
                     "verdicts are excluded by design"}


def _failure_tags(execution, cites, facts, mnc, temporal, abstention, ambiguity,
                  isolation, admission, review, row) -> list:
    """Phase 7 taxonomy. One case may carry several tags.

    The full vocabulary and what each tag means live in
    src/evals/final_eval/failures.py:TAXONOMY.
    """
    tags = []
    if execution["evaluator_error"]:
        tags.append("evaluator_error")
    if execution["synthesis_failed"]:
        tags.append("synthesis_failure")
    # A graph that recorded an error and still answered is distinct from one
    # that failed outright: the answer exists and is scored, but something went
    # wrong producing it and that must not vanish from the taxonomy.
    if execution["graph_errors"] and not execution["synthesis_failed"] \
            and not execution["evaluator_error"]:
        tags.append("execution_error")
    if execution["empty_answer"] and not execution["evaluator_error"]:
        tags.append("empty_answer")
    if not execution["valid_schema"]:
        tags.append("invalid_response_schema")
    if cites["hallucinated_labels_pre_strip"]:
        tags.append("invalid_citation_label")
    # Post-strip is expected to be empty because synthesis repairs the answer.
    # If one ever survives into the stored answer a reader sees it, so it is
    # tagged separately rather than folded into the pre-strip count.
    if cites["hallucinated_labels_post_strip"]:
        tags.append("invalid_visible_citation")
    if cites["uncited_factual_claims"]:
        tags.append("missing_citation")
    if facts["n_contradictions"]:
        tags.append("incorrect_value")
    if not facts["threshold_met"]:
        tags.append("incomplete_answer")
    if facts["n_strict_unmatched"] and facts["threshold_met"]:
        tags.append("unconfirmed_fact")
    if mnc["violations"]:
        tags.append("forbidden_content")
    if temporal["pass"] is False:
        tags.append("incorrect_temporal_interpretation")
    if abstention["pass"] is False:
        tags.append("missed_refusal")
    if ambiguity["pass"] is False:
        tags.append("missed_ambiguity")
    if not isolation["pass"]:
        tags.append("cross_patient_leakage")
    if isolation["unresolved_labels"]:
        tags.append("unresolved_evidence_provenance")
    if admission["pass"] is False:
        tags.append("admission_scope_violation")
    if review["is_review_worthy"] and not row.get("needs_human_review"):
        tags.append("review_routing_miss")
    return tags


def run(run_dir, case_index: dict, progress=None) -> dict:
    """Evaluate every collected response into deterministic.jsonl."""
    say = progress or (lambda *_a, **_k: None)
    rows = run_dir.read_jsonl("responses")
    run_dir.file("deterministic").unlink(missing_ok=True)
    stats = {"evaluated": 0, "case_pass": 0, "review_worthy": 0}
    for row in rows:
        case = case_index.get(row["query_id"])
        if case is None:
            continue
        out = evaluate_case(case, row)
        run_dir.append("deterministic", out)
        stats["evaluated"] += 1
        stats["case_pass"] += bool(out["case_pass"])
        stats["review_worthy"] += bool(out["review_worthy"]["is_review_worthy"])
        say(f"  {out['query_id']:<10} pass={str(out['case_pass']):<5} "
            f"facts={out['facts']['matched']}/{out['facts']['n_expected']} "
            f"contra={out['facts']['n_contradictions']} tags={out['failure_tags']}")
    return stats
