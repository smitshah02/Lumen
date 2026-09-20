"""
Lumen final answer-level evaluation — CLI
=========================================
Deterministic answer-level evaluation, an independent offline LLM judge, and
versioned run artifacts for the FROZEN Lumen architecture.

    LUMEN_DATA_PLANE=demo python scripts/final_eval.py doctor
    LUMEN_DATA_PLANE=demo python scripts/final_eval.py run --subset smoke
    LUMEN_DATA_PLANE=demo python scripts/final_eval.py run --subset all --run-id my-run
    LUMEN_DATA_PLANE=demo python scripts/final_eval.py judge --run-id my-run --resume
    python scripts/final_eval.py score     --run-id my-run
    python scripts/final_eval.py calibrate --run-id my-run
    python scripts/final_eval.py show      --run-id my-run

`doctor` is diagnostic only: it starts nothing, writes nothing and returns
non-zero when a prerequisite for a real benchmark is missing. `calibrate` and
`compare` read existing artifacts and never call a model.

Artifacts land in results/final_eval/<run_id>/ and a completed run is immutable.

The independent judge is configured with LUMEN_JUDGE_MODEL and must not be
either runtime tier; the framework refuses to start otherwise rather than
quietly judging the system with itself.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evals.final_eval import EVALUATOR_VERSION, cases as case_mod      # noqa: E402
from src.evals.final_eval import manifest as man                            # noqa: E402
from src.evals.final_eval import aggregate as agg                           # noqa: E402
from src.evals.final_eval import failures as fail_mod                       # noqa: E402

log = logging.getLogger("final_eval")


def say(*a, **k):
    print(*a, **k, flush=True)


def _require_demo_plane() -> None:
    """The evaluation runs against the synthetic demo plane only. Same refusal
    scripts/cloud_eval.py makes, for the same reason."""
    if os.environ.get("LUMEN_DATA_PLANE") != "demo":
        raise SystemExit("refusing: LUMEN_DATA_PLANE must be demo")


def _selected(args) -> list:
    return case_mod.select(case_mod.load_cases(), ids=args.ids, subset=args.subset)


def _judge_backend(args, required: bool):
    """Build the independent judge, or explain precisely why there is none."""
    from src.evals.final_eval import judge_backend as jb
    model = args.judge_model or jb.DEFAULT_JUDGE_MODEL
    try:
        backend = jb.build_backend(model=model, host=args.judge_host)
    except jb.JudgeNotIndependent as e:
        raise SystemExit(f"refusing: {e}")
    if required:
        d = backend.digest()
        if d.get("installed") is False:
            raise SystemExit(
                f"refusing: judge model {model!r} is not installed on "
                f"{backend.describe()['host']}. The framework does not fall back to a "
                f"runtime model. Pull it, or pass --judge-model.")
    return backend


def _judge_config(backend) -> dict:
    from src.evals.final_eval.judge import JUDGE_PROMPT_VERSION, DEFAULT_CACHE_PATH, DIMENSIONS
    from src.evals.final_eval import judge_backend as jb
    return {**backend.describe(), "digest": backend.digest().get("digest"),
            "prompt_version": JUDGE_PROMPT_VERSION,
            "rubric": "0-4 per dimension",
            "dimensions": list(DIMENSIONS),
            "cache_path": str(Path(DEFAULT_CACHE_PATH).name),
            "independent_of_runtime": True,
            "runtime_models_excluded": jb.runtime_models()}


# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    _require_demo_plane()
    from src.evals.final_eval import collect as collect_mod
    from src.evals.final_eval import deterministic as det_mod

    selected = _selected(args)
    backend = None if args.skip_judge else _judge_backend(args, required=True)
    run_id = args.run_id or man.new_run_id()
    run = man.RunDir(run_id, args.results_root)

    m = man.build_manifest(
        run_id=run_id, case_ids=[c.query_id for c in selected], subset=args.subset,
        judge_config=(_judge_config(backend) if backend else
                      {"model": None, "skipped": True,
                       "reason": "--skip-judge: no independent judge ran"}),
        collection_backend="graph:run_once", notes=args.notes)
    try:
        m = run.open_for_write(m, resume=args.resume)
    except man.RunDirError as e:
        raise SystemExit(f"refusing: {e}")

    say(f"\n=== run {run_id} ===")
    say(f"  cases    : {len(selected)} ({args.subset})")
    say(f"  models   : main={m['models']['main']} fast={m['models']['fast']}")
    say(f"  prompts  : {m['prompt_versions']}")
    say(f"  judge    : {(m.get('judge') or {}).get('model')}")
    say(f"  artifacts: {run.path}\n")

    say("-- phase 2: collection --")
    cstats = collect_mod.collect(run, selected, resume=args.resume, progress=say)
    say(f"   {cstats}\n")

    index = {c.query_id: c for c in selected}
    say("-- phase 3: deterministic --")
    dstats = det_mod.run(run, index, progress=say)
    say(f"   {dstats}\n")

    if backend is not None:
        from src.evals.final_eval import judge as judge_mod
        say("-- phase 4: independent judge --")
        jstats = judge_mod.run(run, index, backend, resume=args.resume, progress=say)
        say(f"   {jstats}\n")
    else:
        say("-- phase 4: SKIPPED (--skip-judge); judge.jsonl will be empty --\n")

    _score(run, m)
    if not args.no_complete:
        run.mark_complete(note=f"{args.subset} run via final_eval.py")
        say(f"\nrun sealed: {run.sentinel.name} written — this directory is now immutable")
    return 0


def cmd_collect(args) -> int:
    _require_demo_plane()
    from src.evals.final_eval import collect as collect_mod
    run = man.RunDir(args.run_id, args.results_root)
    if not run.file("manifest").exists():
        raise SystemExit(f"no manifest in {run.path}; start the run with `run` first")
    # Collection is raw capture: it costs model calls and is not reproducible
    # bit-for-bit. It may never be added to a sealed run.
    if run.is_complete():
        raise SystemExit(f"refusing: run {args.run_id} is complete and immutable")
    stats = collect_mod.collect(run, _selected(args), resume=args.resume, progress=say)
    say(stats)
    return 0


def cmd_deterministic(args) -> int:
    # A derived view: recomputed from responses.jsonl and the frozen gold set,
    # with no model call, so it is safe to regenerate on a sealed run.
    from src.evals.final_eval import deterministic as det_mod
    run = man.RunDir(args.run_id, args.results_root)
    index = {c.query_id: c for c in case_mod.load_cases()}
    say(det_mod.run(run, index, progress=say))
    return 0


def cmd_judge(args) -> int:
    from src.evals.final_eval import judge as judge_mod
    run = man.RunDir(args.run_id, args.results_root)
    if run.is_complete() and not args.allow_complete:
        raise SystemExit(f"refusing: run {args.run_id} is complete and immutable")
    backend = _judge_backend(args, required=True)
    index = {c.query_id: c for c in case_mod.load_cases()}
    say(judge_mod.run(run, index, backend, resume=args.resume, progress=say))
    return 0


def _score(run, m: dict) -> dict:
    """Re-derive the scorecard from this run's raw artifacts.

    summary.json, report.md and failures.jsonl are DERIVED views: they are a
    pure function of responses.jsonl, deterministic.jsonl, judge.jsonl and the
    frozen gold set. Regenerating them (after a reporting fix, say) changes no
    measurement, so it is permitted on a sealed run — and is recorded, so an
    artifact can never silently differ from the run that produced it.
    """
    say("-- phase 6/7: scorecard + failure taxonomy --")
    was_sealed = run.is_complete()
    fstats = fail_mod.build(run)
    summary = agg.build_summary(run, m)
    summary["derivation"] = {
        "generated_at": man.utc_now(),
        "regenerated_after_seal": was_sealed,
        "note": "derived from this run's raw artifacts only; no measurement was re-run",
    }
    run.write_json("summary", summary)
    run.write_text("report", agg.build_report(summary, fstats, m))
    rel = summary["reliability"]
    say(f"   hard gates: {'ALL PASS' if rel['all_gates_pass'] else 'FAILED'}")
    for name, g in rel["gates"].items():
        if not g["pass"]:
            say(f"     FAIL {name}: {g.get('offenders')}")
    say(f"   failures: {fstats['total_findings']} findings / "
        f"{fstats['cases_with_failures']} cases")
    say(f"   wrote {run.file('summary').name}, {run.file('report').name}")
    return summary


def cmd_score(args) -> int:
    run = man.RunDir(args.run_id, args.results_root)
    if not run.file("manifest").exists():
        raise SystemExit(f"no manifest in {run.path}")
    _score(run, run.read_json("manifest"))
    return 0


def cmd_show(args) -> int:
    run = man.RunDir(args.run_id, args.results_root)
    if not run.file("report").exists():
        raise SystemExit(f"no report in {run.path}; run `score` first")
    say(run.file("report").read_text())
    return 0


def cmd_list(args) -> int:
    root = Path(args.results_root or man.DEFAULT_RESULTS_ROOT)
    if not root.exists():
        say(f"no runs under {root}")
        return 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        state = "complete" if (d / man.COMPLETE_SENTINEL).exists() else "incomplete"
        n = len((d / "responses.jsonl").read_text().splitlines()) \
            if (d / "responses.jsonl").exists() else 0
        say(f"  {d.name:<40} {state:<11} {n} responses")
    return 0


def cmd_purge_evidence(args) -> int:
    """Delete the judge's evidence working file. The published artifacts never
    contained note text; this removes the only file that did."""
    run = man.RunDir(args.run_id, args.results_root)
    p = run.file("evidence_cache")
    if p.exists():
        p.unlink()
        say(f"removed {p}")
    else:
        say(f"nothing to remove at {p}")
    return 0


def cmd_calibrate(args) -> int:
    """Phase 5 — where the deterministic checks, the runtime routing and the
    independent judge disagree. Reads artifacts only; calls no model."""
    from src.evals.final_eval import calibration as cal_mod
    run = man.RunDir(args.run_id, args.results_root)
    if not run.file("manifest").exists():
        raise SystemExit(f"no manifest in {run.path}")
    say(f"-- phase 5: calibration ({run.run_id}) --")
    cal = cal_mod.run(run, run.read_json("manifest"), progress=say)
    say(f"\n   {cal['n_cases_with_disagreements']}/{cal['n_cases']} cases carry a "
        f"disagreement")
    for kind, n in (cal["by_kind"] or {}).items():
        say(f"     {kind}: {n}")
    say(f"   wrote {run.file('calibration').name}, {run.file('calibration_report').name}")
    say("\n   Adjudication is pending and is a HUMAN decision: set `adjudication` on "
        "each finding.\n   LLM-as-judge is not ground truth, and neither is strict "
        "matching on a paraphrase.")
    return 0


def cmd_compare(args) -> int:
    """Phase 10 — read-only comparison. Refuses a delta it cannot honestly compute."""
    from src.evals.final_eval import compare as cmp_mod
    run = man.RunDir(args.run_id, args.results_root)
    try:
        new = cmp_mod.load_run(run)
        old = (cmp_mod.load_run(man.RunDir(args.baseline_run_id, args.results_root))
               if args.baseline_run_id else cmp_mod.load_legacy(args.baseline))
        result = cmp_mod.compare(new, old)
    except cmp_mod.IncomparableRuns as e:
        raise SystemExit(f"refusing: {e}")
    say(cmp_mod.render(result))
    if not args.no_write:
        run.write_json("comparison", result)
        say(f"   wrote {run.file('comparison').name}")
    return 0


def cmd_api_crosscheck(args) -> int:
    """Post-evaluation HTTP contract check. Never a prerequisite for a score."""
    from src.evals.final_eval import api_crosscheck as x
    run = man.RunDir(args.run_id, args.results_root)
    if not run.file("responses").exists():
        raise SystemExit(f"no responses.jsonl in {run.path}")
    p = x.probe(args.base_url)
    if not p.get("reachable"):
        raise SystemExit(
            f"refusing: {args.base_url} is not reachable ({p.get('error') or p.get('http')}). "
            f"The API cross-check is a deployed-surface check and is skipped when there "
            f"is no deployment; it is never required for the answer-quality evaluation.")
    say(f"-- api contract cross-check against {args.base_url} --")
    rep = x.run(run, args.base_url, progress=say, limit=args.limit)
    say(f"\n   {rep['n_contract_ok']}/{rep['n_cases']} contract-clean, "
        f"{rep['n_divergent']} divergent, {len(rep['transport_errors'])} transport error(s)")
    say(f"   wrote {run.file('api_crosscheck').name}")
    return 0 if not rep["n_divergent"] and not rep["transport_errors"] else 1


def cmd_doctor(args) -> int:
    """RunPod preflight. Diagnostic only — changes nothing, starts nothing."""
    from src.evals.final_eval import doctor as doc
    rep = doc.run_checks(profile=args.profile, judge_model=args.judge_model,
                         judge_host=args.judge_host, results_root=args.results_root,
                         run_id=args.run_id)
    if args.json:
        say(json.dumps(rep.to_dict(), indent=2))
    else:
        say(doc.render(rep))
    return rep.exit_code


def cmd_cases(args) -> int:
    cs = case_mod.load_cases()
    fp = case_mod.dataset_fingerprint()
    say(f"{fp['n_cases']} cases  sha256={fp['sha256'][:16]}  "
        f"version={fp['dataset_version']}  manifest_match={fp['manifest_sha256_matches']}")
    for c in case_mod.select(cs, ids=args.ids, subset=args.subset):
        say(f"  {c.query_id:<10} subj={c.subject_id} {c.category:<14} {c.answer_type:<20} "
            f"temporal={c.temporal:<7} facts={len(c.expected_facts)} "
            f"(struct {c.n_structured_facts}) min={c.min_facts} "
            f"{'ABSTAIN ' if c.expects_abstention else ''}"
            f"{'AMBIGUOUS' if c.expects_ambiguity else ''}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="final_eval.py",
        description="Lumen final answer-level evaluation framework "
                    f"({EVALUATOR_VERSION}). Artifacts are immutable once sealed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages can be run separately (collect / deterministic / judge / score) "
               "to resume an interrupted run without repeating model calls.")
    ap.add_argument("--results-root", default=None,
                    help=f"default: {man.DEFAULT_RESULTS_ROOT}")
    sub = ap.add_subparsers(dest="command", required=True)

    def _cases_args(p):
        p.add_argument("--subset", default="all", choices=["all", "legacy15", "smoke"],
                       help="all = the full 40-case set; legacy15 = demo_q01-demo_q15; "
                            "smoke = the 3-case triple")
        p.add_argument("--ids", nargs="+", default=None,
                       help="explicit query ids, in order; overrides --subset")

    def _judge_args(p):
        p.add_argument("--judge-model", default=None,
                       help="independent judge model; must NOT be a runtime tier "
                            "(default: $LUMEN_JUDGE_MODEL)")
        p.add_argument("--judge-host", default=None, help="default: $LUMEN_JUDGE_HOST")

    p = sub.add_parser("run", help="collect, evaluate, judge and score in one run")
    _cases_args(p); _judge_args(p)
    p.add_argument("--run-id", default=None, help="default: <UTC timestamp>-<git sha>")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run_id")
    p.add_argument("--skip-judge", action="store_true",
                   help="deterministic only; judge.jsonl stays empty and is reported as such")
    p.add_argument("--no-complete", action="store_true",
                   help="do not seal the run directory when finished")
    p.add_argument("--notes", default="", help="free text recorded in the manifest")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("collect", help="phase 2 only — run cases through the graph")
    _cases_args(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("deterministic", help="phase 3 only — re-evaluate collected responses")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_deterministic)

    p = sub.add_parser("judge", help="phase 4 only — independent offline judge")
    _judge_args(p)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-complete", action="store_true",
                   help="permit judging a sealed run (refused by default)")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("score", help="phases 6/7 — scorecard, failure taxonomy, report")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("show", help="print an existing run's report.md")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("list", help="list runs and their state")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("purge-evidence", help="delete a run's evidence working file")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_purge_evidence)

    p = sub.add_parser("calibrate", help="phase 5 — disagreements between the deterministic "
                                        "checks, the runtime routing and the judge "
                                        "(reads artifacts only; calls no model)")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("compare", help="phase 10 — read-only comparison against another run "
                                       "or a cloud_eval performance.json")
    p.add_argument("--run-id", required=True, help="the newer final_eval run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline-run-id", help="another final_eval run to compare against")
    g.add_argument("--baseline", help="path to a cloud_eval performance.json "
                                      "(operational metrics only)")
    p.add_argument("--no-write", action="store_true", help="print without writing comparison.json")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("api-crosscheck", help="post-evaluation HTTP contract check against a "
                                              "deployed /ask (never required for a score)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_api_crosscheck)

    p = sub.add_parser("doctor", help="preflight: verify every prerequisite before spending "
                                      "GPU time. Diagnostic only; runs no case")
    _judge_args(p)
    p.add_argument("--profile", default="final", choices=["final", "local"],
                   help="final = every prerequisite is required (default); local = demote "
                        "machine-dependent checks (models, GPU, database) to advisory")
    p.add_argument("--run-id", default=None,
                   help="also check that this run id is still available")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("cases", help="print the evaluation set and its fingerprint")
    _cases_args(p)
    p.set_defaults(func=cmd_cases)
    return ap


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
