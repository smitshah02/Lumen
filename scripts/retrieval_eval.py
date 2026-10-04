"""Development retrieval benchmark: build it once, then score four retrieval stages.

    ./scripts/lumen research retrieval-eval build     # freeze the benchmark (once)
    ./scripts/lumen research retrieval-eval run       # lexical / vector / hybrid / hybrid+BGE

DEVELOPMENT benchmark: patients come from the indexed research cohort. It is
not external validation, and it refuses to run against the holdout database.

How relevance is decided — without the retriever and without a model:
  * A discharge summary has standard sections ("Discharge Medications:",
    "Brief Hospital Course:", ...). Each question asks for the content of one
    section. A chunk is relevant when it lies inside that section of the
    patient's own note, located by character offset in the note text.
  * Imaging questions use radiology notes whose exam header names the chest.
  * Temporal questions ("most recent", "earliest") grade the asked-for note 2
    and the same section from any other date 1.
Labels are frozen in the benchmark file before any retrieval is run.

The benchmark and per-question results contain subject and chunk ids and stay
under ~/Lumen_local_results/retrieval_eval. Only aggregate metrics are printed.
No note text is written anywhere.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src import storage  # noqa: E402
from src.evals import retrieval_metrics as rm  # noqa: E402

OUT_DIR = Path.home() / "Lumen_local_results" / "retrieval_eval"
BENCHMARK = OUT_DIR / "benchmark.json"
SEED = 20261005
MIN_TOKENS = 40           # the retriever never returns shorter chunks, so they are not gold either
TOP_K = 50                # deep enough to report candidate recall at 10/20/50

# Standard MIMIC-IV discharge-summary headers, in the order they appear.
HEADERS = {
    "allergies": r"allergies", "chief_complaint": r"chief complaint",
    "procedure": r"major surgical or invasive procedure", "hpi": r"history of present illness",
    "pmh": r"past medical history", "social": r"social history", "family": r"family history",
    "exam": r"physical exam", "results": r"pertinent results", "course": r"brief hospital course",
    "adm_meds": r"medications on admission", "dc_meds": r"discharge medications",
    "dc_disposition": r"discharge disposition", "dc_dx": r"discharge diagnos[ie]s",
    "dc_condition": r"discharge condition", "dc_instr": r"discharge instructions",
    "followup": r"follow\s*-?\s*up instructions",
}
HEADER_RE = re.compile(r"(?im)^[ \t]*(?:" + "|".join(f"(?P<{k}>{v})" for k, v in HEADERS.items()) + r")[ \t]*:")

# (id, kind, question, section graded 2, sections graded 1, temporal target)
TEMPLATES = [
    ("medications", "medication", "What medications was the patient discharged on?", ["dc_meds"], ["adm_meds"], None),
    ("diagnoses", "diagnosis", "What were the patient's discharge diagnoses?", ["dc_dx"], [], None),
    ("procedures", "procedure", "What major surgical or invasive procedures did the patient undergo?", ["procedure"], [], None),
    ("allergies", "factual", "What allergies does the patient have?", ["allergies"], [], None),
    ("history", "factual", "What is the patient's past medical history?", ["pmh"], [], None),
    ("social", "factual", "What is the patient's social history?", ["social"], [], None),
    ("lab_results", "lab_note_fact", "What were the pertinent laboratory results during the hospital stays?", ["results"], [], None),
    ("hospital_course", "longitudinal", "Summarize the patient's hospital course across admissions.", ["course"], ["dc_dx"], None),
    ("admission_reason", "multi_note", "Why was the patient admitted to the hospital?", ["chief_complaint", "hpi"], [], None),
    ("followup", "factual", "What follow-up instructions was the patient given at discharge?", ["dc_instr", "followup"], [], None),
    ("chest_imaging", "imaging", "What did the chest imaging show?", ["chest_radiology"], [], None),
    ("latest_medications", "temporal", "What medications was the patient discharged on at the most recent admission?", ["dc_meds"], [], "latest"),
    ("latest_diagnosis", "temporal", "What was the discharge diagnosis at the most recent admission?", ["dc_dx"], [], "latest"),
    ("latest_course", "temporal", "What happened during the most recent hospital stay?", ["course"], [], "latest"),
    ("latest_chest_imaging", "temporal", "What did the most recent chest imaging show?", ["chest_radiology"], [], "latest"),
    ("earliest_diagnosis", "temporal", "What was the earliest documented discharge diagnosis?", ["dc_dx"], [], "earliest"),
    ("earliest_medications", "temporal", "What were the earliest recorded discharge medications?", ["dc_meds"], [], "earliest"),
]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
WINDOW = 6                 # consecutive words used to find a chunk inside its note


def section_spans(note_text: str) -> list[tuple[str, int, int]]:
    """(section, start, end) character spans, from line-anchored standard headers."""
    matches = list(HEADER_RE.finditer(note_text))
    return [(m.lastgroup, m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(note_text))
            for i, m in enumerate(matches)]


def chunk_sections(note_text: str, chunks: list[tuple[int, str]]) -> dict[int, list[str]]:
    """Which sections each chunk carries, by locating the chunk in the note.

    Chunk text is not a verbatim substring of the stored note (whitespace and
    punctuation differ), so a chunk is located by a run of six consecutive
    words. Its word extent is then compared with the section extents. A chunk
    carries a section when at least half of the chunk lies in the section, or
    at least half of the section lies in the chunk. The second clause matters:
    the chunker merges short sections (allergies, discharge diagnosis) into a
    neighbour, so a midpoint rule left the chunk that holds the whole section
    unlabelled. A chunk that cannot be located is left unlabelled, never guessed."""
    import bisect
    tokens = [(m.group().lower(), m.start()) for m in _WORD_RE.finditer(note_text)]
    words = [w for w, _ in tokens]
    starts = [pos for _, pos in tokens]
    index: dict[tuple, list[int]] = {}
    for i in range(len(words) - WINDOW + 1):
        index.setdefault(tuple(words[i:i + WINDOW]), []).append(i)
    spans = [(name, bisect.bisect_left(starts, start), bisect.bisect_left(starts, end))
             for name, start, end in section_spans(note_text)]
    out, cursor = {}, 0
    for chunk_id, chunk_text in chunks:                      # in chunk_index order
        cw = [w.lower() for w in _WORD_RE.findall(chunk_text or "")]
        position = None
        for offset in range(0, max(1, len(cw) - WINDOW + 1), 3):
            hits = index.get(tuple(cw[offset:offset + WINDOW]))
            if hits:                                         # prefer the occurrence at or after the previous chunk
                later = [h for h in hits if h - offset >= cursor - 150]
                position = (later[0] if later else hits[0]) - offset
                break
        if position is None:
            continue
        cursor = max(cursor, position)
        found = []
        for name, start, end in spans:
            overlap = min(end, position + len(cw)) - max(start, position)
            if overlap > 0 and (2 * overlap >= len(cw) or 2 * overlap >= end - start):
                found.append((overlap, name))
        if found:
            out[chunk_id] = [name for _, name in sorted(found, reverse=True)]
    return out


def build(patients: int, per_patient: int, seed: int) -> int:
    if BENCHMARK.exists():
        print(f"refusing: {BENCHMARK} already exists. The benchmark is frozen once built.", file=sys.stderr)
        return 2
    rng = random.Random(seed)
    questions, located, total_chunks = [], 0, 0
    section_chunks: dict[str, int] = {}
    excluded: dict[str, int] = {}
    with storage.engine.connect() as c:
        pool = list(c.execute(text("""
            SELECT subject_id FROM clinical_notes
            GROUP BY subject_id
            HAVING count(*) FILTER (WHERE note_type = 'discharge') >= 4
               AND count(*) FILTER (WHERE note_type = 'radiology') >= 5
            ORDER BY subject_id""")).scalars())
        chosen = sorted(rng.sample(pool, patients))
        for p_index, sid in enumerate(chosen):
            notes = c.execute(text("""
                -- the note as written: de-identification redacts the "Brief Hospital Course" header
                SELECT note_id, note_type, charttime, COALESCE(text_original, text_deid) FROM clinical_notes
                WHERE subject_id = :s ORDER BY charttime, note_id"""), {"s": sid}).fetchall()
            chunks = c.execute(text("""
                SELECT note_id, chunk_id, chunk_text, token_count FROM note_chunks
                WHERE subject_id = :s ORDER BY note_id, chunk_index"""), {"s": sid}).fetchall()
            by_note: dict[int, list] = {}
            for note_id, chunk_id, chunk_text, tokens in chunks:
                by_note.setdefault(note_id, []).append((chunk_id, chunk_text, tokens or 0))
            # chunk -> (sections, note_id, charttime)
            labelled: dict[int, tuple[list[str], int, str]] = {}
            for note_id, note_type, charttime, body in notes:
                mine = by_note.get(note_id, [])
                total_chunks += len(mine)
                if note_type == "discharge":
                    sections = chunk_sections(body or "", [(cid, t) for cid, t, _ in mine])
                    located += len(sections)
                    for cid, _, tokens in mine:
                        if cid in sections and tokens >= MIN_TOKENS:
                            labelled[cid] = (sections[cid], note_id, str(charttime))
                elif re.search(r"chest|cxr", (body or "")[:300], re.I):
                    located += len(mine)
                    for cid, _, tokens in mine:
                        if tokens >= MIN_TOKENS:
                            labelled[cid] = (["chest_radiology"], note_id, str(charttime))
            for secs, _, _ in labelled.values():
                for sec in secs:
                    section_chunks[sec] = section_chunks.get(sec, 0) + 1
            picks = [TEMPLATES[(p_index * per_patient + j) % len(TEMPLATES)] for j in range(per_patient)]
            for tid, kind, question, strong, weak, temporal in picks:
                gold = {cid: 2 for cid, (secs, _, _) in labelled.items() if set(secs) & set(strong)}
                gold.update({cid: 1 for cid, (secs, _, _) in labelled.items() if set(secs) & set(weak) and cid not in gold})
                if temporal and gold:
                    times = sorted({labelled[cid][2] for cid in gold})
                    target = times[-1] if temporal == "latest" else times[0]
                    if len(times) < 2:
                        excluded[tid] = excluded.get(tid, 0) + 1
                        continue                              # no distractor: not a temporal question
                    gold = {cid: (2 if labelled[cid][2] == target else 1) for cid in gold}
                if not any(g == 2 for g in gold.values()):
                    excluded[tid] = excluded.get(tid, 0) + 1
                    continue                                  # no defensible gold evidence: excluded
                questions.append({"id": f"q{len(questions) + 1:02d}", "template": tid, "kind": kind,
                                  "temporal": temporal, "subject_id": sid, "query": question,
                                  "relevance": {str(cid): g for cid, g in sorted(gold.items())},
                                  "gold_notes": len({labelled[cid][1] for cid in gold})})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.chmod(0o700)
    BENCHMARK.write_text(json.dumps({
        "name": "development retrieval benchmark", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
        "database": storage.engine.url.database, "patients": len(chosen),
        "label_source": "section overlap (half the chunk in the section, or half the section in the chunk) "
                        "in the patient's own notes; no retriever, no model",
        "grades": "2 = target section (and target date for temporal questions); 1 = related section or other date",
        "questions": questions}, indent=1))
    BENCHMARK.chmod(0o600)
    kinds: dict[str, int] = {}
    for q in questions:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    sizes = sorted(len(q["relevance"]) for q in questions)
    print(f"patients {len(chosen)} (pool {len(pool)}, seed {seed}) | questions {len(questions)} of {patients * per_patient} planned")
    print(f"by kind: {dict(sorted(kinds.items()))} | temporal: {sum(1 for q in questions if q['temporal'])}")
    print(f"chunks located in their note: {located}/{total_chunks} | gold chunks per question: "
          f"min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"labelled chunks by section: {dict(sorted(section_chunks.items()))}")
    print(f"excluded (no defensible gold): {dict(sorted(excluded.items()))}")
    print(f"benchmark frozen at {BENCHMARK}")
    return 0


@contextmanager
def _arm(module, retriever, config: str):
    """Run production search() with one stage switched off. Nothing is re-implemented:
    lexical/vector disable the other arm, hybrid skips the reranker, hybrid_bge is untouched."""
    saved = (module.bm25_search, module.vector_search, retriever.reranker, retriever.rerank_candidates)
    try:
        if config == "lexical":
            module.vector_search = lambda **kw: []
        if config == "vector":
            module.bm25_search = lambda **kw: []
        if config != "hybrid_bge":
            retriever.reranker = None
            # same order, longer list: lets Hit@20/50 see the candidates the reranker would be given
            retriever.rerank_candidates = max(retriever.rerank_candidates, TOP_K)
        yield
    finally:
        module.bm25_search, module.vector_search, retriever.reranker, retriever.rerank_candidates = saved


CONFIGS = (("lexical", "Lexical only"), ("vector", "Vector only"), ("hybrid", "Hybrid RRF"), ("hybrid_bge", "Hybrid RRF + BGE"))


def run(configs=None) -> int:
    global CONFIGS
    if configs:
        CONFIGS = tuple(c for c in CONFIGS if c[0] in configs)
    if not BENCHMARK.exists():
        print("no benchmark; run `retrieval-eval build` first", file=sys.stderr)
        return 2
    bench = json.loads(BENCHMARK.read_text())
    if storage.engine.url.database != bench["database"]:
        print(f"refusing: benchmark was built on {bench['database']!r}, this process is on "
              f"{storage.engine.url.database!r}", file=sys.stderr)
        return 2
    import logging
    logging.disable(logging.INFO)
    import src.retrieval.hybrid_retriever_v2 as H
    retriever = H.HybridRetriever()
    questions = bench["questions"]
    per_config: dict[str, list[dict]] = {}
    leaks = 0
    for config, _ in CONFIGS:
        rows = []
        for q in questions:
            relevance = {int(k): v for k, v in q["relevance"].items()}
            t0 = time.perf_counter()
            with _arm(H, retriever, config):
                results = retriever.search(query=q["query"], subject_id=q["subject_id"], temporal_filter="auto", top_k=TOP_K)
            ranked = [r.chunk_id for r in results]
            leaks += sum(1 for r in results if r.subject_id != q["subject_id"])
            row = {"id": q["id"], "kind": q["kind"], "temporal": q["temporal"], "ranked": ranked,
                   "ms": round((time.perf_counter() - t0) * 1000, 1), **rm.score(ranked, relevance)}
            if q["temporal"]:
                row.update(rm.temporal_score(ranked, relevance))
            rows.append(row)
        per_config[config] = rows

    summary = {"benchmark": bench["name"], "seed": bench["seed"], "patients": bench["patients"],
               "questions": len(questions), "temporal_questions": sum(1 for q in questions if q["temporal"]),
               "cross_patient_results": leaks,
               "overall": {c: rm.macro(rows) for c, rows in per_config.items()},
               "by_kind": {c: {k: rm.macro([r for r in rows if r["kind"] == k])
                               for k in sorted({r["kind"] for r in rows})} for c, rows in per_config.items()},
               "temporal": {c: rm.temporal_macro([r for r in rows if r["temporal"]]) for c, rows in per_config.items()},
               "temporal_by_mode": {c: {m: rm.temporal_macro([r for r in rows if r["temporal"] == m])
                                        for m in ("latest", "earliest")} for c, rows in per_config.items()},
               "median_ms": {c: sorted(r["ms"] for r in rows)[len(rows) // 2] for c, rows in per_config.items()}}
    both = "hybrid" in per_config and "hybrid_bge" in per_config
    summary["reranker_lift"] = rm.lift(summary["overall"]["hybrid"], summary["overall"]["hybrid_bge"]) if both else {}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"results-{stamp}.json").write_text(json.dumps({"summary": summary, "per_question": per_config}, indent=1))
    if both:                                   # a partial run never replaces the full aggregate
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"DEVELOPMENT RETRIEVAL BENCHMARK  questions {summary['questions']}  patients {summary['patients']}  "
          f"(small; indexed cohort; not external validation)\n")
    print(f"{'Method':18s} {'P@5':>6s} {'R@5':>6s} {'nDCG@5':>7s} {'MRR':>6s} {'Hit@5':>6s} {'R@10':>6s} {'nDCG@10':>8s} {'Hit@10':>7s} {'median':>8s}")
    for config, label in CONFIGS:
        m = summary["overall"][config]
        print(f"{label:18s} {m['p@5']:6.3f} {m['r@5']:6.3f} {m['ndcg@5']:7.3f} {m['mrr']:6.3f} {m['hit@5']:6.3f} "
              f"{m['r@10']:6.3f} {m['ndcg@10']:8.3f} {m['hit@10']:7.3f} {summary['median_ms'][config]:6.0f}ms")
    print(f"\n{'Candidate depth':18s} {'Hit@10':>7s} {'Hit@20':>7s} {'Hit@50':>7s} {'R@10':>7s} {'R@20':>7s} {'R@50':>7s}")
    for config, label in CONFIGS:
        m = summary["overall"][config]
        print(f"{label:18s} {m['hit@10']:7.3f} {m['hit@20']:7.3f} {m['hit@50']:7.3f} {m['r@10']:7.3f} {m['r@20']:7.3f} {m['r@50']:7.3f}")
    print("\nReranker lift (Hybrid RRF -> Hybrid RRF + BGE):")
    for metric, v in summary["reranker_lift"].items():
        rel = f"{v['relative_pct']:+.1f}%" if v["relative_pct"] is not None else "n/a"
        print(f"  {metric:7s} {v['base']:.3f} -> {v['improved']:.3f}   absolute {v['absolute']:+.3f}   relative {rel}")
    print(f"\nTemporal questions ({summary['temporal_questions']}): target Hit@5 / target MRR / target ranked above other dates")
    for config, label in CONFIGS:
        t = summary["temporal"][config]
        print(f"  {label:18s} {t['target_hit@5']:.3f} / {t['target_rr']:.3f} / {t['ordered_correctly']:.3f}")
    print(f"\ncross-patient results returned: {leaks}")
    print(f"raw results: {OUT_DIR}/results-{stamp}.json   aggregate: {OUT_DIR}/summary.json")
    return 1 if leaks else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--patients", type=int, default=12)
    b.add_argument("--per-patient", type=int, default=3)
    b.add_argument("--seed", type=int, default=SEED)
    r = sub.add_parser("run")
    r.add_argument("--configs", help="comma-separated subset of: " + ",".join(c for c, _ in CONFIGS))
    args = p.parse_args(argv)
    if storage.DATA_PLANE != "research":
        print(f"refusing: the retrieval benchmark runs on the research plane, not {storage.DATA_PLANE!r}", file=sys.stderr)
        return 2
    if storage.engine.url.database == "lumen_holdout":
        print("refusing: the frozen holdout is not used for development benchmarks", file=sys.stderr)
        return 2
    return build(args.patients, args.per_patient, args.seed) if args.command == "build" else run(args.configs.split(",") if args.configs else None)


if __name__ == "__main__":
    raise SystemExit(main())
