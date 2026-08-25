# Lumen — Project Status & Repo Audit Spec

> Reconstructed from source. Use this as the reference to audit the real
> working tree: confirm what exists, flag what's missing or drifted.

---

## 1. What Lumen is

A clinical retrieval-augmented generation system over **MIMIC-IV** (structured
tables + free-text notes), built to answer clinician-style questions against a
single patient's record.

The differentiator is **temporal correctness**. Generic RAG returns *a*
creatinine; Lumen returns the *most recent* creatinine, or the creatinine
*trend*, or the creatinine *from this admission* — and proves it with a metric
separate from topical relevance.

**Stack:** Postgres 16 + pgvector · MedCPT dual-encoder (768-d) · BGE
cross-encoder reranker · Presidio + spaCy · Groq (Llama-3.3-70B) as eval judge ·
Apple Silicon MPS with CPU fallback.

---

## 2. Architecture — layer by layer

```
MIMIC-IV CSVs
     |
 [1] DE-IDENTIFICATION      Presidio + spaCy + custom clinical recognizers
     |
 [2] STORAGE                Postgres + pgvector, 9 tables, HNSW + GIN indexes
     |
 [3] CHUNK + EMBED          section-aware chunker -> MedCPT article encoder
     |
 [4] HYBRID RETRIEVAL v2    BM25 + vector -> RRF -> context -> temporal -> rerank
     |
 [5] EVALUATION             golden set + LLM judge + temporal lift harness
     |
 [6] GENERATION             <-- NOT BUILT
     |
 [7] API / UI               <-- NOT BUILT
```

### Layer 1 — De-identification `src/deid/`

HIPAA Safe Harbor redaction: person names, dates, locations, phone/fax, MRN,
SSN, email, ages over 89, device serials, account/insurance numbers.

- `pipeline.py` — `DeidentificationPipeline` / `DeidResult`, score threshold
  0.35, returns redacted text + entity list + counts.
- `sample_notes.py` — 5 synthetic notes with a manifest of expected PHI at
  known positions, so recall is measurable before touching real data.
- `mimic_adapter.py` — `MIMICNoteLoader` for discharge summaries + radiology.
- `test_pipeline.py` — recall harness via substring disappearance.

### Layer 2 — Storage `src/storage/`

Tables: `patients`, `admissions`, `diagnoses_icd`, `labevents`,
`prescriptions`, `procedures_icd`, `clinical_notes`, `note_chunks`,
`guideline_chunks`, `ingestion_log`.

- `schema.py` — DDL, `vector(768)` columns, HNSW (`m=16, ef_construction=64`),
  GIN on generated `tsvector`, `create_schema()` / `drop_all_tables()` /
  `get_table_counts()`.
- `ingest.py` — orchestrator with `--patients`, `--skip-labs`, `--skip-notes`,
  `--skip-deid`, `min_admissions`, `min_age` filters. Default 5,000-patient
  subset. De-ID runs inline on notes; logs to `ingestion_log`.
- `migrate_chunk_fts.py` — **key correctness fix.** BM25 originally ranked on
  note-level `tsvector`, so every chunk of one note shared an identical score.
  Adds a chunk-level `GENERATED ALWAYS ... STORED` tsvector so chunks compete on
  their own text. Idempotent, no re-embedding.

### Layer 3 — Chunking & indexing `src/retrieval/`

- `chunker.py` — splits on clinical section headers first (HPI, PMH, LABS,
  MEDICATIONS...), then sentence-windowed overlap; preserves section context in
  each chunk; sized for MedCPT's 512-token ceiling.
- `embeddings.py` — `MedCPTEmbedder`, separate `embed_documents()` (article
  encoder) and `embed_query()` (query encoder), batched, MPS-aware.
- `index_notes.py` — notes -> chunks -> vectors -> `note_chunks`.
  `--limit`, `--reindex`.
- `index_guidelines.py` — guideline PDFs (pypdf) -> section-preserving chunks ->
  `guideline_chunks`. `--reindex`, `--dir`.

### Layer 4 — Hybrid retriever `src/retrieval/hybrid_retriever_v2.py`

`HybridRetriever.search(query, subject_id, hadm_id, note_type,
temporal_filter="auto", top_k)`

Nine stages:
1. `expand_query` — plain language -> clinical synonyms ("swollen legs" ->
   edema, lower extremity, fluid overload). Fixes the BM25 vocabulary gap.
2. `bm25_search` — Postgres FTS, expanded, AND + OR combo, `min_tokens=40`.
3. `vector_search` — pgvector cosine over MedCPT embeddings.
4. Min-token quality filter — drops 1–2 sentence radiology indications.
5. `reciprocal_rank_fusion` — weights `bm25=1.0`, `vector=1.2`, plus an
   **overlap bonus (0.5)** rewarding chunks found by both arms.
6. `expand_context` / `fetch_adjacent_chunks` — window=1, capped at 600 tokens,
   so the reranker scores a full clinical picture rather than a fragment.
7. `deduplicate_by_note` — max 2 chunks per note.
8. `apply_temporal_filter` — see below.
9. `BGEReranker` — cross-encoder over the **assembled context**, not raw chunks.

Tunable knobs on `__init__`: `use_reranker`, `use_query_expansion`,
`use_context_window`, `bm25_top_n=60`, `vector_top_n=60`,
`rerank_candidates=40`, `rerank_top_k=10`, `min_chunk_tokens=40`,
`context_window=1`, `max_per_note=2`.

**The temporal logic (most recent significant work).** MIMIC-IV shifts every
patient's dates into 2100–2200 with a *per-patient* offset. Absolute dates are
meaningless across patients, but intervals *within* one patient are real.
Therefore recency anchors to each subject's **own** latest retrieved record —
never a global wall-clock.

- `detect_temporal_mode(query)` — conservative regex intent detection,
  precedence `latest > trend > recent`, defaults to `"all"`.
- Modes: `all` (no-op) · `recent` (drop beyond `recency_days=365` from the
  subject's anchor, then boost) · `latest` (recency boost, keep all) ·
  `trend` (chronological ascending, undated records sink last).
- Recency is a **half-life decay** (`halflife_days=180`) applied **additively**
  to `rrf_score` — additive on purpose, since multiplicative boosting hits the
  "min-maxed 0 stays 0" trap.
- Undated records keep relevance but get no recency signal, and are never
  dropped by `recent` (can't prove they're old).
- `reference_times` hook lets a future patient-context agent supply the true
  latest encounter instead of inferring from the result set.

### Layer 5 — Evaluation `src/evals/`

- `golden_dataset.py` — **28 queries**, 6 categories: `medications`, `labs`,
  `diagnosis`, `imaging`, `plain_language`, `sections`. Each carries
  `relevance_criteria` (OR-groups of keywords), `irrelevance_signals`,
  `min_relevant`, and a note on what it tests. Criteria-based rather than
  hardcoded chunk_ids, so it survives re-indexing.
- `llm_judge.py` — replaces the keyword judge, whose circularity inflated and
  saturated earlier scores. 0–3 graded scale, temperature 0, disk cache keyed on
  `(prompt_version, model, query, chunk_text)`, bounded-concurrency
  `ThreadPoolExecutor` with in-batch dedup, retries with exponential backoff +
  jitter, never raises. Unjudgeable chunks default to 0 and carry an error flag
  so failures are visible. `build_pooled_relevance()` does TREC-style pooling:
  judge the union of all configs once, score every config against that shared
  relevant-set. Graded `ndcg_at_k_graded` + `recall_at_k_pooled`. Ships with a
  mock-LLM self-test including a JSON-mangling torture suite.
- `eval_retrieval.py` — Precision@K, Recall@K, MRR, nDCG@K across 5 configs:
  BM25-only, vector-only, RRF-only, RRF+BGE, RRF+MedCPT cross-encoder.
  `--limit`, `--config`, `--export`.
- `eval_temporal.py` — the differentiator's proof. Splits *topical relevance*
  (LLM judge) from *temporal correctness* (charttime math). 6 cases across three
  assertions: `latest` -> hit@1 on newest, `trend` -> monotonicity, `window` ->
  fraction from the latest admission. Runs each query twice (temporal ON vs
  `"all"`) and reports the **lift**. Patient-scoped with auto-selection of
  longitudinally rich patients. `--selftest` validates the metric math with
  synthetic 2150-style dates and hard assertions, no DB or network.

### Layer 6+ — NOT BUILT
No answer synthesis over retrieved context. No citation/grounding layer. No
agent orchestration. No API. No UI. No guideline retrieval path.

---

## 3. Expected repo tree

```
Lumen/
├── .venv/
├── .env                          # GROQ_API_KEY, DB credentials
├── docker-compose.yml            # Postgres + pgvector
├── requirements.txt
├── README.md
│
├── data/
│   ├── mimiciv/hosp/             # patients, admissions, diagnoses_icd,
│   │                             #   labevents, prescriptions, procedures_icd,
│   │                             #   d_labitems, d_icd_diagnoses  (.csv.gz)
│   ├── mimic-iv-note/note/       # discharge.csv.gz, radiology.csv.gz
│   └── guidelines/               # ADA 2026, GOLD COPD 2025, AHA/ACC CCD 2023
│
├── models/
│   ├── medcpt-query/
│   ├── medcpt-article/
│   ├── medcpt-cross-encoder/
│   └── bge-reranker/
│
├── .overlap_cache/               # written by patient_overlap.py
│
└── src/
    ├── __init__.py
    ├── deid/
    │   ├── __init__.py
    │   ├── pipeline.py
    │   ├── sample_notes.py
    │   ├── mimic_adapter.py
    │   └── test_pipeline.py
    ├── storage/
    │   ├── __init__.py           # MUST export: engine, execute_sql, check_connection
    │   ├── schema.py
    │   ├── ingest.py
    │   └── migrate_chunk_fts.py
    ├── retrieval/
    │   ├── __init__.py
    │   ├── chunker.py
    │   ├── embeddings.py
    │   ├── hybrid_retriever_v2.py
    │   ├── index_notes.py
    │   ├── index_guidelines.py
    │   └── test_retriever_v3.py
    ├── evals/
    │   ├── __init__.py
    │   ├── golden_dataset.py
    │   ├── llm_judge.py
    │   ├── eval_retrieval.py
    │   ├── eval_temporal.py
    │   └── test_judge_live.py
    └── scripts/                  # or tools/ — location unconfirmed
        ├── export_patient_flow.py
        ├── patient_overlap.py
        └── temporal_fix.py       # scratch validation; may sit at repo root
```

---

## 4. Gaps and drift to verify

Ordered by severity. Each is a concrete thing to check in the working tree.

1. **`src/storage/__init__.py` — existence and exports.** Every layer does
   `from src.storage import engine / execute_sql / check_connection`. It was not
   in the reviewed file set. If it's missing or under-exports, nothing imports.
2. **`patient_overlap.py` — missing.** `export_patient_flow.py` consumes
   `.overlap_cache/` written by it. Confirm it exists and the cache is populated.
3. **`test_retriever_v3.py` module-name drift.** Its docstring says to run
   `python -m src.retrieval.test_retriever`. Filename and documented module
   disagree — one of them is wrong.
4. **`temporal_fix.py` is duplicated logic.** `_parse_charttime`,
   `_TEMPORAL_PATTERNS`, `detect_temporal_mode`, and `apply_temporal_filter` now
   exist in both it and `hybrid_retriever_v2.py`. Diff them; they will silently
   drift. Either delete it or import the real ones.
5. **Guideline retrieval is unreachable.** `guideline_chunks` is populated by
   `index_guidelines.py`, but neither `hybrid_retriever_v2.py` nor
   `eval_retrieval.py` mentions it. Indexed data with no query path.
6. **MedCPT cross-encoder config is mislabeled.** `eval_retrieval.py:438`
   instantiates `BGEReranker(model_path=MEDCPT_RERANKER_PATH)` — a BGE class
   pointed at MedCPT weights. Verify tokenizer/head compatibility or the fifth
   eval config is measuring something other than what it claims.
7. **`__init__.py` coverage.** Confirm every `src/*` package has one, or the
   `python -m` invocations in the docstrings fail.
8. **Post-migration reindex.** `migrate_chunk_fts.py` changed how BM25 ranks.
   Confirm `eval_retrieval.py` results were regenerated after it ran, otherwise
   any saved numbers predate the fix.
9. **Model weights present.** All four `models/` subdirs must be populated;
   paths are hardcoded to `Path.home() / "Lumen" / "models"`.
10. **Stray artifacts.** Look for `results.json`, judge cache files,
    `.pyc`/`__pycache__`, and any `hybrid_retriever.py` / `hybrid_retriever_v1.py`
    or `test_retriever_v1/v2.py` left behind by versioning.

---

## 5. Completion estimate

| Layer | State |
|---|---|
| 1 · De-identification | Complete, recall-tested |
| 2 · Storage + ingest | Complete, incl. chunk-FTS migration |
| 3 · Chunk / embed / index | Complete (notes + guidelines) |
| 4 · Hybrid retrieval v2 | Complete, 9 stages, temporal merged in |
| 5 · Evaluation | Complete (IR metrics + LLM judge + temporal lift) |
| 6 · Answer generation | Not started |
| 7 · API / UI | Not started |
| — · Guideline retrieval path | Indexed but not wired |

**Retrieval-and-evaluation half of the system: essentially done.**
**Product half — generation, grounding, serving: not begun.**

---

## 6. Audit instructions

Walk the tree. For each file in §3 report `OK` / `MISSING` / `EXTRA`. For each
gap in §4 report the actual finding with file:line evidence. Do not fix
anything on this pass — produce the report first.
