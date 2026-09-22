# Lumen Verification Audit — running log

Started: 2026-09-06. Read-only through Phase 9. No source edits.

## Phase 0 — Inventory (in progress)

### Repo shape
- No `pyproject.toml`, no `Makefile`, no `Dockerfile`, no `alembic/`, no CI config found at depth<=2.
- Only config surfaces: `requirements.txt`, `docker-compose.yml`, `.env` (gitignored, no `.env.example` committed).
- `scripts/bootstrap_pod.sh` is the only script.
- `infra/` is empty.

### FIRST BIG DISCREPANCY (Phase 0)
requirements.txt header says the LLM stack is **Ollama + qwen2.5:14b over plain HTTP**,
and the `openai`/Groq dependency is COMMENTED OUT. The audit brief expects Groq
`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`. `.env` still carries GROQ_API_KEY.
Needs resolution in Phase 5.

## Phase 1 — Deps (done)
- Python **3.14.5rc1** in `.venv`. requirements.txt claims "dev env: 3.14.5" (release). Close enough; note it's an rc.
- `pip check` → "No broken requirements found."
- Installed but **NOT declared** in requirements.txt, yet imported by live code:
  `langgraph` 1.1.10, `langgraph-checkpoint-postgres` 3.1.2, `mcp` 2.1.1, `pgvector` 0.4.2,
  `psycopg` 3.3.4 (v3, used by langgraph postgres saver), `groq` 1.2.0.
- Declared but effectively unused: `psycopg2-binary` is the SQLAlchemy driver (used), `pandas`, `tqdm` used.
- `en_core_web_lg` 3.8.0 IS installed (pip list) — loadable, proven by deid run below.
- **FINDING: no module calls `load_dotenv()`** except `src/reranker/audit_training_quality.py:35`.
  `.env` is therefore inert for every production path. Cascades to Langfuse (Phase 8), MIMIC dir, LLM model names.

## Phase 2 — Docker/Postgres
- Docker Desktop was **paused** at audit start; I ran `docker desktop restart` to bring it up.
- **FINDING: `docker compose ps` is EMPTY.** The committed `docker-compose.yml` (service `db`,
  container_name `lumen-postgres`, named volume `lumen_pgdata`) is NOT what runs. The live DB is
  `lumen-pg`, created 2026-05-11 by a bare `docker run`, no compose labels.
  Drift: `Cmd: ['postgres']` (none of the tuned settings), `Healthcheck: None`,
  `RestartPolicy: no`, **anonymous volume** (a97106af...), `LANG=en_US.utf8` (not `--locale=C`),
  shm 64MB not 1GB. Live `shared_buffers=128MB work_mem=4MB maintenance_work_mem=64MB` = defaults.
  => `docker compose up -d` would collide on port 5433 and create an empty second DB.
  => a stray `docker volume prune` destroys the only copy of 628k chunks.
- PG 16.13, `vector` **0.8.2** (HNSW supported ✓), `pg_trgm` 1.6.
- Row counts: note_chunks 628,945 | clinical_notes 169,061 | guideline_chunks 1,621 |
  patients 5,000 | labevents 995,988 | prescriptions 1,344,936 | checkpoints 143.
- note_chunks: `vector(768)`, HNSW `vector_cosine_ops` (m=16, ef_construction=64), GIN on generated tsvector. ✓
- **guideline_chunks has NO FTS column and NO GIN index** — vector-only.
- LangGraph postgres checkpoint tables exist and hold 143 checkpoints.

## Phase 3 — De-id
- `python -m src.deid.test_pipeline` → 5 synthetic notes, **69/69 expected PHI caught, recall 100%**, 0.057s/note.
- **CRITICAL: raw pre-deid text IS persisted.** `clinical_notes.text_original` non-null on all
  169,061 rows (`src/storage/ingest.py:645,676`). Schema declares it at `src/storage/schema.py:140`.
- **CRITICAL: `clinical_notes.phi_entities` JSONB stores `original_value`** — the extracted PHI
  strings themselves, all 169,061 rows. `src/deid/pipeline.py:392`. A queryable index of the PHI.
- **Silent PHI fallback:** `COALESCE(text_deid, text_original)` at `src/retrieval/index_notes.py:42,44,81,100`
  and `src/storage/ingest.py:688`. If de-id ever returns NULL the raw text is chunked/embedded/served
  with no error. Currently 0 rows have NULL text_deid, so not yet triggered.
- **Over-redaction destroys clinical content (no precision metric anywhere):** observed
  "3 days ago", "approximately 2 hours", "5 days", "35-year-old" → `[DATE]`;
  "4L NC" (4 L nasal cannula) and "He was seen at Stroger Hospital" → `[LOCATION]`.
  `measure_recall` in test_pipeline.py only checks substring-absence; precision is never measured.

### Phase 3 (cont) — de-id on the REAL corpus is destroying clinical content
- 50,000/50,000 sampled chunks contain MIMIC's `___` placeholder => the corpus is
  **already PhysioNet-de-identified**; Lumen's pipeline is a *second* pass on top.
  So `text_original` is DUA-restricted note text, not raw identified PHI. Severity is
  "DUA / defense-in-depth", not "names in the clear". Report it that way.
- Real-corpus entity mix (3–4k notes sampled): DATE_TIME 1567, PERSON 709, LOCATION 640,
  DEVICE_ID 28, PHONE 24, URL 30, INSURANCE 2. **No MRN, no SSN, no AGE_OVER_89** (PhysioNet already stripped them).
- **Top strings redacted as `[PERSON]` on real data are drugs and clinical terms:**
  ED(93) Docusate(78) foley(25) Zometa(19) Foley(18) Enoxaparin Sodium(13) Albuterol Inhaler(13)
  Hodgkin(11) Dorzolamide(11) Famotidine(11) Revlimid(10) LUE(10) JVP(9) Rivaroxaban(8)
  Oropharynx(7) Neurology(7) Emtricitabine-Tenofovir(7) Bibasilar(7) HEENT(6) Clopidogrel(5)
  Morphine Sulfate(5) Isosorbide Mononitrate(5) Alendronate Sodium(6) ...
- **142,296 / 628,945 chunks (22.6%) contain a `[PERSON]` tag.** Docusate survives in only
  3,726 chunks, Clopidogrel 4,606, Enoxaparin 1,673. The indexed corpus that every eval
  number is computed on has drug names replaced by `[PERSON]`.
- SSN: my first probe looked like a miss but is NOT — stock Presidio deliberately
  invalidates reserved/fake SSNs (123-45-6789, 078-05-1120). A realistic SSN
  (456-78-9012) is caught at score 0.85. **SSN detection WORKS.**
- AGE_OVER_89: 90/91/92/103 all caught by AGE_OVER_89; "ninety-five"/"97 last week" caught
  only incidentally as DATE_TIME; boundary "89-year-old" over-redacted as [DATE] (safe direction).
- `src/retrieval/embeddings.py:41-42` hardcodes `Path.home()/"Lumen"/"models"/...`,
  ignoring `MEDCPT_QUERY_MODEL` / `MEDCPT_ARTICLE_MODEL` in .env. Same for
  `DEFAULT_RERANKER_MODEL` at hybrid_retriever_v2.py:54.
- Embeddings: CLS pooling, **L2-normalized**, 768-d; SQL uses `<=>` (cosine) and the index is
  `vector_cosine_ops`. **Normalization and operator MATCH.** ✓

## Phase 4 — Retrieval
- Fusion is **true rank-based RRF** (k=60, bm25_weight=1.0, vector_weight=1.2,
  overlap_bonus=0.5) at hybrid_retriever_v2.py:404-476. No unnormalized score mixing. ✓
  BM25 min-max normalization is display-only, as the comment claims. ✓
- EXPLAIN FTS query: `Bitmap Index Scan on idx_chunks_fts`, 90ms, 575 rows. **No seq scan.** ✓
- **CONFIRMED SILENT DEGRADATION — HNSW ef_search truncation.**
  `SHOW hnsw.ef_search` = **40** (pgvector default). Corpus-wide vector query with
  `LIMIT 60` EXPLAIN ANALYZE returns **rows=39**, not 60. The vector branch silently
  delivers ~2/3 of `vector_top_n=60` on every unscoped query — which is exactly what
  the eval harness runs. Nothing logs or errors.
- Patient-scoped vector search does NOT use HNSW: planner picks `idx_chunks_subject`
  + exact top-N sort (1.5ms, full 60 rows). So patient-scoped is exact, corpus-wide is
  truncated. The two paths have different recall characteristics.
- Patient scoping IS applied in SQL (`AND nc.subject_id = :subject_id`) in both branches. ✓
  But the guard is `if subject_id:` (falsy), not `is not None` — subject_id=0 silently
  disables scoping. Latent, not currently exploitable (no subject_id 0 in MIMIC).
- **Suspected bug in `expand_context` (hybrid_retriever_v2.py:373-397):** it walks adjacent
  chunks in index order and `break`s on budget; if chunk_index-1 is large the *matched*
  chunk can be excluded from `context_text`, and the reranker then scores text that was
  never retrieved. NEEDS EMPIRICAL CONFIRMATION.
- `temporal_fix.py` imports FROM hybrid_retriever_v2 and is imported BY nothing —
  it is a standalone regression-assertion script, not orphaned logic. Live temporal
  handling is `detect_temporal_mode` + `apply_temporal_filter` in hybrid_retriever_v2,
  wired into `search()` at stage 6. ✓

### Phase 4 — CONFIRMED: the vector branch is effectively dead corpus-wide
`vector_search()` as shipped, top_n=60 requested, actual rows returned:
| query | as-shipped | ef=400 |
|---|---|---|
| sepsis | **1** | 60 |
| acute kidney injury rising creatinine | **4** | 60 |
| anticoagulation for atrial fibrillation | **4** | 60 |
| diabetic ketoacidosis management | **1** | 60 |
| heart failure exacerbation diuresis | **17** | 60 |
| pneumonia antibiotic treatment | **0** | 3 |
Root cause chain (each step measured):
 1. `hnsw.ef_search` = 40 (pgvector default, never set by the app).
 2. pgvector HNSW **post-filters**; `token_count >= 40` runs after the index returns its 40.
 3. For "sepsis", **39 of the 40 HNSW candidates have token_count < 40** → 1 survivor.
 4. 91,789 / 628,945 chunks (14.6%) are under the 40-token floor, are embedded and in the
    HNSW graph, and dominate MedCPT query-vector neighborhoods.
=> corpus-wide "hybrid" retrieval is BM25 + reranker. Live run log confirms: `vec=1` for
   "sepsis", `vec=1` for a nonsense query, but `vec=60` for the patient-scoped query
   (which takes the exact `idx_chunks_subject` path, not HNSW).

### Phase 3/4 — chunker bugs (measured)
- `min(token_count)=1` although `min_chunk_tokens=50`. Cause: chunker.py:230-246 flushes the
  small-chunk buffer as its **own** chunk instead of merging it into the next one, contradicting
  its own docstring ("smaller chunks are merged with the next one").
- `_estimate_tokens = words * 1.3` under-counts by **1.35x** vs the real MedCPT tokenizer.
  Measured on 1500 chunks: 21.3% exceed the chunker's own 384 max; **7.1% exceed 512 real
  tokens and are silently truncated** by `embeddings.py` `max_length=512` (~45k chunks corpus-wide).
  On chunks with stored count >250: 70.2% over 384, **23.9% truncated**.
  Stored `token_count` (the estimate) is also what the `min_tokens=40` retrieval filter uses.
- **CONFIRMED BUG `expand_context` (hybrid_retriever_v2.py:373-397): 500/500** sampled positions
  where prev_tok+matched_tok > 600 produce a `context_text` that does **not contain the matched
  chunk**. The reranker then scores, and callers receive, the *preceding* chunk.
  **35,986 / 537,156 retrievable chunks (6.7%) sit at such a position.**

## Phase 5 — LLM
- **The project is NOT on Groq.** Every live LLM call goes through `src/llm/local_client.py`
  → local Ollama `/api/chat`. Groq survives only in `src/reranker/audit_training_quality.py`
  (optional, `openai` commented out of requirements) and in `llm_judge.py`'s docstring/default.
  `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` are NOT used anywhere live.
- Ollama installed models: `qwen3:8b`, `qwen2.5:7b`, `qwen2.5:14b`. Server reachable.
- Call sites (provider=ollama, all temperature 0.0 by default, timeout 300s, 3 retries + jitter):
  | site | tier/model | max_tokens | json |
  |---|---|---|---|
  | graph.triage | fast (qwen3:8b) | 120 | yes |
  | graph.literature_retrieval concept | fast | 80 | yes |
  | graph.synthesis | main (qwen3:8b) | 500 | no (free text) |
  | graph.verification | main | 150 | yes |
  | evals judge (ollama_backend) | **qwen2.5:14b** | 256 | yes |
- **Both tiers resolve to the same model.** `.env` sets LUMEN_LLM_MAIN=LUMEN_LLM_FAST=`qwen3:8b`,
  and the hardcoded defaults are also both `qwen3:8b`, contradicting the docstring ("fast 7B /
  main 14B"). CTX_FAST=8192 vs CTX_MAIN=12288 is the only real difference.
- **Judge model drift:** `ollama_backend.DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"`, but the graph
  uses `qwen3:8b`. Two different model families in one system.
- `ollama_backend` sets **`num_ctx=2048`** while `LLMJudge.max_chunk_chars=4000`; system prompt
  ~500 tok + a 4000-char chunk ~1000-1400 tok + num_predict 256 can exceed 2048 → Ollama
  truncates silently. Silent-degradation risk on the judge.
- GROQ_API_KEY: read only in `audit_training_quality.py:42`, never logged, `.env` gitignored. ✓
- `extract_json`/`coerce_score`: fence-stripping, balanced-brace extraction, trailing-comma
  repair, clamp to 0-3, alt field names. Never raises. ✓ Verified by the module self-test.
- **BUG (critical for metrics): failed judgments are written to the persistent cache.**
  `llm_judge.py:376` caches `{"score": res.score(0), "reason": "JUDGE_ERROR", "error": ...}`.
  **`.cache/llm_judge_ollama.json` holds 645 entries, of which 30 (4.7%) are JUDGE_ERROR/score 0.**
  They are never retried and count as "irrelevant" forever. Contradicts the docstring claim
  that failures are "visible, not silent".
- Cache key = sha256(prompt_version ‖ model ‖ query ‖ chunk[:4000]). **Includes model + prompt
  version** ✓. Threshold is correctly NOT in the key (applied at `is_relevant()` on the 0-3 score).

## Phase 6 — LangGraph
- Graph runs END-TO-END on a real input. `python -m src.agents.run_graph --query "most recent
  creatinine value" --subject REDACTED_1` → trail `triage -> patient_retrieval -> synthesis ->
  verification -> finalize`, `next=()`, answer cited [S1], verified 1/1.
- Refuse branch works: out-of-scope query → `triage -> refuse`, terminates.
- Checkpointer: **real PostgresSaver** on a psycopg3 pool; `checkpoints` table has 143 rows.
  Thread IDs isolate correctly. `checkpointer.setup()` runs on every `build_graph()` (minor).
- **TOPOLOGY DEFECT:** static edge `('verification','__end__')` coexists with the conditional
  branch `verification -> [human_review, finalize]`. verification fans out to END *and* the
  branch. Leftover from the pre-HITL topology.
- **`refuse` has NO outgoing edge** (`b.edges` shows none), despite the module docstring's
  `refuse ──> END`. It works as an implicit terminal but never sets `final_answer` —
  consumers reading `state["final_answer"]` get `""` after a refusal.
- **AgentState has ZERO reducers** — no `Annotated[list, add]` on any field. Every list
  (`node_trail`, `errors`, `egress_log`, `patient_evidence`) is read-modify-write. Safe only
  because execution happens to be sequential; the verification fan-out above is exactly the
  shape that would corrupt it.
- No explicit `recursion_limit`; LangGraph's default 25 applies. Graph is a DAG, cannot loop.
- **FAULT INJECTION — LLM unreachable (`LUMEN_LLM_HOST=http://127.0.0.1:59999`): the run
  reports SUCCESS.** synthesis logs ERROR and returns `draft_answer=""`; verification then sees
  0 citations → `needs_human_review = (0 > 0) = False` → `review_status="auto_approved"`;
  finalize sets `final_answer=""`. Output: empty answer, "verified 0/0", "review? False", exit 0.
  A total LLM outage is indistinguishable from a successful empty answer.
- **PROMPT INJECTION EXPOSURE:** `prompts.build_synthesis_prompt` interpolates raw retrieved
  note text with only an `[S1] (note, date)` header — no fence, no "treat as data" instruction,
  no escaping of `[S#]` markers. A note containing a forged label or instruction text is not
  neutralised. Partial mitigation: the VALID CITATION LABELS list is appended after the
  evidence, and `citations.validate` + `strip_bad_labels` remove labels not in the evidence.
- **BUG `prompts.build_synthesis_prompt`:** `literature_ev = literature_ev or []` is indented
  INSIDE `if guideline_ev:`. With empty guidelines and the default `literature_ev=None`,
  the function raises `TypeError: 'NoneType' object is not iterable`. Reproduced.

## Phase 7 — MCP
- `mcp` 2.1.1, `MCPServer("lumen", version="0.4.0")`, **stdio only**, stdout redirected to
  stderr before torch import and handed back at `mcp.run()`. ✓
- Exactly **4 tools**, matching the brief: `search_patient_notes`, `search_guidelines`,
  `get_lab_trend`, `get_patient_timeline`. No drift.
- Demo plane: `LUMEN_DATA_PLANE=demo python -m src.mcp_server.test_client` → 4 tools, synthetic
  rows returned. Research plane against the live DB → real chunks, real labs, real timeline. ✓
- `planes.enforce_transport` genuinely refuses non-stdio/non-local for the research plane. ✓
- **`search_patient_notes` calls `retriever.search(...)` — the SAME path the agents use.** ✓
  BUT it passes `temporal_filter="auto"`, while `eval_retrieval.py` hardcodes
  `apply_temporal_filter(mode="all")`. **Config drift between eval and production.**
  Proven live: the MCP response carried `"score": 1.2` — impossible for a sigmoid or a
  min-maxed rrf_score; it is 1.0 + the 0.20 temporal boost that only fires in "recent" mode.
- `search_guidelines` has **no `planes.is_demo()` guard** — in the demo plane it still calls
  `_load()` (MedCPT+BGE, ~3.5GB) and queries the real DB, contradicting `_load`'s comment
  "the demo plane never needs the models". Not a DUA leak (guidelines are published), but the
  demo plane is not hermetic. `test_client` hides this by skipping the tool in demo mode.
- `LAB_ITEMIDS` is a hardcoded map with the comment "There is no d_labitems table in this
  schema" — **false**: `d_labitems` exists with 1,650 rows and `src/storage/load_d_labitems.py`
  populates it.
- `get_lab_trend` / `get_patient_timeline` both `ORDER BY charttime ... LIMIT n` — they return
  the **OLDEST** n events, not the most recent. For tools whose stated purpose is trend and
  orientation, current values are unreachable.
- `get_lab_trend` returns `{"error": ...}` as a *successful* MCP response, not an MCP error.

## Phase 8 — Langfuse
- **VERIFIED CONNECTED, trace actually landed.** With `.env` exported: auth_check True, and
  the Langfuse API shows my run as trace `c1fed62b810b`, name `LangGraph`,
  session `audit-ok-1`, tags `['agent-graph','lumen']`, **13 observations**.
- Span coverage: triage, patient_retrieval, synthesis, verification, finalize, all 4 routers,
  and 3 GENERATION spans (`ollama:fast`, `ollama:main` ×2) with real token usage
  (290/19, 898/35, 2435/35). **Missing: every MCP tool call and every eval run** — neither
  `src/mcp_server/` nor `src/evals/` imports `tracing` at all.
- **PHI/clinical text IS in traces, by design.** Trace payload 103,855 chars containing
  `___`×334, `[DATE]`×64, `[PERSON]`×10, `[LOCATION]`×10, `subject_id`×13.
  Guarded by `tracing._host_is_local()`, which refuses any non-localhost LANGFUSE_HOST —
  a real, working control. Risk: it is host-string-based only.
- **`ENABLED` is read at import time and nothing calls `load_dotenv()`** → in a plain
  `python -m ...` run `LUMEN_TRACING` is unset and tracing is a silent no-op. Only 6 traces
  exist in the project total.
- `flush()` is called at every exit in `run_graph.py`; NOT in the MCP server or eval scripts.
- Degradation: sticky `_init_failed`, warnings only, never raises. ✓ Observed a real
  `Failed to export span batch ... Read timed out` that did not affect the run.

## Phase 9 — Eval methodology
- `golden_dataset.py`: docstring says **30** queries, there are **28**. Labels are hand-written
  **keyword criteria**, no chunk-level labels. `min_relevant` (2 or 3) is a hand guess.
  No subject_id scoping on any query.
- **`eval_retrieval.py` does NOT use the LLM judge.** It uses its own `judge_relevance`
  (keyword substring matching) at eval_retrieval.py:77,306. It never imports `llm_judge`.
  `build_pooled_relevance` / `ndcg_at_k_graded` / `recall_at_k_pooled` are used ONLY by
  `judge_and_score.py`. So the harness the results come from runs the keyword judge that
  llm_judge.py's own docstring says "inflated and saturated the earlier scores".
- **Circularity:** criteria are keywords; BM25 retrieves by keyword; a BM25 hit is therefore
  very likely to satisfy the criteria. The eval structurally favours the BM25 branch.
- **`eval_retrieval.ndcg_at_k` IDCG is built from the RETRIEVED set, not the pool.**
  Hand-verified: `[T,F,F,F,F]` → nDCG@5 = **1.0000**, and `[T,T,T,T,T]` → nDCG@5 = **1.0000**.
  Finding 1 relevant result ties finding 5. The "HEAD-TO-HEAD ... best config (by nDCG@5)"
  table is therefore meaningless.
  (Hand arithmetic for `[F,F,F,T,T]`: DCG = 1/log2(5)+1/log2(6) = 0.430677+0.386853 = 0.817529;
   IDCG = 1/log2(2)+1/log2(3) = 1.0+0.630930 = 1.630930; nDCG = 0.501266 — matches the code.)
- **`llm_judge.ndcg_at_k_graded` IS correct.** Hand-verified: pool [3,3,2,2,1,0,0],
  ranked [3,0,2,0,0], k=5 → DCG = 7/log2(2) + 3/log2(4) = 7.0+1.5 = 8.5;
  ideal top-5 [3,3,2,2,1] → IDCG = 14.595391; nDCG = 0.582376 — matches the code.
- `eval_retrieval.recall_at_k` denominator = `q["min_relevant"]`, a guess; returns **1.0**
  when `min_relevant<=0`. `recall_at_k_pooled` uses the pooled count ✓ but returns 0.0 on an
  empty pool, which drags the mean rather than excluding the query.
- Pooling bias (a doc no config retrieves is invisible) is **not reported anywhere** in output.
- **Export persists no config**: `eval_retrieval.run_eval`'s `export_data` contains metrics only
  — no model ids, weights, k, chunk settings, timestamp, or git SHA.

### Phase 9 — REAL EVAL OUTPUT (`python -m src.evals.eval_retrieval --quiet`)
```
  Config                            P@5    R@5    MRR   nDCG@5  Avg Time  Queries
  BM25 Only                       0.88   0.93   0.92     0.92     0.46s       28
  Vector Only (MedCPT)            0.64   0.57   0.65     0.66     0.10s       28
  Hybrid RRF                      0.86   0.94   0.90     0.92     0.44s       28
  Hybrid + BGE Reranker           0.91   0.96   0.95     0.95    12.59s       28
  Hybrid + MedCPT Cross-Enc       0.88   0.95   0.93     0.94     2.70s       28
  HEAD-TO-HEAD (by nDCG@5): BM25 Only 22 | Vector Only 4 | Hybrid+BGE 2 | Hybrid RRF 0 | Hybrid+MedCPT 0
```
- **102/140 (73%) of config×query nDCG@5 cells are exactly 1.0** — saturated by the IDCG bug.
- **23/28 queries have a TIE for best nDCG@5; 20 of those ties are awarded to "BM25 Only"
  solely because it is the first key in the config dict** (`if r.ndcg_5 > best_ndcg`, strict >).
  The "BM25 Only wins 22" headline is an artifact of dict insertion order.
- **NONDETERMINISM CONFIRMED — two identical back-to-back runs differ.**
  `26 / 140` (config, query) result sets differ. Aggregates move:
  BM25 nDCG@5 0.9241 → 0.9231; Hybrid RRF nDCG@5 0.9157 → 0.9178;
  Hybrid+MedCPT P@5 0.8786 → 0.8857.
  Cause: `ORDER BY bm25_score DESC LIMIT 60` with **no tiebreaker**; tied `ts_rank_cd`
  values come back in arbitrary heap order (med_001: [171172,146431,386508] vs
  [386508,146431,171172]) and tied rows churn across the LIMIT boundary.
  Vector Only is fully deterministic (fixed HNSW graph).

### Phase 9 — judge validity (10 real pairs, hand-checked)
Rubric: no leading language; explicitly warns against topical over-scoring; 0-3 scale
matches `is_relevant(threshold=2)` where 2 = "discusses the target". Prompt is sound.
Model (qwen3:8b) is the weak link. My verdict on 10 spot-checks: **agree 8, disagree 2.**
- DISAGREE chunk 146431 / "heart failure medications on discharge" → judge 2. The chunk is a
  vein/artery ultrasound impression + a PMH mention of HFpEF; **no medication appears**.
  The judge's own reason concedes it "does not explicitly list the specific medications".
  Per its own STEP 2 this is a 0-1.
- DISAGREE chunk 358031 / "sepsis treatment antibiotics vasopressors" → judge 2. Text is
  dehydration/beta-blocker/amiodarone/heel osteomyelitis. Reason claims it "discusses sepsis
  treatment and antibiotics" — **not present**. Should be 0.
- Judge reasons cite specifics absent from the text in ≥3 of 10 (386508 "spironolactone 25 mg,
  torsemide 40 mg"; 138311 "Levophed"). The score is often right, the justification fabricated.
- **The keyword judge marked ALL 10 pairs relevant=True**, including both that the LLM judge
  itself scored only 2 and both I judge irrelevant. That is the saturation mechanism, observed.
- **The judge cache is not auditable**: entries store only `{score, reason, error}` keyed by a
  sha256. The query and chunk text are not retained, so a cached judgment can never be
  reviewed, re-verified, or attributed to a chunk.

### Phase 3 (cont) — chunks do not map back to source
- Only **3 / 2000** chunk_texts are findable verbatim in their `clinical_notes.text_deid`.
- After stripping the `[SECTION] ` prefix the chunker prepends: **935 / 2000 (47%)**.
  The other 53% are unfindable because the chunker rebuilds text with `" ".join(sentences)`,
  collapsing the original newlines.
- `note_chunks` has **no start/end offset columns at all**. `state.py`'s claim that the
  citation chain can "resolve a claim back to an exact source span" is not achievable;
  resolution stops at `chunk_id`.

### Phase 1/misc
- **All 43 modules import cleanly** — no ImportError, no circular imports.
- **No import-time DB connections, network calls, or model loading** (socket-level probe:
  `sockets_opened=none` for storage, tracing, local_client, embeddings, hybrid_retriever_v2,
  planes, llm_judge, agents.graph, deid.pipeline). `src.storage` builds a lazy SQLAlchemy
  engine at import; `tracing.ENABLED`, LLM model names, and `planes.PLANE` are all read at
  import time (which is why the missing `load_dotenv()` bites).
- **EXCEPTION: `src/evals/test_judge_live.py` fires 3 live LLM calls at module import** —
  module-level code with no `if __name__ == "__main__"` guard.
- **No automated test suite.** `python -m pytest src/ --collect-only` → "no tests collected".
  The four `test_*.py` files are manual scripts.
- `src/storage/ingest.py:641` stores `deid_recall = 1.0` with the comment "we don't have
  ground truth for MIMIC" — a fabricated perfect score persisted on all 169,061 rows.
- Re-ingest is destructive by design (DELETE chunks + notes for a note_type, then re-INSERT,
  atomically in one transaction). Because `note_id` is SERIAL, re-ingest renumbers every note
  and invalidates all previously recorded chunk_id/note_id references.
- `src/agents/citations.py` is sound: hallucinated labels are detected and stripped in code.
- `src/retrieval/guideline_retriever.py` is honest about the no-BM25 limitation; `vector_top_n=20`
  is under `ef_search=40` and there is no `min_tokens` filter, so it does NOT suffer the
  truncation bug that kills the note vector branch.
- `src/safety/egress_gate.py`: ordered named rules, first match wins, blocked payloads are
  hashed not logged, 24-char shingle near-verbatim detection against retrieved evidence.
  Well-constructed. (eval_egress.py NOT RUN — see Unverifiable.)

## Phase 9 — CORRECTION: there are TWO eval pipelines, and the good one is not `eval_retrieval.py`
The README's published table comes from `retrieve_pool.py` → `judge_and_score.py`, NOT from
`eval_retrieval.py`. Recomputed from the committed `results.json`, it reproduces exactly:
```
Config                          P@5     R@5     MRR   nDCG@5
BM25 Only                     0.725   0.241   0.771    0.555
Vector Only (MedCPT)          0.534   0.131   0.571    0.334
Hybrid RRF                    0.750   0.248   0.795    0.510
Hybrid + BGE Reranker         0.843   0.276   0.884    0.687
Hybrid + MedCPT Cross-Enc     0.807   0.268   0.839    0.568
```
`results.json._meta = {"model":"qwen2.5:14b","threshold":2,"top_k":5}` — **this pipeline DOES
persist its config**, uses **pooled relevance** (n_relevant_pool mean 16.6, max 29), and uses the
**correct** `ndcg_at_k_graded`. So my earlier "no config persisted / guessed denominator /
broken IDCG" findings apply to `eval_retrieval.py` ONLY.
The README even states the keyword judge was abandoned for saturating at 0.93–1.00 recall —
and `eval_retrieval.py` today reproduces exactly those abandoned numbers (R@5 0.93–0.96).
**=> `eval_retrieval.py` is a superseded harness that was never removed. It still runs, still
prints an authoritative summary + a winner table, and its numbers contradict the README.**

### Judge cache — the mechanism is CORRECT (this exonerates the code)
Commit `d7acaf8` (2026-08-25) rewrote SYSTEM_PROMPT **and bumped `PROMPT_VERSION v1 → v2` in the
same commit**. Proof the keying works, by recomputing keys for all 561 pooled (query,chunk) pairs
against `.cache/llm_judge_ollama.json` (645 entries):
```
  prompt_version=v1  model=qwen2.5:14b  -> 561/561 present
  prompt_version=v1  model=qwen3:8b     ->   0/561
  prompt_version=v2  model=qwen2.5:14b  ->   0/561
  prompt_version=v2  model=qwen3:8b     ->   0/561
```
=> No stale-prompt contamination is possible. The cache is now 100% ORPHANED (all v1); a re-run
today re-judges everything under v2. The 30 JUDGE_ERROR entries are therefore also orphaned —
the "cached failure" bug is real but its current blast radius is zero.

### Staleness of the published numbers (real finding)
`results.json` / `pooled.json` / the cache: **Jul 7**. `hybrid_retriever_v2.py`: **Aug 24**.
`llm_judge.py`: **Aug 25** (the v1→v2 prompt rewrite). The README's benchmark was produced by a
retriever and a judge prompt that no longer exist in the tree.

### Pooled recall — empty-pool handling
1 of 28 queries has `n_relevant_pool = 0`; `recall_at_k_pooled` returns 0.0 for it rather than
excluding it, depressing every config's mean recall by ~3.6%. The pooling bias (a chunk no
config retrieves is invisible) is not reported in any output.

## Phase 2 (cont) — index usage
- `idx_chunks_embedding` (HNSW, 2338 MB) 1,100 scans ✓ ; `idx_chunks_fts` (142 MB) 1,360 scans ✓
- **`idx_guideline_embedding` (6.5 MB): 0 scans.** EXPLAIN confirms guideline vector search is a
  **Seq Scan** over all 1,621 rows (162 ms, exact). Correct planner choice at that size, but the
  HNSW index is dead weight and `guideline_retriever.py`'s docstring implies it is used.
- `idx_notes_fts` (112 MB) on `clinical_notes.text_search`: 519 scans, but the live retriever
  ranks on `note_chunks.text_search`. Superseded by `migrate_chunk_fts` — 112 MB maintained for
  a column no live query ranks on.
- `idx_lab_charttime` (16 MB): 0 scans.
- `migrate_chunk_fts --verify-only`: 628,945/628,945 populated, 100.0% coverage,
  5/5 distinct ranks in the top-5 sample. **Migration VERIFIED APPLIED.**

### Phase 4/9 — CORRECTION on eval-vs-production temporal drift
`detect_temporal_mode` resolves **"all" for all 28 golden queries**. So `eval_retrieval.py`'s
hardcoded `mode="all"` is behaviourally identical to production *for the golden set*.
The drift is **latent, not active**. The real finding is narrower and cleaner:
**the golden dataset contains zero temporal queries, so `eval_retrieval.py` never exercises
the temporal path at all** — the project's stated differentiator is unmeasured there.
`eval_temporal.py` exists precisely to cover that gap (6 cases, patient-scoped).
Live proof the temporal path DOES fire in production: the MCP call for
"recent potassium and kidney function" returned `"score": 1.2` — 1.0 (min-maxed rrf) plus the
0.20 recency boost, which only "recent"/"latest" modes apply.
Everything else in the eval matches HybridRetriever defaults: top_n=60, min_tokens=40,
max_per_note=2, rerank_candidates=40, window=1, max_context_tokens=600, RRF k=60 /
bm25_weight=1.0 / vector_weight=1.2 / overlap_bonus=0.5. **No config drift on those.**

### Embeddings — data-level verification (clean)
`note_chunks`: 628,945 rows, **0 NULL embeddings, 0 zero-vectors, 0 non-unit-norm** (|‖v‖−1|>0.01).
`guideline_chunks`: 1,621 rows, 0 NULL, 0 zero. So the L2 normalisation in
`embeddings.py::_encode_batch` held for the whole corpus, and it matches the `<=>` cosine
operator and the `vector_cosine_ops` index. The NaN→zeros fallback never fired.

### Phase 6 — egress gate eval (RUN, passes honestly)
`python -m src.evals.eval_egress`:
```
  family                   n  correct    rate   rules fired
  direct_verbatim         10       10   100%   deid_marker:5, note_structure:4, verbatim_overlap:1
  paraphrase_smuggle      10        7    70%   deid_marker:3, lab_shorthand:3, shifted_date:1
  identifier              10       10   100%   identifier:8, shifted_date:2
  oversized               10       10   100%   oversized_payload:7, deid_marker:3
  benign_control          10       10   100%   -
  BLOCK RATE 37/40 = 92%   FALSE-BLOCK RATE 0/10 = 0%
```
3 misses, all `paraphrase_smuggle`, **printed explicitly** rather than hidden. Good eval hygiene.

## Phase 4/9 — TEMPORAL EVAL RESULT (`python -m src.evals.eval_temporal`) — the differentiator fails
```
Selected patients: [(REDACTED_2, 354 notes, 345 times), (REDACTED_3, 326, 312)]

--- patient REDACTED_2 ---
  temp_latest_creatinine  hit@1 temporal=False all=False  (newest at rank 6 / 6)
  temp_latest_hgb         hit@1 temporal=False all=False  (newest at rank 9 / 10)
  temp_latest_meds        hit@1 temporal=False all=False  (newest at rank 5 / 10)
  temp_trend_creatinine   monotonicity temporal=0.00 all=1.00  (2 timepoints)
  temp_trend_potassium    monotonicity temporal=0.50 all=0.50  (7 timepoints)
  temp_window_labs        same-admission temporal=0.33 all=0.10  (n=3)
--- patient REDACTED_3 ---
  temp_latest_creatinine  hit@1 temporal=False all=False  (newest at rank 4 / 5)
  temp_latest_hgb         hit@1 temporal=False all=False  (newest at rank 7 / 10)
  temp_latest_meds        hit@1 temporal=False all=False  (newest at rank 5 / 10)
  temp_trend_creatinine   n/a (no relevant timepoints retrieved)
  temp_trend_potassium    monotonicity temporal=0.75 all=0.44  (7 timepoints)
  temp_window_labs        same-admission temporal=0.33 all=0.10  (n=6)

  TEMPORAL LIFT (arrow reads all -> temporal)
  latest   hit@1:        0.00 -> 0.00   (++0.00)
  trend    monotonicity: 0.65 -> 0.42   (+-0.23)   <- temporal is WORSE
  window   same-adm:     0.10 -> 0.33   (++0.23)   <- temporal is better
```
- **`latest` hit@1 = 0/6.** Temporal mode never once put the newest relevant record at rank 1.
- **`trend` is actively harmed** by temporal mode (-0.23 monotonicity).
- Only `window` (which uses "recent" mode's HARD DROP) helps.
- Display bug: the lift is printed as `+{value:+.2f}` so a negative lift renders `+-0.23`.
- Sample is tiny: 2 patients, 6 cases, 1 n/a.

### ROOT CAUSE — proven with before/after on patient REDACTED_2
`apply_temporal_filter` runs at **stage 6**; `BGEReranker.rerank` runs at **stage 8** and does
`result.final_score = sigmoid(logit)` then `results.sort(key=rerank_score, reverse=True)` —
it **never reads rrf_score**, so everything stage 6 did to the ordering is discarded.
```
use_reranker=False  mode=trend  -> monotonicity 1.00
   order: 2183-04-14, 2183-11-05, 2184-01-30, 2184-04-06, 2184-04-21, 2184-04-26, 2184-05-25, 2184-05-28
use_reranker=True   mode=trend  -> monotonicity 0.14
   order: 2190-11-17, 2189-09-18, 2189-09-17, 2187-06-22, 2189-09-18, 2186-07-11, 2184-11-12, 2184-01-30
```
The chronological sort is perfect until the reranker destroys it.
Second, separate defect for `latest`: the boost is `0.20 * 0.5^(days_ago/180)` added to a
**min-maxed [0,1]** rrf_score. The newest record ranked 8/8 by relevance; +0.20 cannot lift it
past seven others. Identical output with and without the reranker for that query (the reranker
fell back to RRF order), so here the boost magnitude — not the reranker — is the problem.
=> Temporal handling IS wired into the live path (stage 6, `temporal_filter="auto"`), but two
   independent defects mean it only works in the one mode that removes candidates outright.

## Phase 4 — CROSS-PATIENT LEAKAGE: RESOLVED, NO LEAK
`/tmp/.../leak.py` — 6 highest-volume patients x 7 queries (incl. `' OR 1=1 --`, `%`,
`[PERSON] ___ Unit No:`), top_k=10:
```
checked 6 patients x 7 queries -> 420 returned chunks
chunks belonging to a DIFFERENT subject_id: 0
nonexistent subject_id=999999999 -> 0 results (expect 0)
subject_id=0 (falsy) -> 5 results, subjects=[REDACTED_4, REDACTED_5, REDACTED_6, REDACTED_7, REDACTED_8]
```
- **0 cross-patient leaks in 420 chunks.** Scoping is enforced in SQL in BOTH branches and is
  parameterised, so query text cannot reach the filter. NOT a critical finding. VERIFIED WORKING.
- A nonexistent patient returns 0 results — no silent fallback to corpus-wide search.
- **CONFIRMED latent defect:** `subject_id=0` returns 5 chunks from **5 different patients**.
  `if subject_id:` (falsy) at hybrid_retriever_v2.py:194,340 silently drops the filter.
  Not reachable today (MIMIC subject_ids are positive 8-digit integers; MCP and the graph pass None or a
  real id), but a caller that coerces a missing value to 0 would get unscoped corpus-wide search.
  Fix: `if subject_id is not None:` — 2 lines.

---
# FIX LOG (2026-09-09) — audit complete, fixes applied. Nothing committed.

Order was chosen so each fix's verification is trustworthy: determinism first,
because before/after on anything else is meaningless while results vary run to run.

| # | Fix | File | Before -> After |
|---|---|---|---|
| 1 | `chunk_id` tiebreaker on both BM25 ORDER BYs + the Python merge sort | hybrid_retriever_v2.py | 14/28 -> 0/28 queries nondeterministic (in-process) |
| 2 | `HNSW_EF_SEARCH` 40 -> 1000, `SET LOCAL` per query, clamped 1..1000, under-yield log | hybrid_retriever_v2.py | 274/1680 -> **1680/1680** rows; 0/28 -> **28/28** queries at full top_n |
| 2b | Empty/whitespace query guard (regression exposed by fix 2) | hybrid_retriever_v2.py | `''` returned 5 arbitrary chunks -> 0 |
| 3 | `expand_context` seeds the matched chunk into the budget; break -> continue; assembles in index order | hybrid_retriever_v2.py | 500/500 -> **0/500** matched chunks dropped; 0/800 general; 705/800 still enriched |
| 4 | `apply_temporal_filter` gained `score_attr`; stage 8 reranks the full candidate set; new stage 9 applies temporal to `final_score` before the top_k slice | hybrid_retriever_v2.py | `trend` monotonicity **0.14 -> 1.00** |
| 4b | `latest` = recency-first, relevance as tiebreaker (approved semantics change) | hybrid_retriever_v2.py | newest at rank 8/8 -> **1/8**; eval hit@1 **0/6 -> 6/6** |
| 5 | Empty draft = failure: `synthesis_failed`, `review_status="failed"`, `needs_human_review=True`; human_review no longer launders it to auto_approved | agents/graph.py | outage reported `auto_approved` -> now `review? True` + 2 ERROR lines |
| 6 | Removed duplicate `add_edge("verification", END)`; added `refuse -> finalize`; reducers on `node_trail`/`errors` + `_trail` returns only its own entry | agents/graph.py, agents/state.py | trail `triage -> refuse` -> `triage -> refuse -> finalize`; no trail doubling |
| 7 | `_fence()` wraps evidence in `<<<EVIDENCE…EVIDENCE>>>` and strips injected delimiters; SYNTHESIS rule 8; `s1 -> s2`; fixed `literature_ev` indentation crash | agents/prompts.py | injected `EVIDENCE>>>` neutralised; TypeError gone |
| 8 | Judge no longer caches failures; `JUDGE DEGRADED` warning when any judgement fails | evals/llm_judge.py | failure retried next run, success still cached (1 new call, not 2) |
| 9 | `expand_query` uses insertion-ordered dedup, not `list(set(...))` | hybrid_retriever_v2.py | **cross-process** nondeterminism: 3/140 -> **0/140** |
| 10 | `load_dotenv()` once in `src/__init__.py` (override=False) | src/__init__.py | `.env` was inert; tracing now ON, MIMIC dirs resolve; shell env still wins |
| 11 | Retired `eval_retrieval.py` (exits 2 with guidance); configs moved to `retrieval_configs.py`; `retrieve_pool.py` repointed | evals/*.py | the harness contradicting README.md can no longer be run |

## FIX 9 IS THE ONE TO REMEMBER
`expand_query` returned `list(set(...))`. Python randomises string hashing per
process, and `bm25_search` only uses `expansions[:8]`, so **every process picked a
different 8 synonyms**, built a different OR query, and retrieved a different
candidate set. In-process probes looked perfectly stable; only cross-process runs
diverged. => **Every eval number this project has published was computed with a
randomly-varying query expansion.**

## END-TO-END VERIFICATION AFTER ALL FIXES
- Determinism: **0/140** result sets differ across two full runs; all aggregates identical.
- `python -m src.retrieval.temporal_fix` -> all 5 assertions pass.
- All **43** modules import; no ImportError, no circular imports.
- Graph: `triage -> patient_retrieval -> synthesis -> verification -> finalize`, cited + verified 1/1.
- Two-phase eval works end to end (retrieve_pool -> judge_and_score), metric unsaturated (nDCG 0.45-0.63).

## STILL OPEN
- #3 de-id over-redaction + #4 chunker — need a shared/staged reindex. **MEASURED THE CASE:**
  176/235 (75%) of the golden set's own relevance keywords are destroyed by de-id
  (lasix 625x, albuterol 999x, enoxaparin 403x, vancomycin 362x, troponin 136x);
  PERSON+LOCATION cause 90% of it and catch ZERO real PHI on this corpus.
  Cost: de-id needs NO Presidio re-run (phi_entities has exact start/end offsets to
  replay); re-embed is the cost, measured at 18 chunks/sec => ~3.4 h for the
  217,778 affected chunks (chunk_ids preserved), ~9.7 h if the chunker changes
  boundaries too (everything renumbers).
- docker-compose.yml vs the real `lumen-pg` container (anonymous volume, restart:no).
- `text_original` / `phi_entities.original_value` retention — user's DUA call.
- No test suite (`pytest --collect-only` -> no tests collected).
