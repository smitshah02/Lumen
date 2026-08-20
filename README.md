# Lumen

A clinical retrieval-augmented generation system over MIMIC-IV. Lumen answers
patient-scoped clinical questions from de-identified discharge summaries and
radiology reports, and grounds every claim in a citation back to the source note.

The design constraint that shapes everything here: **no clinical text leaves the
machine.** Retrieval, reranking, and answer generation all run locally against a
local Postgres instance and a local Ollama model. The only hosted API in the
project is the LLM judge used for offline evaluation.

---

## Why this exists

Most RAG demos retrieve from clean, well-formed documents. Clinical notes are
neither. They are semi-structured, heavily abbreviated, inconsistently
sectioned, and full of de-identification placeholders. A patient's answer is
often spread across several notes written weeks apart.

Lumen is an attempt to build retrieval that survives that, and to measure it
honestly rather than with a metric that flatters itself.

---

## Architecture

```
MIMIC-IV (csv.gz)
      │
      ▼
  ingest  ──────────────►  Postgres + pgvector
      │                     patients, admissions, diagnoses, labevents,
      │                     notes, chunks, guideline_chunks
      ▼
  de-identification        Presidio + spaCy en_core_web_lg
      │                    custom clinical recognizers (MRN, ages >89, …)
      ▼
  section-aware chunking   splits on clinical headers first, then
      │                    overlapping sentence windows @ 512 tokens
      ▼
  MedCPT dual-encoder      article encoder → chunks
                           query encoder → queries          768-dim
```

Retrieval is a nine-stage pipeline ([hybrid_retriever_v2.py](src/retrieval/hybrid_retriever_v2.py)):

1. **Query expansion** — clinical synonyms and abbreviations, to bridge the gap between how a person asks ("swollen legs") and how a clinician writes ("peripheral edema")
2. **BM25** over Postgres `tsvector`, OR + AND combination
3. **Vector search** via pgvector cosine over MedCPT embeddings
4. **Quality filter** — drops the one-line radiology indications that otherwise pollute the top-K
5. **Reciprocal Rank Fusion**, with a bonus for chunks found by *both* arms
6. **Context window expansion** — pulls adjacent chunks so the reranker sees a clinical picture, not a fragment
7. **Note-level dedup** — stops three chunks from one note from monopolizing the results
8. **Temporal filter / boost**
9. **BGE cross-encoder rerank** over the assembled context → final top-K

Generation ([answer_generator.py](src/generation/answer_generator.py)) is grounded-only:
the system prompt forbids outside knowledge about the patient, requires a `[S#]`
citation on every claim, and returns a fixed refusal sentence when the context
doesn't contain the answer. `subject_id` / `hadm_id` pass straight through to
retrieval, so one patient's question can never surface another patient's notes.

---

## Results

28 golden queries across six categories (medications, labs, diagnosis, imaging,
sections, plain-language), graded by an LLM judge at relevance threshold 2 on a
0–3 scale, `top_k=5`. Full per-query output in [results.json](results.json).

| Configuration | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 only | 0.725 | 0.241 | 0.771 | 0.555 |
| Vector only (MedCPT) | 0.534 | 0.131 | 0.571 | 0.334 |
| Hybrid RRF | 0.750 | 0.248 | 0.795 | 0.510 |
| **Hybrid + BGE reranker** | **0.843** | **0.276** | **0.884** | **0.687** |
| Hybrid + MedCPT cross-encoder | 0.807 | 0.268 | 0.839 | 0.568 |

Two things worth reading carefully:

**Recall looks low because the denominator is honest.** Relevance is pooled
TREC-style — the union of every configuration's results for a query is judged
once, producing a shared relevant set that often runs to 17–20 chunks. Recall@5
is then capped at roughly 5/18 by construction. The number is comparable *across
rows*, which is what it is for; it is not a claim that the system finds a
quarter of the relevant material.

**These numbers replaced a much prettier set.** The original judge scored a chunk
by keyword presence — the same signal BM25 retrieves on. It was circular, and it
saturated: every configuration landed between 0.93 and 1.00 recall, which told me
nothing except that the metric was broken. Swapping in an
[LLM judge](src/evals/llm_judge.py) that reads the chunk and grades whether it
actually answers the question cut the scores roughly in half and made the
configurations separable. The lower table is the useful one.

The judge is deterministic (temperature 0), disk-cached by
`(prompt_version, model, query, chunk)` so a pair is graded once ever, and
degrades to score 0 with a visible error flag rather than silently crediting a
chunk it failed to parse.

---

## Layout

| Path | What's in it |
|---|---|
| [src/storage/](src/storage/) | Postgres schema, MIMIC ingestion, FTS migration, lab dictionary loader |
| [src/retrieval/](src/retrieval/) | Chunker, MedCPT embeddings, hybrid retriever (v1 + v2), note/guideline indexers, temporal handling |
| [src/deid/](src/deid/) | Presidio de-identification pipeline and MIMIC adapter |
| [src/generation/](src/generation/) | Grounded answer generation, lab querying, batch harness |
| [src/evals/](src/evals/) | Golden dataset, LLM judge, pooled scoring, retrieval + temporal evaluation |
| [src/reranker/](src/reranker/) | Cross-encoder training-data pipeline and quality audit ([details](src/reranker/README_reranker_csv_pipeline.md)) |

---

## Running it

Requires Python 3.11+, Postgres 15+ with the `vector` and `pg_trgm` extensions,
and roughly 16 GB RAM (the reranker and a 14B Ollama model share it). Apple
Silicon MPS is used automatically when available.

```bash
python -m venv .venv && source .venv/bin/activate
pip install sqlalchemy psycopg2-binary pgvector torch transformers \
            presidio-analyzer presidio-anonymizer spacy \
            pandas numpy tqdm pypdf python-dotenv requests openai groq
python -m spacy download en_core_web_lg
```

Model weights are not vendored. Download to `models/`:

- [`ncbi/MedCPT-Query-Encoder`](https://huggingface.co/ncbi/MedCPT-Query-Encoder) → `models/medcpt-query`
- [`ncbi/MedCPT-Article-Encoder`](https://huggingface.co/ncbi/MedCPT-Article-Encoder) → `models/medcpt-article`
- [`ncbi/MedCPT-Cross-Encoder`](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) → `models/medcpt-cross-encoder`
- [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) → `models/bge-reranker`

Set `DATABASE_URL` and `GROQ_API_KEY` in `.env`, then:

```bash
python -m src.storage.schema          # create tables
python -m src.storage.ingest          # load MIMIC-IV
python -m src.retrieval.index_notes   # chunk + embed notes
python -m src.evals.eval_retrieval    # reproduce the table above
```

Ask a question:

```bash
ollama serve && ollama pull qwen2.5:14b
python -m src.generation.answer_generator \
    "abnormal potassium labs" --subject 10014354 --top-k 6
```

---

## Data

**No patient data is in this repository, and none should be added to it.**

MIMIC-IV is credentialed-access under a PhysioNet data use agreement. Obtaining
it requires CITI training and a signed DUA at
[physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/). The
`.gitignore` here excludes the dataset, model weights, all derived exports
(patient summaries, disease flows, reranker training pairs), and any evaluation
artifact that embeds note text — `results.json` is committed precisely because
it contains metrics only.

If you fork this and wire in your own data, re-check that exclusion list before
your first commit. Note excerpts turn up in eval output in places that are easy
to miss.