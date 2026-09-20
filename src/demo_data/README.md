# Lumen synthetic demo data

**Every record in this directory is synthetic.** The patients, admissions, labs,
medications, diagnoses and notes are fictional. They were written from invented
clinical scenarios in `generate.py` (fixed seed, deterministic output).

- **No MIMIC data.** Nothing here was copied, sampled, paraphrased, summarised or
  derived from MIMIC-IV or any other patient record. The generator never reads
  restricted data.
- **No PHI.** There are no names, contact details or locations. IDs are in
  artificial ranges: subjects `90000001+`, admissions `91000000+`, notes
  `92000000+`, lab items `990001+`. Real MIMIC subject IDs are `1xxxxxxx`.
- **Demonstration only.** This data exists to run Lumen end to end for local and
  cloud deployment demos. It is not for clinical use, not medical advice, and not
  a benchmark of clinical accuracy.
- **Separate plane.** It loads only into the demo database (`lumen_demo`) with
  `LUMEN_DATA_PLANE=demo`. The loader refuses to run in any other plane and never
  touches the research database.

```bash
python -m src.demo_data.generate --check                  # files match the generator
python -m src.demo_data.generate --validate               # integrity checks, writes nothing
LUMEN_DATA_PLANE=demo python scripts/load_synthetic_demo.py
LUMEN_DATA_PLANE=demo python -m src.retrieval.index_notes
LUMEN_DATA_PLANE=demo python scripts/demo_smoke_test.py safety
```

36 fictional patients, 81 admissions and 136 notes covering readmissions, elective
and emergency episodes, longitudinal lab trends, medication starts/stops/restarts,
improving and deteriorating courses, and two deliberately ambiguous records.

`golden_qa.json` holds 40 questions with deterministic expected facts, plus a short
reference answer, difficulty, answer type and evidence admissions for future
retrieval and answer-level scoring. Two questions are deliberately unanswerable or
ambiguous. `manifest.json` records the seed, row counts and file hashes.

Every build runs `validate()`: identifier ranges and uniqueness, orphan rows,
admission date ordering and overlap, events inside their admission window, PHI
patterns in note text, and golden questions pointing at real synthetic records.
