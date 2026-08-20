# MIMIC-IV Reranker Training Data Pipeline (CSV.GZ Version)

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                     MIMIC-IV csv.gz files                        │
│  (patients, diagnoses, labs, procedures, prescriptions, notes)   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Exclude 5000 RAG   │◄── Reads IDs from your PostgreSQL
            │  patient IDs        │    (or a text file)
            └──────────┬──────────┘
                       │
         ┌─────────────▼─────────────┐
         │  Remaining patients only   │
         └─────────────┬─────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌─────────┐    ┌──────────────┐    ┌──────────────┐
│Structured│    │ Clinical     │    │ Section-split │
│ data     │    │ notes (raw)  │──▶│ note chunks   │
│(ICD,labs,│    │              │    │               │
│ meds...) │    └──────────────┘    └──────┬───────┘
└────┬─────┘                               │
     │         ┌───────────────────────────┘
     ▼         ▼
┌──────────────────────┐     ┌─────────────────────┐
│ Pseudo-query from    │────▶│ Best-matching chunk  │
│ structured entry     │     │ = POSITIVE           │
└──────────────────────┘     └─────────────────────┘
                                      │
                             ┌────────▼────────┐
                             │   NEGATIVES:     │
                             │ • Same patient,  │
                             │   diff admission │
                             │ • Same diagnosis,│
                             │   diff patient   │
                             │ • Random chunks  │
                             └────────┬────────┘
                                      ▼
                             ┌─────────────────┐
                             │  train_pairs.jsonl│
                             │  train_triplets.  │
                             │  val_*            │
                             └─────────────────┘
```

## Setup — 3 things to configure

### 1. CSV.GZ directory paths (lines 27-28)
```python
HOSP_DIR = Path("/path/to/mimic-iv/hosp")  # patients.csv.gz, diagnoses_icd.csv.gz, etc.
NOTE_DIR = Path("/path/to/mimic-iv/note")  # discharge.csv.gz, radiology.csv.gz
```

### 2. RAG patient exclusion — pick ONE method

**Option A: Query your PostgreSQL** (default)
```python
DB_CONFIG = {"host": "localhost", "port": 5432, ...}
RAG_PATIENTS_TABLE = "patients"
RAG_PATIENTS_ID_COL = "subject_id"
```

**Option B: Text file with one subject_id per line**
```python
DB_CONFIG = None
RAG_PATIENTS_FILE = Path("./rag_patient_ids.txt")
```

### 3. Tune sampling (optional)
```python
MAX_PATIENTS_FOR_TRAINING = None  # None = all remaining, or set a number
NEGATIVES_PER_QUERY = 5
MIN_CHUNK_LENGTH = 100
```

## Run
```bash
pip install psycopg2-binary tqdm
python mimic_reranker_csv_pipeline.py
```

## Key difference from the previous version

Since the training data comes from raw csv.gz (not your pre-chunked vector store),
the script **splits notes into sections** using clinical section headers
(History of Present Illness, Hospital Course, Findings, Impression, etc.)
as natural chunk boundaries. These section-level chunks are what the reranker
trains on.

## Output
All in `./reranker_training_data/`:

| File | Format | Use Case |
|---|---|---|
| `train_full.jsonl` | Full records + metadata | Debugging, quality review |
| `train_triplets.jsonl` | `{query, positive, negative}` | TripletLoss / MarginMSE |
| `train_pairs.jsonl` | `{query, document, label}` | CrossEncoder training |
| `val_*.jsonl` | Same formats | Validation (10% split) |

## After generating — quality check
```python
from datasets import load_dataset
ds = load_dataset("json", data_files="reranker_training_data/train_pairs.jsonl")
print(ds["train"][0])  # inspect a sample
```

Sample 200 pairs and score with an LLM to verify positive relevance ≥2.5/3
and negative relevance ≤1.0/3 before training.
