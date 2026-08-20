"""
Load the MIMIC-IV d_labitems dictionary into Postgres
=====================================================
labevents stores lab results by numeric `itemid` with no text label, so a
question about "potassium" has nothing to match against. This loads the small
d_labitems dictionary (itemid -> label/fluid/category, ~1,600 rows) that the
structured-labs path needs. Idempotent — safe to re-run.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.storage.load_d_labitems
    python -m src.storage.load_d_labitems --csv data/mimiciv/hosp/d_labitems.csv.gz
"""
from __future__ import annotations

import glob
import argparse
import logging

import pandas as pd
import sqlalchemy as sa

from src.storage import engine

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS d_labitems (
    itemid   INTEGER PRIMARY KEY,
    label    TEXT,
    fluid    TEXT,
    category TEXT
);
CREATE INDEX IF NOT EXISTS idx_dlabitems_label ON d_labitems (lower(label));
"""


def find_csv() -> str:
    hits = glob.glob("data/**/d_labitems*", recursive=True)
    if not hits:
        raise FileNotFoundError("d_labitems.csv(.gz) not found under data/ — pass --csv explicitly.")
    return hits[0]


def load(csv_path: str | None = None):
    csv_path = csv_path or find_csv()
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    cols = [c for c in ["itemid", "label", "fluid", "category"] if c in df.columns]
    df = df[cols].dropna(subset=["itemid"])
    df["itemid"] = df["itemid"].astype(int)
    # ensure the four columns exist even if the file lacks fluid/category
    for c in ["label", "fluid", "category"]:
        if c not in df.columns:
            df[c] = None
    df = df.where(pd.notnull(df), None)   # NaN -> None for clean inserts

    records = df[["itemid", "label", "fluid", "category"]].to_dict("records")

    with engine.begin() as conn:
        for stmt in DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(sa.text(stmt))
        conn.execute(sa.text("TRUNCATE d_labitems"))
        conn.execute(
            sa.text("INSERT INTO d_labitems (itemid, label, fluid, category) "
                    "VALUES (:itemid, :label, :fluid, :category) "
                    "ON CONFLICT (itemid) DO NOTHING"),
            records,
        )
        n = conn.execute(sa.text("SELECT COUNT(*) FROM d_labitems")).scalar()

    print(f"Loaded {n} lab item definitions from {csv_path}")
    # quick self-check on the labs we care about
    with engine.connect() as conn:
        for kw in ("potassium", "creatinine", "hemoglobin"):
            hits = conn.execute(
                sa.text("SELECT itemid, label, fluid FROM d_labitems "
                        "WHERE lower(label) LIKE :kw AND lower(fluid)='blood' ORDER BY itemid"),
                {"kw": f"%{kw}%"},
            ).fetchall()
            print(f"  {kw}: " + ", ".join(f"{r[0]}={r[1]}" for r in hits) or f"  {kw}: (none)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Load d_labitems dictionary into Postgres")
    ap.add_argument("--csv", default=None, help="Path to d_labitems.csv(.gz)")
    args = ap.parse_args()
    load(args.csv)
