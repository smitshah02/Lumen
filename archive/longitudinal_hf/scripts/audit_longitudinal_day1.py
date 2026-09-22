#!/usr/bin/env python3
"""
Lumen — Longitudinal HF Readmission Study: Day-1 Audit  (STRICTLY READ-ONLY)
===========================================================================
Verifies that the existing Lumen Postgres database can support the frozen
cohort/outcome specification in configs/longitudinal_hf.yaml, and writes
reports/longitudinal/day1_audit.md.

This script NEVER writes to the database. Three independent safeguards:
  1. the session is put into SESSION CHARACTERISTICS ... READ ONLY,
  2. every statement must start with SELECT/WITH,
  3. a deny-list regex rejects DML/DDL keywords before execution.

All counting happens in SQL. No table is ever pulled into Python to be counted.

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python scripts/audit_longitudinal_day1.py

    # override config / output, or fail the run on data-integrity warnings
    python scripts/audit_longitudinal_day1.py --config configs/longitudinal_hf.yaml
    python scripts/audit_longitudinal_day1.py --out reports/longitudinal/day1_audit.md
    python scripts/audit_longitudinal_day1.py --mimic-version 3.1
    python scripts/audit_longitudinal_day1.py --strict

Exit codes:
    0  audit completed, all required tables/columns present
    2  required table or column missing (report still written)
    3  database connection / configuration failure
    4  --strict was passed and data-integrity warnings were found
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Allow `python scripts/audit_longitudinal_day1.py` from the repo root to still
# import the existing `src.storage` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required (already a project dependency): pip install pyyaml")

from sqlalchemy import create_engine, text as sa_text

EXIT_OK, EXIT_SCHEMA, EXIT_DB, EXIT_WARNINGS = 0, 2, 3, 4

# Identifiers pulled from YAML are interpolated into SQL, so they are validated
# against these patterns first. Anything else aborts the run.
IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
ICD_PREFIX_RE = re.compile(r"^[A-Za-z0-9.]{1,7}$")

# Safeguard 3: deny-list applied to every statement before it reaches the driver.
WRITE_RE = re.compile(
    r"(?is)\b(insert|update|delete|drop|create|alter|truncate|grant|revoke|copy|merge|vacuum|refresh)\b"
)


# ===========================================================================
# Read-only DB access
# ===========================================================================

class ReadOnlyDB:
    """Thin read-only wrapper around the project's SQLAlchemy engine."""

    def __init__(self) -> None:
        self.engine, self.source = self._resolve_engine()
        # AUTOCOMMIT means no transaction is open, so the session-level
        # read-only characteristic applies to every statement that follows
        # and cannot be reset by an implicit rollback.
        self.conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        self.conn.exec_driver_sql("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")

    @staticmethod
    def _resolve_engine():
        """Reuse src.storage.engine (project convention); fall back to env vars."""
        try:
            from src.storage import engine  # type: ignore
            return engine, "src.storage.engine (project configuration)"
        except Exception:
            url = os.getenv("LUMEN_DATABASE_URL") or os.getenv("DATABASE_URL")
            if not url:
                user = os.getenv("POSTGRES_USER", "postgres")
                pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
                host = os.getenv("POSTGRES_HOST", "localhost")
                port = os.getenv("POSTGRES_PORT", "5432")
                db = os.getenv("POSTGRES_DB", "lumen")
                url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
            safe = re.sub(r"//[^:]+:[^@]*@", "//***:***@", url)
            return create_engine(url, pool_pre_ping=True), f"environment ({safe})"

    def _check(self, sql: str) -> None:
        head = sql.lstrip().lstrip("(").lstrip()
        if not re.match(r"(?is)^(select|with)\b", head):
            raise RuntimeError(f"Refused non-SELECT statement: {head[:60]!r}")
        if WRITE_RE.search(sql):
            raise RuntimeError(f"Refused statement containing a write keyword: {head[:60]!r}")

    def rows(self, sql: str, **params) -> list[dict]:
        self._check(sql)
        return [dict(r) for r in self.conn.execute(sa_text(sql), params).mappings().all()]

    def one(self, sql: str, **params) -> dict:
        res = self.rows(sql, **params)
        return res[0] if res else {}

    def scalar(self, sql: str, **params) -> Any:
        self._check(sql)
        return self.conn.execute(sa_text(sql), params).scalar()

    def close(self) -> None:
        self.conn.close()


# ===========================================================================
# Small formatting helpers
# ===========================================================================

def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def fmt(n: Optional[int]) -> str:
    return "n/a" if n is None else f"{n:,}"


def pct(num: Optional[int], den: Optional[int]) -> str:
    if not den or num is None:
        return "n/a"
    return f"{100.0 * num / den:.2f}%"


def check(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def safe_ident(name: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"Unsafe identifier in config: {name!r}")
    return name


# ===========================================================================
# SQL fragment builders (driven entirely by the frozen YAML config)
# ===========================================================================

def hf_predicate(cfg: dict) -> str:
    """Build the frozen HF diagnosis predicate. Alias for diagnoses_icd is `d`."""
    hf = cfg["cohort"]["heart_failure"]
    parts: list[str] = []
    for p in hf.get("icd10_prefixes", []):
        if not ICD_PREFIX_RE.match(p):
            raise ValueError(f"Unsafe ICD prefix: {p!r}")
        parts.append(f"(d.icd_version = 10 AND UPPER(TRIM(d.icd_code)) LIKE '{p.upper()}%')")
    for p in hf.get("icd9_prefixes", []):
        if not ICD_PREFIX_RE.match(p):
            raise ValueError(f"Unsafe ICD prefix: {p!r}")
        parts.append(f"(d.icd_version = 9 AND UPPER(TRIM(d.icd_code)) LIKE '{p.upper()}%')")
    if not parts:
        raise ValueError("No HF ICD prefixes configured.")
    return " OR ".join(parts)


def cte_blocks(cfg: dict) -> str:
    """
    Shared CTEs used by the cohort/outcome queries.

      seq     — every admission with LEAD(admittime) over the patient's own
                timeline. LEAD is computed over ALL admissions BEFORE any HF
                filtering, so the "next admission" is genuinely the next one
                (any cause), not the next HF one.
      hf_hadm — hadm_ids carrying a qualifying HF code (any diagnosis position).
      hf_adm  — HF admissions with their next-admission pointer.
      idx     — eligible index admissions after the frozen exclusions.
    """
    pred = hf_predicate(cfg)
    return f"""
seq AS (
    SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.deathtime,
           a.hospital_expire_flag,
           LEAD(a.admittime) OVER (
               PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
           ) AS next_admittime
    FROM admissions a
),
hf_hadm AS (
    SELECT DISTINCT d.hadm_id
    FROM diagnoses_icd d
    WHERE d.hadm_id IS NOT NULL AND ({pred})
),
hf_adm AS (
    SELECT s.* FROM seq s JOIN hf_hadm h ON h.hadm_id = s.hadm_id
),
idx AS (
    SELECT * FROM hf_adm
    WHERE admittime IS NOT NULL
      AND dischtime IS NOT NULL
      AND COALESCE(hospital_expire_flag, 0) = 0
      AND deathtime IS NULL
)"""


# ===========================================================================
# Audit sections
# ===========================================================================

def audit_schema(db: ReadOnlyDB, cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Verify required/expected tables and columns. Returns (md, errors, warnings)."""
    required = cfg["audit"]["required_tables"]
    expected = cfg["audit"].get("expected_tables", {}) or {}

    present = {
        (r["table_name"], r["column_name"])
        for r in db.rows("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
        """)
    }
    tables_present = {t for t, _ in present}

    md, errors, warnings = [], [], []
    rows = []
    for group, spec, hard in (("required", required, True), ("expected", expected, False)):
        for table, cols in spec.items():
            if table not in tables_present:
                status = "MISSING TABLE"
                missing_cols = cols
                (errors if hard else warnings).append(f"{group} table `{table}` is missing")
            else:
                missing_cols = [c for c in cols if (table, c) not in present]
                status = "OK" if not missing_cols else "MISSING COLUMNS"
                if missing_cols:
                    msg = f"{group} table `{table}` missing columns: {', '.join(missing_cols)}"
                    (errors if hard else warnings).append(msg)
            rows.append([table, group, status,
                         len(cols) - len(missing_cols), len(cols),
                         ", ".join(missing_cols) if missing_cols else "—"])

    md.append(md_table(
        ["table", "class", "status", "cols found", "cols expected", "missing"], rows))
    return md, errors, warnings


def audit_counts(db: ReadOnlyDB, cfg: dict) -> list[str]:
    """Exact row counts, computed in SQL."""
    tables = list(cfg["audit"]["required_tables"]) + list(cfg["audit"].get("expected_tables", {}))
    rows = []
    for t in tables:
        t = safe_ident(t)
        try:
            rows.append([t, fmt(db.scalar(f"SELECT COUNT(*) FROM {t}"))])
        except Exception as e:
            rows.append([t, f"error: {type(e).__name__}"])
    return [md_table(["table", "rows"], rows)]


def audit_cohort(db: ReadOnlyDB, cfg: dict) -> tuple[list[str], dict, list[str]]:
    ctes = cte_blocks(cfg)
    pred = hf_predicate(cfg)
    min_adm = int(cfg["cohort"]["index_admission"].get(
        "longitudinal_depth_probe_min_admissions", 3))

    core = db.one(f"""
WITH {ctes}
SELECT
    (SELECT COUNT(DISTINCT subject_id) FROM patients)                       AS all_patients,
    (SELECT COUNT(*) FROM admissions)                                       AS all_admissions,
    (SELECT COUNT(DISTINCT subject_id) FROM hf_adm)                         AS hf_patients,
    (SELECT COUNT(*) FROM hf_adm)                                           AS hf_admissions,
    (SELECT COUNT(*) FROM hf_adm
      WHERE COALESCE(hospital_expire_flag,0) = 1 OR deathtime IS NOT NULL)  AS excl_death,
    (SELECT COUNT(*) FROM hf_adm
      WHERE admittime IS NULL OR dischtime IS NULL)                         AS excl_times,
    (SELECT COUNT(*) FROM idx)                                              AS eligible_index,
    (SELECT COUNT(DISTINCT subject_id) FROM idx)                            AS eligible_patients
""")

    # Longitudinal depth: how many HF patients have enough history to be worth
    # modelling as a trajectory. Descriptive only — not an eligibility filter.
    depth = db.one(f"""
WITH hf_pat AS (
    SELECT DISTINCT d.subject_id FROM diagnoses_icd d WHERE {pred}
),
adm_cnt AS (
    SELECT a.subject_id, COUNT(*) AS n
    FROM admissions a JOIN hf_pat p ON p.subject_id = a.subject_id
    GROUP BY a.subject_id
)
SELECT COUNT(*)                                  AS hf_patients,
       COUNT(*) FILTER (WHERE n >= 2)            AS ge2,
       COUNT(*) FILTER (WHERE n >= :m)           AS ge_min,
       COUNT(*) FILTER (WHERE n >= 5)            AS ge5,
       ROUND(AVG(n)::numeric, 2)                 AS mean_adm,
       MAX(n)                                    AS max_adm
FROM adm_cnt
""", m=min_adm)

    # Cohort flow. The raw HF count and the eligible count are DIFFERENT stages
    # and each is compared against its own frozen reference - comparing the
    # post-exclusion cohort to the pre-exclusion count is a category error.
    ref = cfg.get("feasibility_reference", {}) or {}
    tol = float(ref.get("tolerance_pct", 1.0))
    warnings: list[str] = []

    def flow_row(label: str, got: int, exp: Optional[int]) -> list[Any]:
        if exp is None:
            return [label, fmt(got), "—", "—", "—"]
        delta = got - exp
        if exp == 0:
            ok, drift = got == 0, "n/a"
        else:
            drift_val = abs(delta) / exp * 100
            ok, drift = drift_val <= tol, f"{drift_val:.2f}%"
        if not ok:
            warnings.append(
                f"Cohort-flow drift: {label} = {got:,} vs reference {exp:,} (drift {drift}, tolerance {tol}%).")
        return [label, fmt(got), fmt(exp), f"{delta:+,}", f"{drift} {check(ok)}"]

    flow = [
        flow_row("Raw HF admissions (before exclusions)",
                 core["hf_admissions"], ref.get("raw_hf_admissions")),
        flow_row("Excluded — in-hospital death",
                 core["excl_death"], ref.get("excluded_in_hospital_death")),
        flow_row("Excluded — missing discharge time",
                 core["excl_times"], ref.get("excluded_missing_times")),
        flow_row("Final eligible readmission cohort",
                 core["eligible_index"], ref.get("eligible_hf_index_admissions")),
    ]

    # Arithmetic identity: raw - exclusions must reconcile to the eligible count.
    # A mismatch means an admission was caught by more than one exclusion.
    residual = core["hf_admissions"] - core["excl_death"] - core["excl_times"]
    if residual != core["eligible_index"]:
        warnings.append(
            f"Cohort flow does not reconcile: {core['hf_admissions']:,} - {core['excl_death']:,} - "
            f"{core['excl_times']:,} = {residual:,}, but the eligible cohort is "
            f"{core['eligible_index']:,}. Exclusion categories overlap.")

    md = [
        "**Cohort flow** — each stage is checked against its own frozen reference.\n",
        md_table(["stage", "observed", "reference", "delta", "drift"], flow),
        f"\nReconciliation: {fmt(core['hf_admissions'])} raw − {fmt(core['excl_death'])} in-hospital "
        f"death − {fmt(core['excl_times'])} missing discharge time = "
        f"**{fmt(residual)}** "
        f"({'reconciles' if residual == core['eligible_index'] else 'DOES NOT reconcile'} "
        f"with the eligible cohort of {fmt(core['eligible_index'])}).\n",
        "\n**Supporting counts**\n",
        md_table(["metric", "value"], [
            ["Patients in DB", fmt(core["all_patients"])],
            ["Admissions in DB", fmt(core["all_admissions"])],
            ["HF patients (any qualifying code)", fmt(core["hf_patients"])],
            ["Distinct patients contributing index admissions", fmt(core["eligible_patients"])],
            ["Index admissions per contributing patient (mean)",
             f"{core['eligible_index'] / core['eligible_patients']:.2f}"
             if core["eligible_patients"] else "n/a"],
        ]),
        "\n**Longitudinal depth of HF patients** (descriptive; not an eligibility filter)\n",
        md_table(["metric", "value"], [
            ["HF patients", fmt(depth["hf_patients"])],
            ["… with >= 2 admissions", f"{fmt(depth['ge2'])} ({pct(depth['ge2'], depth['hf_patients'])})"],
            [f"… with >= {min_adm} admissions",
             f"{fmt(depth['ge_min'])} ({pct(depth['ge_min'], depth['hf_patients'])})"],
            ["… with >= 5 admissions", f"{fmt(depth['ge5'])} ({pct(depth['ge5'], depth['hf_patients'])})"],
            ["Mean admissions per HF patient", depth["mean_adm"]],
            ["Max admissions for one HF patient", fmt(depth["max_adm"])],
        ]),
    ]
    return md, core, warnings


def audit_outcomes(db: ReadOnlyDB, cfg: dict, core: dict) -> tuple[list[str], list[str]]:
    ctes = cte_blocks(cfg)
    w90 = int(cfg["outcomes"]["primary"]["window_days"])
    w30 = int(cfg["outcomes"]["secondary"]["window_days"])
    warnings: list[str] = []

    # (0, W] days after discharge: strictly after dischtime, inclusive of the
    # window edge. make_interval keeps the windows config-driven.
    out = db.one(f"""
WITH {ctes}
SELECT
    COUNT(*)                                                       AS n_index,
    COUNT(*) FILTER (WHERE next_admittime IS NOT NULL)             AS has_next,
    COUNT(*) FILTER (WHERE next_admittime > dischtime
        AND next_admittime <= dischtime + make_interval(days => :w30))   AS r30,
    COUNT(*) FILTER (WHERE next_admittime > dischtime
        AND next_admittime <= dischtime + make_interval(days => :w90))   AS r90,
    COUNT(*) FILTER (WHERE next_admittime IS NOT NULL
        AND next_admittime <= dischtime)                            AS nonpositive_gap,
    ROUND(percentile_cont(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (next_admittime - dischtime))/86400.0
    ) FILTER (WHERE next_admittime > dischtime
        AND next_admittime <= dischtime + make_interval(days => :w90))::numeric, 1)
                                                                    AS median_days_r90
FROM idx
""", w30=w30, w90=w90)

    # Competing risk: patients who die inside the window without a recorded
    # readmission cannot contribute a positive outcome.
    comp = db.one(f"""
WITH {ctes}
SELECT COUNT(*) AS died_in_window_no_readmit
FROM idx i
JOIN patients p ON p.subject_id = i.subject_id
WHERE p.dod IS NOT NULL
  AND p.dod > i.dischtime
  AND p.dod <= i.dischtime + make_interval(days => :w90)
  AND (i.next_admittime IS NULL
       OR i.next_admittime > i.dischtime + make_interval(days => :w90))
""", w90=w90)

    n = out["n_index"]
    md = [
        f"Outcome = next recorded admission in **(0, W] days** after index discharge, "
        f"computed with `LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)` "
        f"over all admissions, then restricted to eligible index admissions.\n",
        md_table(["outcome", "n", "denominator", "rate"], [
            [f"readmit_{w30}d (secondary)", fmt(out["r30"]), fmt(n), pct(out["r30"], n)],
            [f"readmit_{w90}d (primary)", fmt(out["r90"]), fmt(n), pct(out["r90"], n)],
            ["any later admission (unbounded)", fmt(out["has_next"]), fmt(n), pct(out["has_next"], n)],
            [f"no readmission within {w90}d", fmt(n - out["r90"]), fmt(n), pct(n - out["r90"], n)],
        ]),
        f"\nMedian days to readmission among {w90}-day readmits: "
        f"**{out['median_days_r90']}**\n",
        f"\nCompeting risk — died within {w90}d of discharge with no recorded readmission: "
        f"**{fmt(comp['died_in_window_no_readmit'])}** "
        f"({pct(comp['died_in_window_no_readmit'], n)} of index admissions).\n",
    ]

    if out["nonpositive_gap"]:
        warnings.append(
            f"{out['nonpositive_gap']:,} index admissions have a next admittime <= dischtime "
            f"(overlapping/erroneous admissions). These are correctly scored as non-events by the "
            f"strict `> dischtime` bound but should be inspected on Day 2."
        )

    # Compare against the frozen feasibility reference.
    ref = cfg.get("feasibility_reference", {}) or {}
    if ref:
        tol = float(ref.get("tolerance_pct", 1.0))
        rows = []
        for label, got, exp in (
             (f"readmit_{w30}d n", out["r30"], ref.get("readmit_30d_n")),
             (f"readmit_{w90}d n", out["r90"], ref.get("readmit_90d_n")),
         ):
             if exp is None:
                 continue
             delta = got - exp
             drift = abs(delta) / exp * 100 if exp else 0.0
             ok = drift <= tol
             if not ok:
                 warnings.append(
                    f"Outcome drift: {label} = {got:,} vs reference {exp:,} ({drift:.2f}% > {tol}%)."
                 )
             rows.append([label, fmt(got), fmt(exp), f"{delta:+,}", f"{drift:.2f}%", check(ok)])
        md.append("\n**Outcome counts vs frozen reference** "
                  "(cohort size is reconciled in section 4)\n")
        md.append(md_table(["metric", "observed", "reference", "delta", "drift", "status"], rows))
        # Rates are reported on the post-exclusion denominator.
        for key, window, obs_n in (("readmit_30d_rate", w30, out["r30"]),
                                   ("readmit_90d_rate", w90, out["r90"])):
            exp_rate = ref.get(key)
            if exp_rate is not None and n:
                md.append(f"\n- `readmit_{window}d` rate: observed {pct(obs_n, n)} vs reference "
                          f"{100 * float(exp_rate):.2f}% (denominator = eligible cohort, not raw HF count).")
        md.append("\n")

    return md, warnings


def audit_integrity(db: ReadOnlyDB, cfg: dict) -> tuple[list[str], list[str]]:
    ctes = cte_blocks(cfg)
    warnings: list[str] = []

    bad = db.one("""
SELECT COUNT(*)                                                             AS n,
   COUNT(*) FILTER (WHERE dischtime < admittime)                            AS disch_before_admit,
   COUNT(*) FILTER (WHERE dischtime = admittime)                            AS zero_length_stay,
   COUNT(*) FILTER (WHERE deathtime IS NOT NULL AND deathtime < admittime)  AS death_before_admit,
   COUNT(*) FILTER (WHERE deathtime IS NOT NULL AND dischtime IS NOT NULL
                      AND deathtime > dischtime)                            AS death_after_disch,
   COUNT(*) FILTER (WHERE hospital_expire_flag = 1 AND deathtime IS NULL)   AS flag_no_deathtime,
   COUNT(*) FILTER (WHERE COALESCE(hospital_expire_flag,0) = 0
                      AND deathtime IS NOT NULL)                            AS deathtime_no_flag,
   COUNT(*) FILTER (WHERE dischtime > admittime + make_interval(days => 365)) AS los_over_1y
FROM admissions
""")

    overlap = db.one(f"""
WITH {ctes}
SELECT COUNT(*) AS overlapping
FROM seq
WHERE next_admittime IS NOT NULL AND dischtime IS NOT NULL
  AND next_admittime < dischtime
""")

    # dod in MIMIC is date-granular, so allow a 1-day tolerance before flagging.
    dod = db.one("""
SELECT COUNT(*) AS dod_before_last_discharge
FROM (
    SELECT a.subject_id, MAX(a.dischtime) AS last_disch
    FROM admissions a GROUP BY a.subject_id
) x
JOIN patients p ON p.subject_id = x.subject_id
WHERE p.dod IS NOT NULL AND x.last_disch IS NOT NULL
  AND p.dod < x.last_disch - make_interval(days => 1)
""")

    checks = [
        ["dischtime < admittime", bad["disch_before_admit"], "must be 0"],
        ["dischtime = admittime (zero-length stay)", bad["zero_length_stay"], "review"],
        ["deathtime < admittime", bad["death_before_admit"], "must be 0"],
        ["deathtime > dischtime", bad["death_after_disch"], "review"],
        ["hospital_expire_flag=1 but deathtime NULL", bad["flag_no_deathtime"], "review"],
        ["deathtime present but expire_flag=0", bad["deathtime_no_flag"], "review"],
        ["length of stay > 365 days", bad["los_over_1y"], "review"],
        ["next admittime < current dischtime (overlap)", overlap["overlapping"], "review"],
        ["dod earlier than last discharge (>1d)", dod["dod_before_last_discharge"], "review"],
    ]
    rows = []
    for label, n, expectation in checks:
        status = "PASS" if n == 0 else ("FAIL" if expectation == "must be 0" else "WARN")
        rows.append([label, fmt(n), expectation, status])
        if status == "FAIL":
            warnings.append(f"Impossible dates: {label} affects {n:,} admissions.")
        elif status == "WARN" and n:
            warnings.append(f"Date anomaly: {label} affects {n:,} admissions.")

    return [md_table(["check", "rows affected", "expectation", "status"], rows)], warnings


def audit_missingness(db: ReadOnlyDB, cfg: dict) -> list[str]:
    rows = []
    for table, cols in (cfg["audit"].get("missingness_columns", {}) or {}).items():
        table = safe_ident(table)
        # One pass per table: NULL counts for every configured column at once.
        exprs = ", ".join(
            f"SUM(CASE WHEN {safe_ident(c)} IS NULL THEN 1 ELSE 0 END) AS n_{safe_ident(c)}"
            for c in cols
        )
        try:
            res = db.one(f"SELECT COUNT(*) AS total, {exprs} FROM {table}")
        except Exception as e:
            rows.append([table, "—", f"error: {type(e).__name__}", "", ""])
            continue
        total = res["total"]
        for c in cols:
            nulls = res[f"n_{c}"] or 0
            rows.append([table, c, fmt(total), fmt(nulls), pct(nulls, total)])
    return [md_table(["table", "column", "rows", "nulls", "% missing"], rows)]


def audit_mortality(db: ReadOnlyDB, cfg: dict) -> list[str]:
    ctes = cte_blocks(cfg)
    m = db.one(f"""
WITH {ctes}
SELECT
    (SELECT COUNT(*) FROM patients)                                        AS n_patients,
    (SELECT COUNT(dod) FROM patients)                                      AS n_dod,
    (SELECT COUNT(*) FROM admissions WHERE deathtime IS NOT NULL)          AS n_deathtime,
    (SELECT COUNT(*) FROM admissions WHERE hospital_expire_flag = 1)       AS n_expire_flag,
    (SELECT COUNT(*) FROM admissions WHERE hospital_expire_flag IS NULL)   AS n_flag_null,
    (SELECT COUNT(*) FROM hf_adm WHERE COALESCE(hospital_expire_flag,0)=1) AS hf_inhospital_deaths
""")
    n_p = m["n_patients"]
    return [md_table(["field", "source", "n", "coverage"], [
        ["dod (date of death)", "patients.dod", fmt(m["n_dod"]), pct(m["n_dod"], n_p) + " of patients"],
        ["deathtime", "admissions.deathtime", fmt(m["n_deathtime"]), "admissions with in-hospital death time"],
        ["hospital_expire_flag = 1", "admissions", fmt(m["n_expire_flag"]), "admissions flagged as died"],
        ["hospital_expire_flag IS NULL", "admissions", fmt(m["n_flag_null"]), "must be 0 for a clean exclusion"],
        ["HF admissions excluded for in-hospital death", "derived", fmt(m["hf_inhospital_deaths"]), "—"],
    ])]


def audit_labs(db: ReadOnlyDB, cfg: dict, have_labs: bool) -> tuple[list[str], list[str]]:
    """Discover lab itemids by label and measure pre-cutoff coverage. No features built."""
    if not have_labs:
        return ["_labevents/d_labitems not available — lab audit skipped._\n"], [
            "labevents or d_labitems missing: all lab candidate variables are unconfirmed."
        ]

    ctes = cte_blocks(cfg)
    concepts: dict[str, list[str]] = cfg["audit"]["lab_concepts"]
    md, warnings = [], []
    summary_rows, detail_rows = [], []

    for concept, patterns in concepts.items():
        where = " OR ".join(f"label ILIKE :p{i}" for i in range(len(patterns)))
        params = {f"p{i}": p for i, p in enumerate(patterns)}
        items = db.rows(
            f"SELECT itemid, label, fluid, category FROM d_labitems WHERE {where} ORDER BY label",
            **params)
        if not items:
            summary_rows.append([concept, "0", "—", "—", "—", "NOT FOUND"])
            warnings.append(f"No d_labitems label matched concept '{concept}'.")
            continue

        # itemids come straight from the DB; int() makes the inlining injection-safe.
        ids = ",".join(str(int(i["itemid"])) for i in items)

        vol = db.rows(f"""
            SELECT itemid, COUNT(*) AS n_rows, COUNT(valuenum) AS n_numeric
            FROM labevents WHERE itemid IN ({ids}) GROUP BY itemid
        """)
        vol_map = {v["itemid"]: v for v in vol}

        # Coverage = eligible index admissions with >=1 numeric result timestamped
        # at or before the discharge cutoff (i.e. legally usable as a predictor).
        cov = db.one(f"""
WITH {ctes}
SELECT COUNT(*) AS n_covered, (SELECT COUNT(*) FROM idx) AS n_index
FROM idx i
WHERE EXISTS (
    SELECT 1 FROM labevents l
    WHERE l.subject_id = i.subject_id
      AND l.itemid IN ({ids})
      AND l.charttime IS NOT NULL
      AND l.charttime <= i.dischtime
      AND l.valuenum IS NOT NULL
)
""")
        total_rows = sum(v["n_rows"] for v in vol)
        total_num = sum(v["n_numeric"] for v in vol)
        summary_rows.append([
            concept, len(items), fmt(total_rows), pct(total_num, total_rows),
            f"{fmt(cov['n_covered'])} ({pct(cov['n_covered'], cov['n_index'])})",
            "OK" if cov["n_covered"] else "NO PRE-CUTOFF DATA",
        ])
        if not cov["n_covered"]:
            warnings.append(f"Concept '{concept}' has no pre-cutoff numeric values for the cohort.")

        for it in items[:6]:  # cap the detail table; full list is reproducible from the query
            v = vol_map.get(it["itemid"], {})
            detail_rows.append([concept, it["itemid"], it["label"], it["fluid"],
                                it["category"], fmt(v.get("n_rows", 0)), fmt(v.get("n_numeric", 0))])

    md.append(md_table(
        ["concept", "itemids matched", "lab rows", "% numeric",
         "index admissions with >=1 pre-cutoff value", "status"], summary_rows))
    md.append("\n**Matched itemids (first 6 per concept)**\n")
    md.append(md_table(
        ["concept", "itemid", "label", "fluid", "category", "rows", "numeric rows"], detail_rows))
    return md, warnings


def audit_medications(db: ReadOnlyDB, cfg: dict, have_rx: bool) -> tuple[list[str], list[str]]:
    """Availability of medication NAMES only. No medication features engineered."""
    if not have_rx:
        return ["_prescriptions not available — medication audit skipped._\n"], [
            "prescriptions missing: all medication candidate variables are unconfirmed."
        ]

    ctes = cte_blocks(cfg)
    warnings: list[str] = []

    base = db.one("""
SELECT COUNT(*) AS n_rows,
       COUNT(DISTINCT drug) AS n_distinct_drug,
       SUM(CASE WHEN drug IS NULL OR TRIM(drug) = '' THEN 1 ELSE 0 END) AS n_missing_drug,
       SUM(CASE WHEN starttime IS NULL THEN 1 ELSE 0 END)  AS n_missing_starttime,
       SUM(CASE WHEN route IS NULL THEN 1 ELSE 0 END)      AS n_missing_route,
       SUM(CASE WHEN dose_val_rx IS NULL THEN 1 ELSE 0 END) AS n_missing_dose
FROM prescriptions
""")

    cov = db.one(f"""
WITH {ctes}
SELECT (SELECT COUNT(*) FROM idx) AS n_index,
       COUNT(*) AS n_covered
FROM idx i
WHERE EXISTS (
    SELECT 1 FROM prescriptions r
    WHERE r.hadm_id = i.hadm_id
      AND r.starttime IS NOT NULL
      AND r.starttime <= i.dischtime
)
""")

    md = [md_table(["metric", "value"], [
        ["Prescription rows", fmt(base["n_rows"])],
        ["Distinct drug names", fmt(base["n_distinct_drug"])],
        ["Missing drug name", f"{fmt(base['n_missing_drug'])} ({pct(base['n_missing_drug'], base['n_rows'])})"],
        ["Missing starttime", f"{fmt(base['n_missing_starttime'])} ({pct(base['n_missing_starttime'], base['n_rows'])})"],
        ["Missing route", f"{fmt(base['n_missing_route'])} ({pct(base['n_missing_route'], base['n_rows'])})"],
        ["Missing dose_val_rx", f"{fmt(base['n_missing_dose'])} ({pct(base['n_missing_dose'], base['n_rows'])})"],
        ["Index admissions with >=1 pre-cutoff prescription",
         f"{fmt(cov['n_covered'])} ({pct(cov['n_covered'], cov['n_index'])})"],
    ])]

    if cov["n_covered"] < 0.5 * cov["n_index"]:
        warnings.append("Fewer than half of index admissions have a pre-cutoff prescription record.")

    # Single pass over prescriptions for all HF-relevant name probes.
    probes: list[str] = cfg["audit"].get("medication_probe_names", []) or []
    if probes:
        exprs = ", ".join(
            f"SUM(CASE WHEN drug ILIKE :q{i} THEN 1 ELSE 0 END) AS c{i}" for i in range(len(probes)))
        params = {f"q{i}": p for i, p in enumerate(probes)}
        hits = db.one(f"SELECT {exprs} FROM prescriptions", **params)
        rows = [[p, fmt(hits[f"c{i}"] or 0), "present" if (hits[f"c{i}"] or 0) else "ABSENT"]
                for i, p in enumerate(probes)]
        md.append("\n**HF-relevant drug-name probes** (name availability only — no features built)\n")
        md.append(md_table(["name pattern", "rows matched", "status"], rows))

    limit = int(cfg["audit"].get("top_drug_names_limit", 20))
    top = db.rows(f"""
WITH {ctes}
SELECT LOWER(TRIM(r.drug)) AS drug, COUNT(*) AS n
FROM prescriptions r JOIN idx i ON i.hadm_id = r.hadm_id
WHERE r.drug IS NOT NULL AND TRIM(r.drug) <> ''
GROUP BY 1 ORDER BY n DESC LIMIT :lim
""", lim=limit)
    md.append(f"\n**Top {limit} drug names within eligible index admissions**\n")
    md.append(md_table(["drug", "rows"], [[t["drug"], fmt(t["n"])] for t in top]))
    return md, warnings


def audit_notes(db: ReadOnlyDB, cfg: dict, have_notes: bool) -> list[str]:
    if not have_notes:
        return ["_clinical_notes not available — note audit skipped._\n"]
    ctes = cte_blocks(cfg)
    n = db.one(f"""
WITH {ctes}
SELECT (SELECT COUNT(*) FROM clinical_notes)                       AS n_notes,
       (SELECT COUNT(*) FROM clinical_notes WHERE charttime IS NULL) AS n_no_charttime,
       (SELECT COUNT(*) FROM idx)                                  AS n_index,
       (SELECT COUNT(*) FROM idx i WHERE EXISTS (
            SELECT 1 FROM clinical_notes c
            WHERE c.subject_id = i.subject_id
              AND c.charttime IS NOT NULL AND c.charttime <= i.dischtime)) AS n_covered
""")
    by_type = db.rows(
        "SELECT note_type, COUNT(*) AS n FROM clinical_notes GROUP BY 1 ORDER BY n DESC")
    return [
        md_table(["metric", "value"], [
            ["Clinical notes", fmt(n["n_notes"])],
            ["Notes with NULL charttime (unusable under the cutoff rule)",
             f"{fmt(n['n_no_charttime'])} ({pct(n['n_no_charttime'], n['n_notes'])})"],
            ["Index admissions with >=1 pre-cutoff note",
             f"{fmt(n['n_covered'])} ({pct(n['n_covered'], n['n_index'])})"],
        ]),
        "\n", md_table(["note_type", "rows"], [[r["note_type"], fmt(r["n"])] for r in by_type]),
    ]


# ===========================================================================
# Report assembly
# ===========================================================================

def build_report(cfg: dict, db_source: str, sections: dict[str, list[str]],
                 errors: list[str], warnings: list[str], mimic_version: str) -> str:
    o = cfg["outcomes"]
    hf = cfg["cohort"]["heart_failure"]
    out: list[str] = []
    A = out.append

    A("# Lumen — Longitudinal HF Readmission: Day-1 Audit\n")
    A(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- Config: `configs/longitudinal_hf.yaml` (version {cfg['version']}, frozen={cfg['frozen']})")
    A(f"- Connection: {db_source}")
    A(f"- Database: `{cfg['data_source']['database']}` in container `{cfg['data_source']['container']}`")
    A(f"- **MIMIC version: `{mimic_version}`**"
      + ("" if cfg["data_source"].get("mimic_version_confirmed")
         else "  _(not manually confirmed — must be resolved before results are reported)_"))
    A("- Access mode: **READ-ONLY** (session read-only + SELECT/WITH-only statement guard)\n")
    A(f"- Result: **{'FAILED — required schema missing' if errors else 'PASSED'}** "
      f"({len(errors)} error(s), {len(warnings)} warning(s))\n")

    A("## 1. Database & schema availability\n")
    out += sections["schema"]
    A("\n### Row counts\n")
    out += sections["counts"]

    A("\n## 2. Cohort definition (FROZEN)\n")
    A(f"- **Heart failure**: `{hf['source_column']}` starting with "
      f"{', '.join(hf['icd10_prefixes'])} (ICD-10) or {', '.join(hf['icd9_prefixes'])} (ICD-9), "
      f"matched on `{hf['normalize']}`, any diagnosis position (`seq_num` unrestricted).")
    A(f"- **Index admission**: {cfg['cohort']['index_admission']['definition']}")
    A("- **Exclusions**: " + "; ".join(
        f"`{e['rule']}`" for e in cfg["cohort"]["index_admission"]["exclusions"]))
    A(f"- **Unit of analysis**: {cfg['study']['unit_of_analysis']} "
      f"(a patient may contribute more than one index admission).\n")

    A("## 3. Outcome definition (FROZEN)\n")
    A(f"- **Primary** `{o['primary']['name']}`: next recorded hospital admission > 0 and "
      f"<= {o['primary']['window_days']} days after index `dischtime`.")
    A(f"- **Secondary** `{o['secondary']['name']}`: same rule at "
      f"<= {o['secondary']['window_days']} days.")
    A(f"- **Prediction cutoff**: `{cfg['prediction']['cutoff_column']}`.")
    A("- **Next admission scope**: any cause (the next admission need not carry an HF code).\n")

    A("## 4. Cohort flow and counts\n")
    out += sections["cohort"]

    A("\n## 5. Outcome counts and rates\n")
    out += sections["outcomes"]

    A("\n## 6. Data availability checklist\n")
    A("### 6.1 Date integrity\n")
    out += sections["integrity"]
    A("\n### 6.2 Missingness in key outcome/time fields\n")
    out += sections["missingness"]
    A("\n### 6.3 Mortality fields\n")
    out += sections["mortality"]
    A("\n### 6.4 Clinical notes\n")
    out += sections["notes"]

    A("\n## 7. Candidate-variable availability\n")
    A("### 7.1 Laboratory concepts\n")
    out += sections["labs"]
    A("\n### 7.2 Medications (names only — no features engineered at Day 1)\n")
    out += sections["meds"]
    A("\nFull candidate list with roles, timestamps and leakage rules: "
      "`reports/longitudinal/data_dictionary.csv`.\n")

    A("\n## 8. Leakage rules (FROZEN)\n")
    out.append(md_table(["id", "rule"], [
        [r["id"], " ".join(r["rule"].split())] for r in cfg["leakage_rules"]]))
    sp = cfg["splitting"]
    A(f"\n**Split rule**: patient-level on `{sp['group_key_column']}` "
      f"({sp['proportions']['train']}/{sp['proportions']['valid']}/{sp['proportions']['test']}, "
      f"seed {sp['seed']}, stratified on {', '.join(sp['stratify_on'])}). "
      "Admission-level splitting is prohibited.\n")

    A("\n## 9. Warnings and limitations\n")
    if errors:
        A("### Errors (blocking)\n")
        for e in errors:
            A(f"- **{e}**")
        A("")
    A("### Warnings\n")
    if warnings:
        for w in warnings:
            A(f"- {w}")
    else:
        A("- None raised by this run.")
    A("""
### Standing limitations

- **Date shifting.** MIMIC shifts each patient's timeline by a random per-patient
  offset. Within-patient intervals are valid; absolute calendar time, `anchor_year`
  and `anchor_year_group` must never be used as population-level temporal features.
- **Observability.** Readmissions to hospitals outside this database are invisible,
  so the observed rates are lower bounds and the negative class is contaminated.
- **Competing risk.** Post-discharge death censors the readmission outcome; the
  count is reported in section 5.
- **Discharge-coded diagnoses.** ICD codes are finalized at discharge, so index-
  admission comorbidities are only cutoff-safe under the convention in LR9.
- **Cohort size.** This is a 5,000-patient development subset, not full MIMIC-IV;
  rates here should not be read as population estimates.
- **Repeated measures.** Patients contribute multiple index admissions, so the
  patient-level split in section 8 is mandatory, not a preference.
- **Not clinical decision support.** Retrospective research/portfolio work only.
""")
    A(f"\n## 10. MIMIC version\n\n`{mimic_version}`"
      + ("  — confirmed." if cfg["data_source"].get("mimic_version_confirmed")
         else "  — **unconfirmed.** " + " ".join(
             cfg["data_source"].get("mimic_version_source_note", "").split())))
    A("\n---\n_Generated by `scripts/audit_longitudinal_day1.py` (read-only)._")
    return "\n".join(out) + "\n"


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Day-1 read-only audit for the Lumen longitudinal HF layer")
    ap.add_argument("--config", default="configs/longitudinal_hf.yaml")
    ap.add_argument("--out", default=None, help="override report path from config")
    ap.add_argument("--mimic-version", default=None, help="report-only override of data_source.mimic_version")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any warning is raised")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return EXIT_DB
    cfg = yaml.safe_load(cfg_path.read_text())

    mimic_version = args.mimic_version or cfg["data_source"].get("mimic_version") or "unknown"

    try:
        db = ReadOnlyDB()
        db.scalar("SELECT 1")
    except Exception as e:
        print(f"ERROR: cannot connect to the Lumen database: {e}", file=sys.stderr)
        print("Is the lumen-pg container running? (docker ps)", file=sys.stderr)
        return EXIT_DB

    print(f"Connected via {db.source} — running READ-ONLY Day-1 audit...")
    sections: dict[str, list[str]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    try:
        print("  [1/9] schema check")
        sections["schema"], errors, warnings = audit_schema(db, cfg)

        print("  [2/9] row counts")
        sections["counts"] = audit_counts(db, cfg)

        if errors:
            # Cohort logic depends on the required tables; stop the SQL work but
            # still emit a report so the failure is documented.
            for key in ("cohort", "outcomes", "integrity", "missingness",
                        "mortality", "notes", "labs", "meds"):
                sections[key] = ["_skipped: required schema missing._\n"]
        else:
            present = {r["table_name"] for r in db.rows(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}

            print("  [3/9] cohort counts")
            sections["cohort"], core, w = audit_cohort(db, cfg)
            warnings += w

            print("  [4/9] outcomes (LEAD-based)")
            sections["outcomes"], w = audit_outcomes(db, cfg, core)
            warnings += w

            print("  [5/9] date integrity")
            sections["integrity"], w = audit_integrity(db, cfg)
            warnings += w

            print("  [6/9] missingness")
            sections["missingness"] = audit_missingness(db, cfg)

            print("  [7/9] mortality fields")
            sections["mortality"] = audit_mortality(db, cfg)
            sections["notes"] = audit_notes(db, cfg, "clinical_notes" in present)

            print("  [8/9] lab concept availability")
            sections["labs"], w = audit_labs(
                db, cfg, "labevents" in present and "d_labitems" in present)
            warnings += w

            print("  [9/9] medication name availability")
            sections["meds"], w = audit_medications(db, cfg, "prescriptions" in present)
            warnings += w
    finally:
        db.close()

    out_path = Path(args.out or cfg["audit"]["report_path"])
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(cfg, db.source, sections, errors, warnings, mimic_version))

    print(f"\nReport written to {out_path}")
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN:  {w}")

    if errors:
        return EXIT_SCHEMA
    if warnings and args.strict:
        return EXIT_WARNINGS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
