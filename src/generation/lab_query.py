"""
Structured lab retrieval for Lumen  (labs-only quantitative path)
================================================================
Turns a quantitative question ("potassium values", "how did creatinine
change") into a dated lab series pulled straight from `labevents`, so the
generator can answer numeric/trend questions the notes-RAG can't.

The router is deterministic and explainable: a small, curated synonym map
turns question phrases into d_labitems label keywords, which resolve to
itemids. You can always see exactly which analytes a question matched — no
opaque model call in the routing decision.

Public API:
    resolver = LabResolver()                       # loads d_labitems once
    itemids, matched = resolver.match(question)     # router (empty -> not a lab Q)
    rows = resolver.fetch(subject_id, itemids, hadm_id=None)
    context, source_index = render_lab_context(rows)

    # one-shot used by the generator:
    context, source_index, matched = resolver.labs_for_question(
        question, subject_id, hadm_id=None)
"""
from __future__ import annotations

import re
import logging
from collections import OrderedDict

import sqlalchemy as sa

from src.storage import engine

logger = logging.getLogger(__name__)

# Curated synonym map: question phrase -> d_labitems.label keyword(s) to match
# (case-insensitive substring, fluid='Blood' preferred). Explicit on purpose —
# this is the whole routing logic, and it's auditable. Add rows as you find gaps.
# NOTE: deliberately no bare "pt"/"ph" keys — "pt" collides with "patient".
SYNONYMS: dict[str, list[str]] = {
    "potassium": ["potassium"],
    "sodium": ["sodium"],
    "chloride": ["chloride"],
    "bicarbonate": ["bicarbonate"],
    "magnesium": ["magnesium"],
    "calcium": ["calcium"],
    "phosphate": ["phosphate"],
    "phosphorus": ["phosphate"],
    "glucose": ["glucose"],
    "blood sugar": ["glucose"],
    "creatinine": ["creatinine"],
    "bun": ["urea nitrogen"],
    "urea": ["urea nitrogen"],
    "kidney function": ["creatinine", "urea nitrogen"],
    "renal function": ["creatinine", "urea nitrogen"],
    "hemoglobin": ["hemoglobin"],
    "haemoglobin": ["hemoglobin"],
    "hematocrit": ["hematocrit"],
    "platelet": ["platelet count"],
    "platelets": ["platelet count"],
    "white blood cell": ["white blood cells"],
    "white blood cells": ["white blood cells"],
    "wbc": ["white blood cells"],
    "red blood cell": ["red blood cells"],
    "rbc": ["red blood cells"],
    "blood count": ["hemoglobin", "hematocrit", "white blood cells", "platelet count"],
    "cbc": ["hemoglobin", "hematocrit", "white blood cells", "platelet count"],
    "anemia": ["hemoglobin", "hematocrit"],
    "inr": ["inr"],
    "ptt": ["ptt"],
    "lactate": ["lactate"],
    "albumin": ["albumin"],
    "bilirubin": ["bilirubin"],
    "alt": ["alanine aminotransferase"],
    "ast": ["aspartate aminotransferase", "asparate aminotransferase"],  # MIMIC misspells one
    "alkaline phosphatase": ["alkaline phosphatase"],
    "liver function": ["alanine aminotransferase", "aspartate aminotransferase",
                       "asparate aminotransferase", "alkaline phosphatase", "bilirubin"],
    "lft": ["alanine aminotransferase", "aspartate aminotransferase",
            "asparate aminotransferase", "alkaline phosphatase", "bilirubin"],
    "troponin": ["troponin"],
    "lipase": ["lipase"],
    "cholesterol": ["cholesterol"],
}


def _fmt_num(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _mentions(question_lc: str, phrase: str) -> bool:
    """Whole-word / whole-phrase match, so 'wbc' doesn't fire inside another word."""
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", question_lc) is not None


class LabResolver:
    def __init__(self):
        # (itemid, label_lower, fluid_lower) — small table, load once into memory.
        self._items: list[tuple[int, str, str]] = []
        with engine.connect() as c:
            for itemid, label, fluid in c.execute(
                sa.text("SELECT itemid, label, fluid FROM d_labitems")
            ):
                self._items.append((int(itemid), (label or "").lower(), (fluid or "").lower()))
        logger.info("LabResolver loaded %d lab definitions", len(self._items))

    def _keyword_to_itemids(self, keyword: str) -> list[int]:
        kw = keyword.lower()
        blood = [i for (i, lab, fl) in self._items if kw in lab and fl == "blood"]
        if blood:
            return blood
        # fall back to any fluid if there's no blood specimen for this analyte
        return [i for (i, lab, fl) in self._items if kw in lab]

    def match(self, question: str) -> tuple[list[int], list[str]]:
        """Router. Returns (itemids, matched_concept_labels). Empty -> not a lab question."""
        q = question.lower()
        itemids: list[int] = []
        matched: list[str] = []
        for concept, keywords in SYNONYMS.items():
            if _mentions(q, concept):
                matched.append(concept)
                for kw in keywords:
                    itemids.extend(self._keyword_to_itemids(kw))
        # dedupe, preserve order
        seen = set()
        itemids = [i for i in itemids if not (i in seen or seen.add(i))]
        return itemids, sorted(set(matched))

    def fetch(self, subject_id: int, itemids: list[int], hadm_id: int | None = None,
              per_lab_cap: int = 80) -> list[dict]:
        if not itemids:
            return []
        hadm_clause = "AND l.hadm_id = :hadm" if hadm_id is not None else ""
        stmt = sa.text(f"""
            SELECT d.label, d.itemid, l.charttime, l.valuenum, l.valueuom, l.flag
            FROM labevents l
            JOIN d_labitems d ON d.itemid = l.itemid
            WHERE l.subject_id = :sid
              AND l.itemid IN :itemids
              AND l.valuenum IS NOT NULL
              {hadm_clause}
            ORDER BY d.label, l.charttime
        """).bindparams(sa.bindparam("itemids", expanding=True))
        params = {"sid": subject_id, "itemids": list(itemids)}
        if hadm_id is not None:
            params["hadm"] = hadm_id

        with engine.connect() as c:
            raw = c.execute(stmt, params).fetchall()

        # group by label, cap per analyte to protect the context window
        grouped: "OrderedDict[str, list]" = OrderedDict()
        for label, itemid, charttime, valuenum, uom, flag in raw:
            grouped.setdefault(label, []).append({
                "charttime": str(charttime),
                "date": str(charttime)[:10],
                "valuenum": valuenum,
                "uom": uom,
                "abnormal": bool(flag and str(flag).strip()),
            })

        out = []
        for label, vals in grouped.items():
            capped = vals[:per_lab_cap]
            out.append({
                "label": label,
                "uom": next((v["uom"] for v in capped if v["uom"]), ""),
                "values": capped,
                "n_total": len(vals),
                "n_shown": len(capped),
                "n_abnormal": sum(1 for v in vals if v["abnormal"]),
            })
        return out

    def labs_for_question(self, question: str, subject_id: int, hadm_id: int | None = None):
        """Router + fetch + render. Returns (context, source_index, matched_concepts)."""
        itemids, matched = self.match(question)
        if not itemids:
            return "", [], []            # not a lab question at all
        series = self.fetch(subject_id, itemids, hadm_id=hadm_id)
        if not series:
            return "", [], matched       # lab question, but this patient has no such labs
        context, source_index = render_lab_context(series)
        return context, source_index, matched


def render_lab_context(series: list[dict]) -> tuple[str, list[dict]]:
    """Render each analyte as one [L#] source with its full dated series."""
    lines: list[str] = []
    source_index: list[dict] = []

    for grp in series:
        tag = f"L{len(source_index) + 1}"
        uom = f" {grp['uom']}" if grp["uom"] else ""
        points = "; ".join(
            f"{v['date']} {_fmt_num(v['valuenum'])}{uom}" + (" (abnormal)" if v["abnormal"] else "")
            for v in grp["values"]
        )
        more = "" if grp["n_shown"] >= grp["n_total"] else f" (+{grp['n_total'] - grp['n_shown']} more not shown)"
        header = (f"[{tag}] {grp['label']} — {grp['n_total']} value(s), "
                  f"{grp['n_abnormal']} abnormal:")
        lines.append(f"{header}\n{points}{more}")
        source_index.append({
            "tag": tag,
            "kind": "lab",
            "label": grp["label"],
            "uom": grp["uom"],
            "n_values": grp["n_total"],
            "n_abnormal": grp["n_abnormal"],
        })

    return "\n\n".join(lines), source_index
