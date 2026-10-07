"""Section labels for discharge-summary chunks: search metadata only.

The chunker writes one "[SECTION]" prefix per chunk, and that prefix is often
wrong for search:
  * de-identification rewrites the "Brief Hospital Course:" header, so the
    course inherits whatever header came before it ("[IMPRESSION]", "[AP]");
  * short sections are merged into the previous chunk and lose their header,
    so a discharge diagnosis is filed under "[Discharge Disposition]";
  * "Major Surgical or Invasive Procedure" and "Pertinent Results" are not
    headers the chunker knows.

This module reads the standard headers from the note as written and records,
per chunk, which sections it actually covers. The result is stored in
chunk_search_labels and used by lexical ranking only. chunk_text, the
embedding and everything shown to a user are untouched. Labels come from a
fixed list of template headers, so no note content is copied into them.

    python -m src.retrieval.section_labels            # label chunks that have none
    python -m src.retrieval.section_labels --all      # relabel everything

ponytail: discharge summaries only. Radiology notes have no stable exam-title
line (45% carry "EXAMINATION:"); add when a reliable source for it exists.
"""
from __future__ import annotations

import argparse
import bisect
import re

# header pattern -> label. The MIMIC-IV discharge summary template, in note order.
STANDARD_HEADERS = (
    (r"allergies", "Allergies"),
    (r"chief complaint", "Chief Complaint"),
    (r"major surgical or invasive procedure", "Major Surgical or Invasive Procedure"),
    (r"history of present illness", "History of Present Illness"),
    (r"past medical history", "Past Medical History"),
    (r"social history", "Social History"),
    (r"family history", "Family History"),
    (r"physical exam(?:ination)?", "Physical Exam"),
    (r"pertinent results", "Pertinent Results"),
    (r"brief hospital course", "Brief Hospital Course"),
    (r"medications on admission", "Medications on Admission"),
    (r"discharge medications", "Discharge Medications"),
    (r"discharge disposition", "Discharge Disposition"),
    (r"discharge diagnos[ie]s", "Discharge Diagnosis"),
    (r"discharge condition", "Discharge Condition"),
    (r"discharge instructions", "Discharge Instructions"),
    (r"follow\s*-?\s*up instructions", "Followup Instructions"),
)
_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:" + "|".join(f"(?P<h{i}>{p})" for i, (p, _) in enumerate(STANDARD_HEADERS)) + r")[ \t]*:")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_WINDOW = 6          # consecutive words used to find a chunk inside its note


def label_chunks(note_text: str, chunk_texts: list[str]) -> list[str | None]:
    """One label string per chunk ("Discharge Disposition; Discharge Diagnosis"), or None.

    A chunk covers a section when half the chunk lies in it, or half the section
    lies in the chunk. Chunk text is not a verbatim slice of the note (the "[X]"
    prefix, de-identification), so each chunk is located by a run of six
    consecutive words. A chunk that cannot be located gets no label."""
    matches = list(_HEADER_RE.finditer(note_text or ""))
    if not matches:
        return [None] * len(chunk_texts)
    tokens = [(m.group().lower(), m.start()) for m in _WORD_RE.finditer(note_text)]
    words = [w for w, _ in tokens]
    starts = [pos for _, pos in tokens]
    index: dict[tuple, list[int]] = {}
    for i in range(len(words) - _WINDOW + 1):
        index.setdefault(tuple(words[i:i + _WINDOW]), []).append(i)
    spans = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note_text)
        spans.append((STANDARD_HEADERS[int(m.lastgroup[1:])][1],
                      bisect.bisect_left(starts, m.start()), bisect.bisect_left(starts, end)))
    labels, cursor = [], 0
    for chunk_text in chunk_texts:                            # in chunk_index order
        cw = [w.lower() for w in _WORD_RE.findall(chunk_text or "")]
        position = None
        for offset in range(0, max(1, len(cw) - _WINDOW + 1), 3):
            hits = index.get(tuple(cw[offset:offset + _WINDOW]))
            if hits:                                          # prefer the occurrence at or after the previous chunk
                later = [h for h in hits if h - offset >= cursor - 150]
                position = (later[0] if later else hits[0]) - offset
                break
        if position is None:
            labels.append(None)
            continue
        cursor = max(cursor, position)
        found = []
        for name, start, end in spans:
            overlap = min(end, position + len(cw)) - max(start, position)
            if overlap > 0 and (2 * overlap >= len(cw) or 2 * overlap >= end - start):
                found.append((start, name))
        labels.append("; ".join(dict.fromkeys(name for _, name in sorted(found))) or None)
    return labels


def backfill(relabel_all: bool = False, batch: int = 500) -> int:
    """Write chunk_search_labels for discharge-note chunks. Returns chunks labelled."""
    from sqlalchemy import text
    from src.storage import engine
    todo = "" if relabel_all else """ AND NOT EXISTS (SELECT 1 FROM note_chunks nc JOIN chunk_search_labels l USING (chunk_id)
                                            WHERE nc.note_id = cn.note_id)"""
    with engine.connect() as conn:
        note_ids = conn.execute(text(
            f"SELECT cn.note_id FROM clinical_notes cn WHERE cn.note_type = 'discharge'{todo} ORDER BY cn.note_id"
        )).scalars().all()
    labelled = 0
    for i in range(0, len(note_ids), batch):
        ids = note_ids[i:i + batch]
        with engine.begin() as conn:
            notes = dict(conn.execute(text(
                "SELECT note_id, COALESCE(text_original, text_deid) FROM clinical_notes WHERE note_id = ANY(:ids)"),
                {"ids": ids}).fetchall())
            chunks = conn.execute(text(
                "SELECT note_id, chunk_id, chunk_text FROM note_chunks WHERE note_id = ANY(:ids) "
                "ORDER BY note_id, chunk_index"), {"ids": ids}).fetchall()
            by_note: dict[int, list] = {}
            for note_id, chunk_id, chunk_text in chunks:
                by_note.setdefault(note_id, []).append((chunk_id, chunk_text))
            rows = []
            for note_id, mine in by_note.items():
                for (chunk_id, _), label in zip(mine, label_chunks(notes.get(note_id) or "", [t for _, t in mine])):
                    if label:
                        rows.append({"chunk_id": chunk_id, "labels": label})
            if rows:
                conn.execute(text("""INSERT INTO chunk_search_labels (chunk_id, labels) VALUES (:chunk_id, :labels)
                                     ON CONFLICT (chunk_id) DO UPDATE SET labels = EXCLUDED.labels"""), rows)
                labelled += len(rows)
        print(f"  {min(i + batch, len(note_ids)):,}/{len(note_ids):,} notes, {labelled:,} chunks labelled", flush=True)
    return labelled


def demo() -> None:
    note = ("Name: X\nAllergies:\nPenicillin\n\nBrief Hospital Course:\nThe patient was admitted for chest pain and "
            "was treated with aspirin, then monitored on telemetry for two days without further events.\n\n"
            "Discharge Disposition:\nHome\n\nDischarge Diagnosis:\nUnstable angina and hypertension were the diagnoses.\n")
    course = "[IMPRESSION] The patient was admitted for chest pain and was treated with aspirin, then monitored on telemetry"
    merged = "[Discharge Disposition] Home\nUnstable angina and hypertension were the diagnoses."
    # the merged chunk keeps "[Discharge Disposition]" as its own prefix; the label adds what the prefix lost
    assert label_chunks(note, [course, merged, "words that are not in this note at all, anywhere"]) == [
        "Brief Hospital Course", "Discharge Diagnosis", None]
    assert label_chunks("no headers here", ["no headers here"]) == [None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="relabel chunks that already have labels")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        demo()
        print("ok")
    else:
        print(f"labelled {backfill(relabel_all=args.all):,} chunks")
