"""
Clinical Guideline Indexer
============================
Extracts text from clinical guideline PDFs, chunks them preserving
section structure, embeds with MedCPT, and stores in the
guideline_chunks table in Postgres.

Handles:
  - ADA Standards of Care in Diabetes 2026
  - GOLD COPD Report 2025
  - AHA/ACC Chronic Coronary Disease 2023

Usage:
    cd ~/Lumen
    source .venv/bin/activate
    python -m src.retrieval.index_guidelines

    # Reindex (drop existing and rebuild):
    python -m src.retrieval.index_guidelines --reindex

    # Custom directory:
    python -m src.retrieval.index_guidelines --dir data/guidelines
"""

from __future__ import annotations

import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from pypdf import PdfReader
from sqlalchemy import text as sa_text

from src.storage import engine
from src.retrieval.embeddings import MedCPTEmbedder

logger = logging.getLogger(__name__)

DEFAULT_GUIDELINE_DIR = Path.home() / "Lumen" / "data" / "guidelines"


# ===========================================================================
# PDF Text Extraction
# ===========================================================================

def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF, page by page.
    Returns list of {"page": int, "text": str}.
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Clean up common PDF extraction artifacts
            text = clean_pdf_text(text)
            pages.append({"page": i + 1, "text": text})

    logger.info(f"  Extracted {len(pages)} pages from {pdf_path.name} ({len(reader.pages)} total)")
    return pages


def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    # Fix broken hyphenation at line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page header/footer patterns (common in guidelines)
    text = re.sub(r"(?m)^S\d+\s+Diabetes Care.*$", "", text)  # ADA headers
    text = re.sub(r"(?m)^©\s*\d{4}.*$", "", text)  # Copyright lines
    text = re.sub(r"(?m)^\d+\s*$", "", text)  # Standalone page numbers
    # Collapse whitespace runs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ===========================================================================
# Guideline-Aware Chunking
# ===========================================================================

# Section header patterns for clinical guidelines
GUIDELINE_SECTION_RE = re.compile(
    r"(?m)^(?:"
    # Numbered sections: "1.", "1.1", "2.3.4", "Chapter 1"
    r"(?:Chapter\s+)?\d{1,2}(?:\.\d{1,2}){0,3}\s+[A-Z]"
    # ALL-CAPS headers
    r"|[A-Z][A-Z\s]{5,60}$"
    # "RECOMMENDATION" or "EVIDENCE" blocks
    r"|(?:RECOMMENDATION|EVIDENCE|SUMMARY|KEY POINTS|TABLE|FIGURE)"
    # Common guideline sections
    r"|(?:Introduction|Background|Methodology|Methods|Results|Discussion"
    r"|Screening|Diagnosis|Treatment|Management|Prevention|Monitoring"
    r"|Assessment|Classification|Pharmacotherapy|Pharmacologic"
    r"|Nonpharmacologic|Lifestyle|Goals|Targets|Follow.?up"
    r"|Complications|Comorbidities|Special Populations"
    r"|Older Adults|Children|Pregnancy|Hospitalized)"
    r")",
    re.IGNORECASE,
)


def chunk_guideline_pages(
    pages: list[dict],
    source_file: str,
    max_tokens: int = 384,
    overlap_tokens: int = 64,
    min_chunk_tokens: int = 50,
) -> list[dict]:
    """
    Chunk guideline pages preserving section structure.

    Strategy:
      1. Concatenate all pages into full text
      2. Split on section headers
      3. Split long sections into overlapping windows
      4. Prefix each chunk with source and section for context
    """
    # Concatenate all pages
    full_text = "\n\n".join(p["text"] for p in pages)

    # Build page offset map for tracking which page a chunk came from
    page_starts = []
    offset = 0
    for p in pages:
        page_starts.append((offset, p["page"]))
        offset += len(p["text"]) + 2  # +2 for \n\n

    def find_page(char_pos: int) -> int:
        """Find which page a character position falls on."""
        for i in range(len(page_starts) - 1, -1, -1):
            if char_pos >= page_starts[i][0]:
                return page_starts[i][1]
        return 1

    # Split on section headers
    matches = list(GUIDELINE_SECTION_RE.finditer(full_text))

    sections = []
    if not matches:
        # No headers found — treat as one big section
        sections.append(("GUIDELINE", full_text, 0))
    else:
        # Text before first header
        if matches[0].start() > 100:
            preamble = full_text[:matches[0].start()].strip()
            if preamble:
                sections.append(("INTRODUCTION", preamble, 0))

        # Each section
        for i, match in enumerate(matches):
            header = match.group().strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            section_text = full_text[start:end].strip()
            if section_text and len(section_text.split()) > 10:
                sections.append((header[:80], section_text, match.start()))

    # Chunk each section
    chunks = []
    chunk_index = 0

    for section_name, section_text, section_start in sections:
        words = section_text.split()
        est_tokens = int(len(words) * 1.3)

        if est_tokens <= max_tokens:
            # Section fits in one chunk
            if est_tokens >= min_chunk_tokens:
                page_num = find_page(section_start)
                chunks.append({
                    "source_file": source_file,
                    "section_title": section_name,
                    "chunk_index": chunk_index,
                    "chunk_text": f"[{source_file}] [{section_name}] {section_text}",
                    "token_count": est_tokens,
                    "page_num": page_num,
                })
                chunk_index += 1
        else:
            # Split into overlapping windows by sentence
            sentences = re.split(r"(?<=[.!?])\s+", section_text)
            current = []
            current_tokens = 0

            for sentence in sentences:
                sent_tokens = int(len(sentence.split()) * 1.3)

                if current_tokens + sent_tokens > max_tokens and current:
                    # Emit chunk
                    text = " ".join(current)
                    if int(len(text.split()) * 1.3) >= min_chunk_tokens:
                        page_num = find_page(section_start)
                        chunks.append({
                            "source_file": source_file,
                            "section_title": section_name,
                            "chunk_index": chunk_index,
                            "chunk_text": f"[{source_file}] [{section_name}] {text}",
                            "token_count": int(len(text.split()) * 1.3),
                            "page_num": page_num,
                        })
                        chunk_index += 1

                    # Keep overlap
                    overlap_sents = []
                    overlap_t = 0
                    for s in reversed(current):
                        st = int(len(s.split()) * 1.3)
                        if overlap_t + st > overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_t += st
                    current = overlap_sents
                    current_tokens = overlap_t

                current.append(sentence)
                current_tokens += sent_tokens

            # Flush remaining
            if current:
                text = " ".join(current)
                if int(len(text.split()) * 1.3) >= min_chunk_tokens:
                    page_num = find_page(section_start)
                    chunks.append({
                        "source_file": source_file,
                        "section_title": section_name,
                        "chunk_index": chunk_index,
                        "chunk_text": f"[{source_file}] [{section_name}] {text}",
                        "token_count": int(len(text.split()) * 1.3),
                        "page_num": page_num,
                    })
                    chunk_index += 1

    return chunks


# ===========================================================================
# Database Storage
# ===========================================================================

def store_guideline_chunks(chunks: list[dict], embeddings: np.ndarray):
    """Store guideline chunks with embeddings in Postgres."""
    records = []
    for i, chunk in enumerate(chunks):
        vec = embeddings[i]
        records.append({
            "source_file": chunk["source_file"],
            "section_title": chunk["section_title"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "token_count": chunk["token_count"],
            "embedding": f"[{','.join(str(float(x)) for x in vec)}]",
        })

    with engine.connect() as conn:
        for i in range(0, len(records), 100):
            batch = records[i:i + 100]
            conn.execute(
                sa_text("""
                    INSERT INTO guideline_chunks
                        (source_file, section_title, chunk_index, chunk_text, token_count, embedding)
                    VALUES
                        (:source_file, :section_title, :chunk_index, :chunk_text, :token_count, CAST(:embedding AS vector))
                """),
                batch,
            )
        conn.commit()


# ===========================================================================
# Main Pipeline
# ===========================================================================

def index_guidelines(
    guideline_dir: Path = DEFAULT_GUIDELINE_DIR,
    reindex: bool = False,
):
    """
    Full guideline indexing pipeline:
      1. Find PDF files in directory
      2. Extract text from each PDF
      3. Chunk with section awareness
      4. Embed with MedCPT Article Encoder
      5. Store in guideline_chunks table
    """
    print("=" * 70)
    print("  LUMEN GUIDELINE INDEXING PIPELINE")
    print("=" * 70)
    print()

    # Find PDFs
    if not guideline_dir.exists():
        print(f"ERROR: Directory not found: {guideline_dir}")
        return

    pdf_files = sorted(guideline_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {guideline_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files:")
    for f in pdf_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    print()

    # Clear existing if reindexing
    if reindex:
        print("Clearing existing guideline chunks (--reindex)...")
        with engine.connect() as conn:
            conn.execute(sa_text("DELETE FROM guideline_chunks"))
            conn.commit()
        print()

    # Check what's already indexed
    with engine.connect() as conn:
        result = conn.execute(sa_text("SELECT DISTINCT source_file FROM guideline_chunks"))
        existing = {row[0] for row in result}

    # Step 1: Extract and chunk
    print("Step 1: Extracting text and chunking...")
    t0 = time.time()

    all_chunks = []
    for pdf_path in pdf_files:
        short_name = pdf_path.stem[:60]  # truncate long filenames

        if short_name in existing and not reindex:
            print(f"  ⏭ Skipping {short_name} (already indexed)")
            continue

        print(f"  📄 Processing {pdf_path.name}...")
        pages = extract_pdf_text(pdf_path)

        if not pages:
            print(f"    ⚠ No text extracted — may be a scanned PDF")
            continue

        chunks = chunk_guideline_pages(pages, source_file=short_name)
        all_chunks.extend(chunks)
        print(f"    → {len(pages)} pages → {len(chunks)} chunks")

    if not all_chunks:
        print("\n  No new chunks to index.")
        with engine.connect() as conn:
            total = conn.execute(sa_text("SELECT COUNT(*) FROM guideline_chunks")).scalar()
        print(f"  Total guideline chunks in database: {total}")
        return

    chunk_time = time.time() - t0
    print(f"\n  Total: {len(all_chunks)} chunks in {chunk_time:.1f}s")
    print()

    # Step 2: Embed with MedCPT
    print("Step 2: Embedding with MedCPT Article Encoder...")
    embedder = MedCPTEmbedder()

    chunk_texts = [c["chunk_text"] for c in all_chunks]
    t0 = time.time()
    embeddings = embedder.embed_documents(chunk_texts, show_progress=True)
    embed_time = time.time() - t0
    print(f"  Embedded {len(chunk_texts)} chunks in {embed_time:.1f}s")
    print()

    # Step 3: Store in database
    print("Step 3: Storing in Postgres...")
    t0 = time.time()
    store_guideline_chunks(all_chunks, embeddings)
    store_time = time.time() - t0
    print(f"  Stored {len(all_chunks)} chunks in {store_time:.1f}s")
    print()

    # Summary
    print("=" * 70)
    print("  GUIDELINE INDEXING COMPLETE")
    print("=" * 70)

    with engine.connect() as conn:
        total = conn.execute(sa_text("SELECT COUNT(*) FROM guideline_chunks")).scalar()

    print(f"\n  PDFs processed:      {len(pdf_files)}")
    print(f"  New chunks created:  {len(all_chunks)}")
    print(f"  Avg tokens/chunk:    {sum(c['token_count'] for c in all_chunks) // len(all_chunks)}")
    print(f"  Total time:          {chunk_time + embed_time + store_time:.1f}s")
    print(f"\n  Total guideline chunks in database: {total}")

    # Per-source breakdown
    print(f"\n  Per-guideline breakdown:")
    with engine.connect() as conn:
        rows = conn.execute(sa_text(
            "SELECT source_file, COUNT(*) as cnt FROM guideline_chunks GROUP BY 1 ORDER BY 1"
        )).mappings().all()
    for row in rows:
        print(f"    {row['source_file']:<55} {row['cnt']:>5} chunks")

    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Index Clinical Guidelines")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_GUIDELINE_DIR), help="Guidelines directory")
    parser.add_argument("--reindex", action="store_true", help="Drop and rebuild")
    args = parser.parse_args()

    index_guidelines(
        guideline_dir=Path(args.dir),
        reindex=args.reindex,
    )
