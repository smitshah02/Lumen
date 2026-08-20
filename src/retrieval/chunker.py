"""
Clinical Note Chunker
======================
Splits de-identified clinical notes into overlapping chunks for embedding.

Clinical notes have structure (sections like HISTORY, LABS, MEDICATIONS)
that we want to preserve. This chunker:
  1. Splits on section headers first (if detected)
  2. Then splits long sections into overlapping windows by sentence
  3. Preserves section context in each chunk

Chunk sizes are tuned for MedCPT's 512-token limit.

Usage:
    from src.retrieval.chunker import ClinicalNoteChunker

    chunker = ClinicalNoteChunker()
    chunks = chunker.chunk_text(note_text, note_type="discharge")
    # Returns list of {"text": str, "chunk_index": int, "token_count": int}
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Common section headers in MIMIC clinical notes
SECTION_PATTERNS = [
    # Discharge summary sections
    r"(?m)^(?:CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|HPI|"
    r"PAST MEDICAL HISTORY|PMH|SOCIAL HISTORY|FAMILY HISTORY|"
    r"MEDICATIONS|MEDICATIONS ON ADMISSION|MEDICATIONS ON DISCHARGE|"
    r"ALLERGIES|REVIEW OF SYSTEMS|ROS|"
    r"PHYSICAL EXAM(?:INATION)?|PHYSICAL FINDINGS|"
    r"LABORATORY DATA|LABS|LAB(?:ORATORY)? RESULTS|"
    r"IMAGING|RADIOLOGY|"
    r"HOSPITAL COURSE|BRIEF HOSPITAL COURSE|"
    r"ASSESSMENT AND PLAN|ASSESSMENT|PLAN|A/?P|"
    r"DISCHARGE DIAGNOSIS|DISCHARGE DIAGNOSES|"
    r"DISCHARGE INSTRUCTIONS|DISCHARGE CONDITION|"
    r"DISCHARGE DISPOSITION|DISCHARGE MEDICATIONS|"
    r"FOLLOW(?:\s*-?\s*)UP|FOLLOW UP INSTRUCTIONS|"
    r"PROCEDURES?|OPERATIONS?|OPERATIVE FINDINGS|"
    r"IMPRESSION|FINDINGS|CONCLUSION|"
    r"ADDENDUM|ATTESTATION)\s*:?",
]


@dataclass
class Chunk:
    text: str
    chunk_index: int
    token_count: int
    section: Optional[str] = None


class ClinicalNoteChunker:
    """
    Chunks clinical notes preserving section structure.

    Parameters:
        max_tokens: Maximum tokens per chunk (default 384, leaves room
                    within MedCPT's 512-token limit for special tokens)
        overlap_tokens: Number of overlapping tokens between consecutive chunks
        min_chunk_tokens: Minimum tokens for a chunk (smaller chunks are merged
                          with the next one)
    """

    def __init__(
        self,
        max_tokens: int = 384,
        overlap_tokens: int = 64,
        min_chunk_tokens: int = 50,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

        # Compile section header regex
        self.section_re = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token count estimate. Clinical text averages ~1.3 tokens/word.
        Good enough for chunking; exact counts come from the tokenizer at
        embedding time (which truncates to max_length anyway).
        """
        return int(len(text.split()) * 1.3)

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split text on section headers. Returns list of (section_name, section_text).
        If no sections are detected, returns the whole text as one section.
        """
        matches = list(self.section_re.finditer(text))

        if not matches:
            return [("FULL_NOTE", text)]

        sections = []

        # Text before the first section header
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("PREAMBLE", preamble))

        # Each section: from this header to the next header
        for i, match in enumerate(matches):
            section_name = match.group().strip().rstrip(":")
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_name, section_text))

        return sections

    def _split_section_into_chunks(
        self, section_name: str, section_text: str
    ) -> list[tuple[str, str]]:
        """
        Split a single section into overlapping chunks by sentence.
        Returns list of (section_name, chunk_text).
        """
        token_count = self._estimate_tokens(section_text)

        # If section fits in one chunk, return as-is
        if token_count <= self.max_tokens:
            return [(section_name, section_text)]

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", section_text)
        if not sentences:
            return [(section_name, section_text)]

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)

            # If a single sentence exceeds max_tokens, force-split by words
            if sent_tokens > self.max_tokens:
                # Flush current buffer first
                if current_sentences:
                    chunks.append(
                        (section_name, " ".join(current_sentences))
                    )
                    current_sentences = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                wc_tokens = 0
                for word in words:
                    wt = self._estimate_tokens(word)
                    if wc_tokens + wt > self.max_tokens and word_chunk:
                        chunks.append((section_name, " ".join(word_chunk)))
                        # Overlap: keep last N tokens worth of words
                        overlap_words = []
                        overlap_t = 0
                        for w in reversed(word_chunk):
                            overlap_t += self._estimate_tokens(w)
                            if overlap_t > self.overlap_tokens:
                                break
                            overlap_words.insert(0, w)
                        word_chunk = overlap_words
                        wc_tokens = overlap_t
                    word_chunk.append(word)
                    wc_tokens += wt
                if word_chunk:
                    chunks.append((section_name, " ".join(word_chunk)))
                continue

            # Normal case: accumulate sentences
            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                chunks.append((section_name, " ".join(current_sentences)))

                # Overlap: keep last sentences that fit within overlap_tokens
                overlap_sents = []
                overlap_t = 0
                for s in reversed(current_sentences):
                    st = self._estimate_tokens(s)
                    if overlap_t + st > self.overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_t += st

                current_sentences = overlap_sents
                current_tokens = overlap_t

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Flush remaining
        if current_sentences:
            chunks.append((section_name, " ".join(current_sentences)))

        return chunks

    def chunk_text(
        self, text: str, note_type: str = "discharge"
    ) -> list[Chunk]:
        """
        Split a clinical note into chunks.

        Returns list of Chunk objects with text, index, token count, and section name.
        """
        if not text or not text.strip():
            return []

        # Step 1: Split into sections
        sections = self._split_into_sections(text)

        # Step 2: Split each section into chunks
        raw_chunks = []
        for section_name, section_text in sections:
            section_chunks = self._split_section_into_chunks(
                section_name, section_text
            )
            raw_chunks.extend(section_chunks)

        # Step 3: Merge tiny chunks with the next chunk
        merged = []
        buffer_section = None
        buffer_text = ""

        for section_name, chunk_text in raw_chunks:
            token_count = self._estimate_tokens(chunk_text)

            if token_count < self.min_chunk_tokens and buffer_text:
                # Append to buffer
                buffer_text += "\n" + chunk_text
            elif token_count < self.min_chunk_tokens:
                # Start buffering
                buffer_section = section_name
                buffer_text = chunk_text
            else:
                # Flush buffer if any
                if buffer_text:
                    merged.append((buffer_section, buffer_text))
                    buffer_text = ""
                    buffer_section = None
                merged.append((section_name, chunk_text))

        # Flush remaining buffer
        if buffer_text:
            if merged:
                # Append to last chunk
                last_section, last_text = merged[-1]
                merged[-1] = (last_section, last_text + "\n" + buffer_text)
            else:
                merged.append((buffer_section or "UNKNOWN", buffer_text))

        # Step 4: Build Chunk objects with section context prefix
        chunks = []
        for i, (section_name, chunk_text) in enumerate(merged):
            # Prefix chunk with section name for context
            if section_name and section_name not in ("FULL_NOTE", "PREAMBLE"):
                contextualized = f"[{section_name}] {chunk_text}"
            else:
                contextualized = chunk_text

            chunks.append(
                Chunk(
                    text=contextualized,
                    chunk_index=i,
                    token_count=self._estimate_tokens(contextualized),
                    section=section_name,
                )
            )

        return chunks

    def chunk_batch(
        self, texts: list[str], note_type: str = "discharge"
    ) -> list[list[Chunk]]:
        """Chunk a batch of notes."""
        return [self.chunk_text(t, note_type) for t in texts]
