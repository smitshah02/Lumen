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
import math
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

    @staticmethod
    def _max_words_for_tokens(token_limit: int) -> int:
        """Largest word count accepted by the canonical 1.3x estimator."""
        return max(1, math.ceil((token_limit + 1) / 1.3) - 1)

    def _split_oversized_chunk(
        self, section_name: str, chunk_text: str
    ) -> list[str]:
        """Enforce the configured ceiling after merging and context prefixing."""
        prefix_words = 0 if section_name in ("FULL_NOTE", "PREAMBLE") else 1
        content_words = max(
            1, self._max_words_for_tokens(self.max_tokens) - prefix_words
        )
        overlap_words = min(
            self._max_words_for_tokens(self.overlap_tokens), content_words - 1
        )
        words = chunk_text.split()
        pieces = []
        start = 0
        while start < len(words):
            end = min(start + content_words, len(words))
            pieces.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap_words
        return pieces

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
        for section_name, chunk_text in merged:
            # Prefix chunk with section name for context
            has_prefix = section_name and section_name not in ("FULL_NOTE", "PREAMBLE")
            contextualized = f"[{section_name}] {chunk_text}" if has_prefix else chunk_text
            pieces = (
                self._split_oversized_chunk(section_name, chunk_text)
                if self._estimate_tokens(contextualized) > self.max_tokens
                else [chunk_text]
            )
            for piece in pieces:
                contextualized = f"[{section_name}] {piece}" if has_prefix else piece
                chunks.append(
                    Chunk(
                        text=contextualized,
                        chunk_index=len(chunks),
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


# ===========================================================================
# Section-aware chunking for the v2 index (data-foundation plan, E12)
# ===========================================================================
# Built on parse_sections (section_labels.py). Everything above this line is the
# control chunker and is untouched. Differences that matter:
#   * tokens are counted with the MedCPT article tokenizer, not words x 1.3;
#   * a chunk is an exact slice of the note as written: offsets, newlines, lists
#     and result tables survive, and nothing is merged across sections;
#   * no chunk marked for embedding can exceed the model's limit: asserted here,
#     not left to the tokenizer's silent truncation.
# Chunks are produced in memory; storing them is E13.

V2_CHUNKER_VERSION = "section-chunker-v1"   # recorded with every v2 build; bump when chunk_note changes its output
V2_TARGET_TOKENS = 480        # a section that fits in this stays whole (leaves room under the limit)
V2_MAX_TOKENS = 512           # MedCPT's input limit, special tokens included
V2_MIN_EMBED_TOKENS = 40      # smaller chunks are kept and searchable by word, but not embedded
# The parameters of a chunk build, in the shape a build records as provenance:
# chunk_note(text, note_type, tokenizer, **config). min_embed_tokens decides only
# which chunks are embedded; it never moves a boundary. The note header is not
# embedded at any setting.
V2_CHUNK_CONFIG = {"target_tokens": V2_TARGET_TOKENS, "max_tokens": V2_MAX_TOKENS,
                   "min_embed_tokens": V2_MIN_EMBED_TOKENS}
_SPECIAL_TOKENS = 2           # [CLS] and [SEP]

# A line that starts a unit which should not be cut from its continuation lines:
# a result row ("___ 07:10AM BLOOD ..."), a numbered list item, or a bullet.
_UNIT_START_RE = re.compile(r"[ \t]*(?:___[ \t]+\d{1,2}:\d{2}|\d{1,3}[.)][ \t]|[-*#•][ \t])")


@dataclass(frozen=True)
class SectionChunk:
    section_name: str
    section_ord: int      # the section's position in the note
    chunk_ord: int        # the chunk's position within its section
    start: int            # offsets into the note as written: text == note[start:end]
    end: int
    text: str
    token_count: int      # real tokenizer count of `text`, special tokens included
    embed: bool


def load_article_tokenizer():
    """The tokenizer of the MedCPT article encoder, from the local model directory."""
    from transformers import AutoTokenizer
    from src.retrieval.embeddings import DEFAULT_ARTICLE_MODEL
    return AutoTokenizer.from_pretrained(DEFAULT_ARTICLE_MODEL, local_files_only=True)


def _token_starts(tokenizer, text: str) -> list[int]:
    """Start offset of every token of `text` (no special tokens)."""
    return [a for a, _ in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                    truncation=False, verbose=False)["offset_mapping"]]


def chunk_note(text: str, note_type: str, tokenizer, *, target_tokens: int = V2_TARGET_TOKENS,
               max_tokens: int = V2_MAX_TOKENS, min_embed_tokens: int = V2_MIN_EMBED_TOKENS) -> list[SectionChunk]:
    """Cut a note as written into section chunks for the v2 index.

    A section that fits in `target_tokens` is one chunk. A larger one is split,
    preferring in order: blank-line blocks (date blocks in a results table),
    then rows / list items with their continuation lines, then single lines,
    then whitespace, and a hard cut between tokens only when one unbroken run
    is itself too long. A radiology report that fits is one chunk for the whole
    report. The note header and chunks under `min_embed_tokens` are returned
    with embed=False; that threshold changes only the embed flag, never where a
    chunk starts or ends. Same input, tokenizer and parameters, same output."""
    import bisect
    from src.retrieval.section_labels import PREAMBLE, WHOLE_REPORT, Section, parse_sections

    text = text or ""
    sections = parse_sections(text, note_type)
    if not sections:
        return []
    starts = _token_starts(tokenizer, text)
    budget = target_tokens - _SPECIAL_TOKENS

    def tokens(a: int, b: int) -> int:
        return bisect.bisect_left(starts, b) - bisect.bisect_left(starts, a)

    if note_type == "radiology" and tokens(0, len(text)) <= budget:
        sections = [Section(WHOLE_REPORT, 0, 0, len(text), text)]         # the approved whole-report rule

    def pack(spans: list[tuple[int, int]], level: int) -> list[tuple[int, int]]:
        """Greedily join neighbouring spans up to the budget; break up any span that is too big alone."""
        out, cur = [], None
        for a, b in spans:
            if tokens(a, b) > budget:
                if cur:
                    out.append(cur)
                    cur = None
                out.extend(split(a, b, level + 1))
            elif cur and tokens(cur[0], b) <= budget:
                cur = (cur[0], b)
            else:
                if cur:
                    out.append(cur)
                cur = (a, b)
        if cur:
            out.append(cur)
        return out

    def split(a: int, b: int, level: int) -> list[tuple[int, int]]:
        if tokens(a, b) <= budget:
            return [(a, b)]
        if level == 0:        # blocks: runs of lines separated by blank lines
            pieces = [(a + m.start(), a + m.end()) for m in re.finditer(r"(?:[^\n]*\S[^\n]*\n?)+(?:[ \t]*\n)*|(?:[ \t]*\n)+", text[a:b])]
        elif level == 1:      # units: a row or list item with its continuation lines; otherwise one line each
            pieces, grouped = [], False
            for m in re.finditer(r"[^\n]*\n?", text[a:b]):
                if m.end() == m.start():
                    continue
                starts_unit = bool(_UNIT_START_RE.match(m.group()))
                if pieces and grouped and not starts_unit and m.group().strip():
                    pieces[-1] = (pieces[-1][0], a + m.end())             # continuation of the row above
                else:
                    pieces.append((a + m.start(), a + m.end()))
                    grouped = starts_unit
        elif level == 2:      # single lines
            pieces = [(a + m.start(), a + m.end()) for m in re.finditer(r"[^\n]*\n?", text[a:b]) if m.end() > m.start()]
        elif level == 3:      # whitespace-separated runs
            pieces = [(a + m.start(), a + m.end()) for m in re.finditer(r"\s*\S+\s*", text[a:b])]
        else:                 # one unbroken run longer than the budget: cut between tokens
            first, last = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
            cuts = [a] + [starts[i] for i in range(first + budget, last, budget)] + [b]
            return list(zip(cuts, cuts[1:]))
        if not pieces or pieces == [(a, b)]:
            return split(a, b, level + 1)
        return pack(pieces, level)

    chunks: list[SectionChunk] = []
    for section in sections:
        spans = split(section.start, section.end, 0)
        merged: list[tuple[int, int]] = []
        for a, b in spans:                                 # whitespace-only spans ride with the chunk before them
            if merged and not text[a:b].strip():
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for chunk_ord, (a, b) in enumerate(merged):
            piece = text[a:b]
            count = len(_token_starts(tokenizer, piece)) + _SPECIAL_TOKENS     # counted on the chunk as it will be embedded
            embed = section.name != PREAMBLE and count >= min_embed_tokens
            # The invariant the old path lacked: an over-limit chunk is a build error, never a truncated embedding.
            assert not embed or count <= max_tokens, (
                f"chunk {section.name}[{chunk_ord}] has {count} tokens, over the {max_tokens}-token limit")
            chunks.append(SectionChunk(section.name, section.order, chunk_ord, a, b, piece, count, embed))
    return chunks
