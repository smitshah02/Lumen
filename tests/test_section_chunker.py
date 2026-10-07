"""Section-aware chunker (data-foundation plan, E12). Notes are invented. Most tests
use a small stand-in tokenizer so they run anywhere; the real MedCPT tokenizer is
used when the local model directory is present."""
import re
from pathlib import Path

import pytest

from src.retrieval.chunker import (V2_CHUNK_CONFIG, V2_MAX_TOKENS, V2_MIN_EMBED_TOKENS, V2_TARGET_TOKENS,
                                   SectionChunk, chunk_note)
from src.retrieval.embeddings import DEFAULT_ARTICLE_MODEL
from src.retrieval.section_labels import parse_sections


class _Words:
    """Stand-in tokenizer: one token per word or punctuation mark, with offsets."""
    def __call__(self, text, **_):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]}


@pytest.fixture(scope="module")
def real():
    if not (Path(DEFAULT_ARTICLE_MODEL) / "tokenizer_config.json").exists():
        pytest.skip("MedCPT article tokenizer not present locally")
    from src.retrieval.chunker import load_article_tokenizer
    return load_article_tokenizer()


def _row(i):
    return f"___ 07:{i % 60:02d}AM BLOOD WBC-{i}.4 RBC-2.63* Hgb-8.1* Hct-26.8* \nMCV-102*# MCH-30.8 Creat-{i}.3* mg/dL\n"


RESULTS = ("Pertinent Results:\nADMISSION LABS\n" + "".join(_row(i) for i in range(30))
           + "\nDISCHARGE LABS\n" + "".join(_row(i) for i in range(30, 60)) + "\n")
MEDS = "Discharge Medications:\n" + "".join(
    f"{i}. Drug{i} {i * 5} mg PO DAILY with food and a full glass of water \nRX *drug{i}* {i} tablet(s) by mouth daily Disp #*30\n"
    for i in range(1, 60)) + "\n"
NOTE = ("Name: ___  Unit No: ___\nService: MEDICINE\n\nAllergies:\nPenicillins\n\nChief Complaint:\nChest pain\n\n"
        "History of Present Illness:\n" + "The patient reports chest pain on exertion for three days. " * 6 + "\n\n"
        + RESULTS + "Brief Hospital Course:\n" + "She was diuresed with good effect and improved steadily. " * 8 + "\n\n"
        + MEDS + "Discharge Disposition:\nHome\n")


def _check(text, chunks, tok):
    """The invariants every result must hold."""
    assert all(isinstance(c, SectionChunk) and text[c.start:c.end] == c.text for c in chunks)      # offsets map back exactly
    assert chunks[0].start == _first_start(text, chunks) and chunks[-1].end == len(text)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start                                                                     # nothing lost, nothing repeated
    for c in chunks:
        assert c.token_count == len(tok(c.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]) + 2
        assert not c.embed or (V2_MIN_EMBED_TOKENS <= c.token_count <= V2_MAX_TOKENS)
        assert c.text.strip()                                                                       # no whitespace-only chunk


def _first_start(text, chunks):
    return chunks[0].start


def _by_section(chunks):
    out = {}
    for c in chunks:
        out.setdefault((c.section_ord, c.section_name), []).append(c)
    return out


def test_sections_under_the_limit_stay_whole_and_sections_never_merge():
    tok = _Words()
    chunks = chunk_note(NOTE, "discharge", tok)
    _check(NOTE, chunks, tok)
    sections = parse_sections(NOTE, "discharge")
    grouped = _by_section(chunks)
    assert [k for k in grouped] == [(s.order, s.name) for s in sections]                  # one group per section, in order
    for s in sections:
        mine = grouped[(s.order, s.name)]
        assert "".join(c.text for c in mine) == s.text                                     # a section's chunks are exactly that section
        assert [c.chunk_ord for c in mine] == list(range(len(mine)))
    whole = {name: len(mine) for (_, name), mine in grouped.items()}
    assert whole["History of Present Illness"] == 1 and whole["Brief Hospital Course"] == 1 and whole["Pertinent Results"] > 1


def test_tiny_sections_and_the_note_header_are_kept_but_not_embedded():
    chunks = chunk_note(NOTE, "discharge", _Words())
    by = {c.section_name: c for c in chunks if c.section_name in ("Header", "Allergies", "Chief Complaint", "Discharge Disposition")}
    assert set(by) == {"Header", "Allergies", "Chief Complaint", "Discharge Disposition"}
    assert not any(c.embed for c in by.values())
    assert by["Allergies"].text == "Allergies:\nPenicillins\n\n" and by["Allergies"].token_count < V2_MIN_EMBED_TOKENS
    long_header = "Name: ___ " * 30 + "\n\nAllergies:\nNone\n"
    assert chunk_note(long_header, "discharge", _Words())[0].embed is False                # the header is never embedded, whatever its size


def test_oversized_results_split_on_date_blocks_then_whole_rows():
    tok = _Words()
    mine = [c for c in chunk_note(NOTE, "discharge", tok) if c.section_name == "Pertinent Results"]
    assert len(mine) > 2 and all(c.embed and c.token_count <= V2_TARGET_TOKENS for c in mine)
    assert "".join(c.text for c in mine) == RESULTS
    # every chunk starts at the section header, a block title, or the first line of a row: never on a continuation line
    assert all(re.match(r"Pertinent Results:|DISCHARGE LABS|___ \d\d:\d\dAM", c.text) for c in mine)
    assert not any(c.text.startswith("MCV-") for c in mine)
    # the discharge block starts its own chunk rather than being glued to the last admission rows
    assert any(c.text.startswith("DISCHARGE LABS\n") for c in mine)
    # values and units are untouched
    for i in (0, 29, 30, 59):
        assert sum(c.text.count(f"WBC-{i}.4 RBC-2.63* Hgb-8.1* Hct-26.8* \nMCV-102*# MCH-30.8 Creat-{i}.3* mg/dL\n") for c in mine) == 1


def test_oversized_list_splits_between_items_and_keeps_each_item_whole():
    mine = [c for c in chunk_note(NOTE, "discharge", _Words()) if c.section_name == "Discharge Medications"]
    assert len(mine) > 1 and "".join(c.text for c in mine) == MEDS
    for c in mine:
        assert re.match(r"Discharge Medications:|\d+\. Drug", c.text)                      # starts at an item, not at its RX line
        assert c.text.count("\n") >= 2 and "  " not in c.text.replace(" \n", "")           # line breaks kept; not flattened


def test_long_prose_splits_on_lines_then_words_and_one_unbroken_run_is_cut_safely():
    tok = _Words()
    lines = "Brief Hospital Course:\n" + "".join(f"Line {i} " + "word " * 40 + "\n" for i in range(30))
    by_line = chunk_note(lines, "discharge", tok)
    _check(lines, by_line, tok)
    assert len(by_line) > 1 and all(c.text.endswith("\n") for c in by_line)                # cut at line ends
    one_line = "Brief Hospital Course: " + "word " * 1500
    by_word = chunk_note(one_line, "discharge", tok)
    _check(one_line, by_word, tok)
    assert len(by_word) >= 3 and all(c.token_count <= V2_TARGET_TOKENS for c in by_word)
    assert all(not c.text[0].isspace() or c.chunk_ord == 0 for c in by_word)               # cut at whitespace, not mid-word
    row = "Pertinent Results:\n___ 07:10AM BLOOD " + ",".join(f"A{i}-{i}.5*" for i in range(900)) + "\n"   # one row, no spaces
    hard = chunk_note(row, "discharge", tok)
    _check(row, hard, tok)
    assert len(hard) > 2 and max(c.token_count for c in hard) <= V2_MAX_TOKENS and "".join(c.text for c in hard) == row


def _shape(chunks):
    return [(c.section_name, c.section_ord, c.chunk_ord, c.start, c.end, c.text, c.token_count) for c in chunks]


def test_min_embed_tokens_is_an_explicit_build_parameter_with_default_40():
    tok = _Words()
    assert V2_CHUNK_CONFIG == {"target_tokens": 480, "max_tokens": 512, "min_embed_tokens": 40}
    default = chunk_note(NOTE, "discharge", tok)
    assert chunk_note(NOTE, "discharge", tok, **V2_CHUNK_CONFIG) == default                      # the recorded config is the default
    assert chunk_note(NOTE, "discharge", tok, min_embed_tokens=40) == default
    short = {c.section_name for c in default if not c.embed}
    assert short == {"Header", "Allergies", "Chief Complaint", "Discharge Disposition"}          # default behaviour, unchanged


def test_threshold_zero_embeds_short_clinical_sections_but_never_the_header():
    tok = _Words()
    default = chunk_note(NOTE, "discharge", tok)
    zero = chunk_note(NOTE, "discharge", tok, **{**V2_CHUNK_CONFIG, "min_embed_tokens": 0})
    assert _shape(zero) == _shape(default)                                                       # same boundaries, text, offsets, counts
    assert {c.section_name for c in zero if not c.embed} == {"Header"}                           # only the header stays out
    flipped = {c.section_name for c, d in zip(zero, default) if c.embed != d.embed}
    assert flipped == {"Allergies", "Chief Complaint", "Discharge Disposition"}
    assert all(c.embed for c, d in zip(zero, default) if d.embed)                                # nothing embedded before is dropped
    for threshold in (0, 1, 40, 200, 10_000):                                                    # the header at every setting
        got = chunk_note(NOTE, "discharge", tok, min_embed_tokens=threshold)
        assert _shape(got) == _shape(default) and not got[0].embed and got[0].section_name == "Header"
    assert not any(c.embed for c in chunk_note(NOTE, "discharge", tok, min_embed_tokens=10_000))


RADIOLOGY = ("EXAMINATION:  CHEST (PA AND LAT)\n\nINDICATION:  ___ year old woman with cough.\n\nTECHNIQUE:  Chest PA and lateral\n\n"
             "COMPARISON:  ___\n\nFINDINGS:\n" + "The lungs are clear without focal consolidation. " * 6
             + "\n\nIMPRESSION:\nNo acute cardiopulmonary process.\n")


def test_radiology_report_that_fits_is_one_whole_report_chunk():
    (only,) = chunk_note(RADIOLOGY, "radiology", _Words())
    assert (only.section_name, only.section_ord, only.chunk_ord, only.start, only.end) == ("Report", 0, 0, 0, len(RADIOLOGY))
    assert only.text == RADIOLOGY and only.embed and only.token_count <= V2_TARGET_TOKENS


def test_radiology_report_over_the_limit_splits_by_its_sections():
    tok = _Words()
    long = RADIOLOGY.replace("FINDINGS:\n", "FINDINGS:\n" + "".join(f"{i}. Finding {i} " + "detail " * 30 + "\n" for i in range(1, 40)))
    chunks = chunk_note(long, "radiology", tok)
    _check(long, chunks, tok)
    assert [n for _, n in _by_section(chunks)] == ["Examination", "Indication", "Technique", "Comparison", "Findings", "Impression"]
    findings = [c for c in chunks if c.section_name == "Findings"]
    assert len(findings) > 1 and all(c.embed and c.token_count <= V2_TARGET_TOKENS for c in findings)
    assert "".join(c.text for c in chunks) == long


def test_same_input_gives_the_same_chunks_and_degenerate_input_gives_none():
    tok = _Words()
    assert chunk_note(NOTE, "discharge", tok) == chunk_note(NOTE, "discharge", tok)
    assert chunk_note("", "discharge", tok) == [] and chunk_note(" \n\n ", "radiology", tok) == [] and chunk_note(None, "discharge", tok) == []
    bare = chunk_note("Allergies:\n\n\nChief Complaint:\n", "discharge", tok)              # headers with empty bodies
    assert [c.section_name for c in bare] == ["Allergies", "Chief Complaint"] and not any(c.embed for c in bare)
    _check("Allergies:\n\n\nChief Complaint:\n", bare, tok)


def test_real_tokenizer_never_lets_an_embedded_chunk_over_the_model_limit(real):
    row = "Pertinent Results:\n___ 07:10AM BLOOD " + ",".join(f"A{i}-{i}.5*" for i in range(900)) + "\n"
    for text, note_type in ((NOTE, "discharge"), (RADIOLOGY, "radiology"), (row, "discharge")):
        chunks = chunk_note(text, note_type, real)
        _check(text, chunks, real)
        for c in chunks:
            # counted the way the embedder will see it: special tokens on, no truncation
            assert c.token_count == len(real(c.text, truncation=False, verbose=False)["input_ids"])
            assert not c.embed or c.token_count <= V2_MAX_TOKENS
        assert chunk_note(text, note_type, real) == chunks
    numbers = [c for c in chunk_note(NOTE, "discharge", real) if c.section_name == "Pertinent Results"]
    assert len(numbers) > 2 and all(re.match(r"Pertinent Results:|DISCHARGE LABS|___ \d\d:\d\dAM", c.text) for c in numbers)
