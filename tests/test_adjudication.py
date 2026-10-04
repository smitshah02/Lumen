"""Blinded human adjudication: the sample, what the reviewer sees, and the comparison."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.evals import adjudication as adj

ROOT = Path(__file__).resolve().parents[1]


def _claim(i, label, verdict, subject=1, case="factual_rag"):
    return {"case": case, "subject_id": subject, "thread_id": f"t{i}", "claim": f"Claim number {i} [{label}1].",
            "labels": [f"{label}1"] if label else [], "verifier_verdict": verdict,
            "verifier_note": "SECRET-VERIFIER-NOTE", "sources": {f"{label}1": f"excerpt {i}"} if label else {}}


ROWS = ([_claim(i, "S", True, subject=i % 3) for i in range(20)] + [_claim(i, "S", False) for i in range(20, 24)]
        + [_claim(i, "L", True) for i in range(24, 30)] + [_claim(i, "G", True) for i in range(30, 33)]
        + [_claim(i, "G", False) for i in range(33, 35)] + [_claim(i, "P", True) for i in range(35, 38)]
        + [_claim(38, "P", False), _claim(39, "", False)])


def test_sample_is_stratified_reproducible_and_deduplicated():
    sample = adj.build_sample(ROWS + ROWS[:5], target=25, seed=7)
    assert len(sample) == 25 and [r["id"] for r in sample] == list(range(1, 26))
    assert len({r["claim"] for r in sample}) == 25                             # duplicates collapsed
    strata = adj.strata_counts(sample)
    assert set(strata) == {"S:supported", "S:flagged", "L:supported", "G:supported", "G:flagged",
                           "P:supported", "P:flagged", "none:flagged"}        # every stratum is represented
    assert strata["P:flagged"] == 1 and strata["none:flagged"] == 1 and strata["S:flagged"] == 4
    assert strata["L:supported"] == adj.STRUCTURED_CAP                         # templated SQL claims are capped
    assert [r["claim"] for r in adj.build_sample(ROWS, 25, 7)] == [r["claim"] for r in adj.build_sample(ROWS, 25, 7)]
    assert [r["claim"] for r in adj.build_sample(ROWS, 25, 7)] != [r["claim"] for r in adj.build_sample(ROWS, 25, 8)]
    assert len(adj.build_sample(ROWS[:6], target=25)) == 6                     # never padded with invented claims
    assert all(r["human_verdict"] is None for r in sample)                     # no label is assigned by code


def test_reviewer_view_hides_the_verifier_and_the_test_case():
    sample = adj.build_sample(ROWS, target=10, seed=1)
    for row in sample:
        shown = json.dumps(adj.blind_view(row))
        assert "SECRET-VERIFIER-NOTE" not in shown and "verifier" not in shown and "hidden" not in shown
        assert "factual_rag" not in shown and "subject_id" not in shown and "thread" not in shown
        assert row["claim"] in shown
    uncited = next(r for r in adj.build_sample(ROWS, 40) if not r["labels"])
    assert adj.blind_view(uncited)["sources"] == [{"label": None, "type": "no citation", "excerpt": ""}]
    assert adj.blind_view(sample[0])["sources"][0]["type"] in adj.SOURCE_TYPES.values()


def _labelled(pairs):
    return [{"id": i, "claim": f"c{i}", "labels": ["S1"], "sources": {}, "human_verdict": human,
             "hidden": {"verifier_verdict": verifier}} for i, (verifier, human) in enumerate(pairs, 1)]


def test_summary_measures_agreement_and_false_support():
    rows = _labelled([(True, "SUPPORTED"), (True, "SUPPORTED"), (True, "PARTIALLY_SUPPORTED"), (True, "UNSUPPORTED"),
                      (False, "UNSUPPORTED"), (False, "SUPPORTED"), (True, "UNCLEAR"), (True, "SKIP"), (True, None)])
    s = adj.summarize(rows)
    assert (s["sampled"], s["labelled"], s["unlabelled"], s["judged"]) == (9, 8, 1, 6)
    assert s["human"] == {"SUPPORTED": 3, "PARTIALLY_SUPPORTED": 1, "UNSUPPORTED": 2, "UNCLEAR": 1, "SKIP": 1}
    assert s["human_supported_rate"] == (3, 6) and s["human_supported_or_partial_rate"] == (4, 6)
    assert s["exact_agreement"] == (3, 6) and s["grouped_agreement"] == (4, 6)
    assert s["false_supported"] == 1 and s["verifier_false_support_rate"] == (1, 4)
    assert s["false_flagged"] == 1 and s["verifier_supported_but_only_partial"] == 1
    assert s["disagreement_ids"] == [3, 4, 6]                                  # listed, never dropped
    text = adj.render(s)
    assert "VERIFIER FALSE-SUPPORT RATE" in text and "25% (1/4)" in text and "Not yet labelled" in text


def test_summary_of_an_unlabelled_sample_claims_nothing():
    s = adj.summarize(adj.build_sample(ROWS, 10))
    assert s["judged"] == 0 and s["verifier_false_support_rate"] == (0, 0) and "n/a" in adj.render(s)


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("adjudicate_cli", ROOT / "scripts" / "adjudicate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labelling_is_blind_persists_each_answer_and_resumes(cli, tmp_path, capsys):
    path = tmp_path / "sample.jsonl"
    cli.write_jsonl(path, adj.build_sample(ROWS, target=4, seed=3))
    args = type("A", (), {"file": str(path), "redo": False})
    answers = iter(["x", "s", "looks right", "u", "", "q"])               # bad key is re-asked; quit after two
    assert cli.cmd_label(args, ask=lambda prompt="": next(answers)) == 0
    shown = capsys.readouterr().out
    assert "SECRET-VERIFIER-NOTE" not in shown and "factual_rag" not in shown and "verifier" not in shown.lower()
    rows = cli.read_jsonl(path)
    assert [r["human_verdict"] for r in rows] == ["SUPPORTED", "UNSUPPORTED", None, None]
    assert rows[0]["human_note"] == "looks right" and rows[0]["hidden"]["verifier_note"] == "SECRET-VERIFIER-NOTE"
    rest = iter(["p", "", "c", ""])
    assert cli.cmd_label(args, ask=lambda prompt="": next(rest)) == 0      # resumes at the third claim
    assert [r["human_verdict"] for r in cli.read_jsonl(path)] == ["SUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED", "UNCLEAR"]
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_sampling_refuses_to_overwrite_human_verdicts(cli, tmp_path, capsys):
    source = tmp_path / "claims-1.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    out = tmp_path / "adjudication-sample.jsonl"
    args = type("A", (), {"inputs": [str(source)], "out": str(out), "target": 25, "seed": 7, "cohort": "fixture"})
    assert cli.cmd_sample(args) == 0 and len(cli.read_jsonl(out)) == 25
    assert cli.read_jsonl(out)[0]["hidden"]["cohort"] == "fixture"
    rows = cli.read_jsonl(out)
    rows[0]["human_verdict"] = "SUPPORTED"
    cli.write_jsonl(out, rows)
    assert cli.cmd_sample(args) == 2 and cli.read_jsonl(out)[0]["human_verdict"] == "SUPPORTED"
    assert "already holds human verdicts" in capsys.readouterr().err
