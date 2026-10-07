"""Model-assisted admission resolver (data-foundation plan, E10 / decisions R2, R2b).
No model and no database: the model is a function that returns the JSON under test."""
import json

import pytest

from src.agents import admission_scope as ads
from src.agents.admission_scope import (MAX_CANDIDATES, accept_model_choice, candidate_prompt, content_terms,
                                        descriptive_reference, evidence_is_relevant, resolve_with_model)

A, B, C = 2000100, 2000200, 2000300
CANDIDATES = [
    {"hadm_id": A, "admitted": "2180-05-06", "discharged": "2180-05-07",
     "evidence": ["diagnosis: Essential (primary) hypertension", "diagnosis: Chest pain, unspecified", "note: Chest pain"]},
    {"hadm_id": B, "admitted": "2180-06-26", "discharged": "2180-06-30",
     "evidence": ["diagnosis: Essential (primary) hypertension", "procedure: Insertion of drug-eluting coronary artery stent",
                  "note: Cardiac catheterization with stent placement"]},
    {"hadm_id": C, "admitted": "2181-01-02", "discharged": "2181-01-09",
     "evidence": ["diagnosis: End stage renal disease", "procedure: Hemodialysis", "note: Chest pain"]},
]
QUESTION = "What medications were given during the stay when she had the stent?"
STENT = "the stay when she had the stent"
HTN = "What happened in the admission for hypertension?"
PAIN = "What was done in the admission for chest pain?"


def _model(**reply):
    return lambda prompt: json.dumps(reply)


# --- when the step applies ---------------------------------------------------------------------
@pytest.mark.parametrize("question,phrase", [
    (QUESTION, "during the stay when she had the stent"),
    ("What happened in the admission when dialysis was started?", "in the admission when dialysis was started"),
    ("What was found during the hospitalization when he had the procedure?", "during the hospitalization when he had the procedure"),
    ("What was her creatinine in the admission for pneumonia?", "in the admission for pneumonia"),
])
def test_descriptive_references_are_recognised(question, phrase):
    assert descriptive_reference(question).group().strip() == phrase


@pytest.mark.parametrize("question", [
    "What medications is the patient taking?", "What were the admission labs?",
    "What was her creatinine on admission for chest pain?",            # no determiner: not a reference to a stay
    "When was she admitted?", "What was the plan for discharge?",
])
def test_ordinary_questions_are_not_descriptive_references(question):
    assert descriptive_reference(question) is None


# --- the acceptance rule -------------------------------------------------------------------------
def test_unique_verified_evidence_resolves_the_admission_and_strips_the_phrase():
    r = resolve_with_model(QUESTION, CANDIDATES, _model(
        hadm_id=B, evidence="Insertion of drug-eluting coronary artery stent", status="resolved", confidence=0.4))
    assert (r.status, r.hadm_id, r.source, r.reason) == ("resolved", B, "model", None)
    assert r.phrase == "during the stay when she had the stent" and r.retrieval_query == "What medications were given?"
    # case and spacing do not matter; it is still the admission's own text
    assert accept_model_choice({"hadm_id": str(B), "evidence": "  cardiac  catheterization with STENT placement "},
                               CANDIDATES, STENT)[:2] == (B, "resolved")


REJECTED = [
    # question, name, model reply, status, reason fragment
    (QUESTION, "id outside the candidates", {"hadm_id": 99999999, "evidence": "Insertion of drug-eluting coronary artery stent"},
     "unresolved", "is not one of this patient's admissions"),
    (QUESTION, "fabricated evidence", {"hadm_id": B, "evidence": "Placement of a bare-metal stent in the LAD"},
     "unresolved", "not in that admission's record"),
    (QUESTION, "real evidence, wrong admission", {"hadm_id": A, "evidence": "Insertion of drug-eluting coronary artery stent"},
     "unresolved", "not in that admission's record"),
    (HTN, "relevant evidence in several admissions", {"hadm_id": A, "evidence": "Essential (primary) hypertension"},
     "ambiguous", f"also appears in admission(s) [{B}]"),
    (PAIN, "same note line in two admissions", {"hadm_id": C, "evidence": "Chest pain"}, "ambiguous", f"also appears in admission(s) [{A}]"),
    (QUESTION, "no evidence", {"hadm_id": B, "evidence": ""}, "unresolved", "cited no evidence"),
    (QUESTION, "evidence too short to mean anything", {"hadm_id": B, "evidence": "the"}, "unresolved", "cited no evidence"),
    (QUESTION, "no admission selected", {"hadm_id": None, "evidence": "", "status": "none"}, "unresolved", "selected no admission"),
    (QUESTION, "id that is not a number", {"hadm_id": "the second one", "evidence": "Hemodialysis"}, "unresolved", "usable admission id"),
    # the new condition: real, unique evidence that is not about what was asked
    ("What happened in the admission when she had a hip replacement?", "unique but unrelated evidence",
     {"hadm_id": C, "evidence": "procedure: Hemodialysis", "status": "resolved", "confidence": 0.95},
     "unresolved", "does not mention what the question describes"),
    ("What happened in the admission when she had brain surgery?", "requested event absent from every admission",
     {"hadm_id": B, "evidence": "Insertion of drug-eluting coronary artery stent"}, "unresolved", "does not mention what the question describes"),
    ("What was found in the hospitalization when he had the procedure?", "generic overlap only (procedure)",
     {"hadm_id": C, "evidence": "procedure: Hemodialysis"}, "unresolved", "does not mention what the question describes"),
    ("What happened in the stay when the patient had surgery and treatment?", "generic overlap only (surgery, treatment, patient, stay)",
     {"hadm_id": C, "evidence": "diagnosis: End stage renal disease"}, "unresolved", "does not mention what the question describes"),
    # a high confidence changes nothing: only the checks decide
    (QUESTION, "confident but unverifiable", {"hadm_id": B, "evidence": "stent placed in March", "confidence": 0.99, "status": "resolved"},
     "unresolved", "not in that admission's record"),
    (PAIN, "confident but ambiguous", {"hadm_id": A, "evidence": "Chest pain", "confidence": 1.0}, "ambiguous", "also appears"),
    ("What happened in the admission when she had a hip replacement?", "confident but unrelated",
     {"hadm_id": C, "evidence": "Hemodialysis", "confidence": 1.0, "status": "resolved"}, "unresolved", "does not mention"),
]


@pytest.mark.parametrize("question,name,reply,status,reason", REJECTED, ids=[c[1] for c in REJECTED])
def test_model_choice_is_never_applied_without_verified_relevant_unique_evidence(question, name, reply, status, reason):
    r = resolve_with_model(question, CANDIDATES, lambda prompt: json.dumps(reply))
    assert (r.status, r.hadm_id, r.source) == (status, None, "model") and reason in r.reason
    assert r.retrieval_query == question and r.phrase                    # named, not applied; the question is left whole


def test_relevance_is_a_deterministic_overlap_of_distinctive_terms():
    assert evidence_is_relevant("Insertion of drug-eluting coronary artery stent(s)", STENT)
    assert evidence_is_relevant("procedure: Electroencephalogram", "the stay when she had the electroencephalogram")
    assert evidence_is_relevant("Other endoscopy of small intestine", "the admission when she had the small bowel endoscopic exam")
    assert evidence_is_relevant("note: EEG monitoring", "the admission when she had an EEG")
    assert not evidence_is_relevant("procedure: Hemodialysis", "the admission when she had a hip replacement")
    assert not evidence_is_relevant("procedure: Thoracentesis", "the hospitalization when he had the procedure")
    assert not evidence_is_relevant("note: Surgery was performed during this hospital stay", "the stay when the patient had surgery")
    assert not evidence_is_relevant("Hemodialysis", "") and not evidence_is_relevant("", STENT)
    assert content_terms("the admission, hospital stay; patient's procedure / surgery & treatment") == set()
    assert content_terms("Coronary STENTS, drug-eluting!") == {"corona", "stent", "drug", "elutin"}


@pytest.mark.parametrize("a, b, same", [
    ("stent", "stents", True),                  # plain plural
    ("biopsy", "biopsies", True),               # -ies plural
    ("endoscopy", "endoscopic", True),          # first six letters
    ("dialysis", "hemodialysis", False),        # a prefix is not bridged: a documented safe miss
    ("abscess", "abscesses", True),
    ("EEG", "EEGs", True),
    ("EEG", "EKG", False),                      # three-letter words match whole or not at all
    ("walk", "walking", False),                 # only plurals are folded; other endings are a safe miss
])
def test_normalisation_of_inflections_is_exactly_this(a, b, same):
    assert evidence_is_relevant(a, b) is same and evidence_is_relevant(b, a) is same
    assert (content_terms(a) == content_terms(b)) is same


def test_function_words_and_coding_boilerplate_never_make_evidence_relevant():
    question = "the admission when she could not walk without help"
    assert not evidence_is_relevant("Fracture, not elsewhere classified", question)
    assert not evidence_is_relevant("Diabetes without mention of complication, type II", question)
    assert evidence_is_relevant("Difficulty in walking, not elsewhere classified", "the stay when she had difficulty walking")


def test_low_confidence_does_not_block_a_verified_choice():
    choice = {"hadm_id": C, "evidence": "Hemodialysis", "confidence": 0.01, "status": "ambiguous"}
    assert accept_model_choice(choice, CANDIDATES, "the admission when hemodialysis was started") == (C, "resolved", None)
    # and no confidence, however high, stands in for a failed check
    assert accept_model_choice({**choice, "confidence": 1.0}, CANDIDATES, "the admission when she had a hip replacement")[1] == "unresolved"
    assert accept_model_choice({**choice, "confidence": 1.0}, CANDIDATES)[1] == "unresolved"       # no description to be relevant to


@pytest.mark.parametrize("ask", [
    lambda p: (_ for _ in ()).throw(RuntimeError("model down")),
    lambda p: "not json at all", lambda p: "[1, 2]", lambda p: "",
])
def test_model_failure_is_unresolved_never_a_guess(ask):
    r = resolve_with_model(QUESTION, CANDIDATES, ask)
    assert (r.status, r.hadm_id, r.source) == ("unresolved", None, "model") and "did not return a usable answer" in r.reason


def test_too_many_admissions_or_none_are_not_compared():
    many = [{**CANDIDATES[0], "hadm_id": 3000000 + i} for i in range(MAX_CANDIDATES + 1)]
    asked = []
    r = resolve_with_model(QUESTION, many, lambda p: asked.append(p) or "{}")
    assert (r.status, r.hadm_id) == ("unresolved", None) and f"more than {MAX_CANDIDATES} are not compared" in r.reason
    assert asked == []                                                   # the model is not even asked
    assert resolve_with_model(QUESTION, [], lambda p: "{}").status == "unresolved"


def test_prompt_lists_only_these_admissions_and_checks_against_all_their_evidence():
    long = [{"hadm_id": A, "admitted": "x", "discharged": "y", "evidence": [f"diagnosis: Title {i}" for i in range(60)]},
            {"hadm_id": B, "admitted": "x", "discharged": "y", "evidence": ["diagnosis: Title 59"]}]
    prompt = candidate_prompt("q?", long)
    assert prompt.count("ADMISSION ") == 2 and "Title 24" in prompt and "Title 25" not in prompt     # bounded
    # "Title 59" is beyond what the model was shown for A, but it is in A's record and in B's: ambiguous, not unique
    assert accept_model_choice({"hadm_id": B, "evidence": "diagnosis: Title 59"}, long, "the admission with Title 59")[1] == "ambiguous"


# --- where it runs ------------------------------------------------------------------------------------
ADMISSIONS = [(A, "2180-05-06 22:00:00", "2180-05-07 17:00:00"), (B, "2180-06-26 18:00:00", "2180-06-30 18:00:00")]


@pytest.fixture
def g(monkeypatch):
    import src.agents.graph as graph
    monkeypatch.setattr(graph, "load_admissions", lambda sid: ADMISSIONS)
    monkeypatch.setattr(graph, "load_candidates", lambda sid: CANDIDATES)
    graph.calls = []
    monkeypatch.setattr(graph, "_ask_admission_model", lambda prompt: graph.calls.append(prompt) or json.dumps(
        {"hadm_id": B, "evidence": "Insertion of drug-eluting coronary artery stent", "status": "resolved", "confidence": 0.9}))
    monkeypatch.setattr(graph, "chat_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model in tests")))
    return graph


def test_structured_profile_uses_the_model_only_for_a_descriptive_reference(g, monkeypatch):
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": True, "sql_paths": True})
    scope = g.triage({"query": QUESTION, "subject_id": 7})["admission_scope"]
    assert (scope["status"], scope["hadm_id"], scope["source"]) == ("resolved", B, "model") and len(g.calls) == 1
    plain = g.triage({"query": "What medications is she taking?", "subject_id": 7})["admission_scope"]
    assert plain["status"] == "none" and len(g.calls) == 1               # nothing described: the model is not asked


def test_deterministic_rules_and_the_request_field_win_before_the_model(g, monkeypatch):
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": True, "sql_paths": True})
    by_rule = g.triage({"query": "During her last admission, what happened in the stay when she had the stent?",
                        "subject_id": 7})["admission_scope"]
    assert (by_rule["status"], by_rule["hadm_id"], by_rule["source"]) == ("resolved", B, "rule:last")
    by_request = g.triage({"query": QUESTION, "subject_id": 7, "request_hadm_id": A})["admission_scope"]
    assert (by_request["hadm_id"], by_request["source"]) == (A, "request")
    unresolved = g.triage({"query": "In the admission that ended on 2199-01-01, what about the stay when she had the stent?",
                           "subject_id": 7})["admission_scope"]
    assert unresolved["status"] == "unresolved" and unresolved["source"] == "rule:date"      # named and failed: not handed to the model
    assert g.calls == []


@pytest.mark.parametrize("settings", [{"admission_scope": False, "sql_paths": False},       # control
                                      {"admission_scope": True, "sql_paths": False}])       # scoped
def test_control_and_scoped_never_call_the_model_resolver(g, monkeypatch, settings):
    monkeypatch.setattr(g, "PROFILE_SETTINGS", settings)
    monkeypatch.setattr(g, "load_candidates", lambda sid: pytest.fail("must not load candidates"))
    out = g.triage({"query": QUESTION, "subject_id": 7})
    assert g.calls == []
    assert out.get("admission_scope", {"status": "none"})["status"] == "none"                # scoped: rules only, nothing found


def test_model_failure_in_the_graph_is_unresolved_and_readiness_errors_are_raised(g, monkeypatch):
    from src.storage.readiness import STRUCTURED, DataSourceNotReady
    monkeypatch.setattr(g, "PROFILE_SETTINGS", {"admission_scope": True, "sql_paths": True})
    monkeypatch.setattr(g, "_ask_admission_model", lambda prompt: (_ for _ in ()).throw(TimeoutError("slow")))
    scope = g.triage({"query": QUESTION, "subject_id": 7})["admission_scope"]
    assert (scope["status"], scope["hadm_id"], scope["source"]) == ("unresolved", None, "model")

    def not_ready(sid):
        raise DataSourceNotReady(STRUCTURED, "the latest structured load is running")
    monkeypatch.setattr(g, "load_candidates", not_ready)
    with pytest.raises(DataSourceNotReady):
        g.triage({"query": QUESTION, "subject_id": 7})


def test_candidate_loader_checks_readiness_first(monkeypatch):
    from src.storage import readiness
    from src.storage.readiness import STRUCTURED, DataSourceNotReady

    def not_ready(component):
        raise DataSourceNotReady(component, "no structured load has been run")
    monkeypatch.setattr(readiness, "require", not_ready)
    with pytest.raises(DataSourceNotReady) as e:
        ads.load_candidates(7)
    assert e.value.component == STRUCTURED


def test_dev_eval_counts_a_wrong_admission_separately_from_a_safe_abstention():
    from src.agents.admission_scope import AdmissionResolution
    from src.evals.admission_resolver_eval import score
    cases = [{"subject_id": 1, "question": "a", "expected_hadm_id": A}, {"subject_id": 1, "question": "b", "expected_hadm_id": A},
             {"subject_id": 1, "question": "c", "expected_hadm_id": None}, {"subject_id": 1, "question": "d", "expected_hadm_id": None},
             {"subject_id": 1, "question": "e", "expected_hadm_id": B}]
    answers = {"a": AdmissionResolution("resolved", "a", A, "model"), "b": AdmissionResolution("unresolved", "b", source="model"),
               "c": AdmissionResolution("ambiguous", "c", source="model"), "d": AdmissionResolution("resolved", "d", B, "model"),
               "e": AdmissionResolution("resolved", "e", A, "model")}
    out = score(cases, lambda sid, q: answers[q])
    assert (out["correct"], out["missed"], out["safe_abstention"], out["wrong_admission_applied"]) == (1, 1, 1, 2)
    assert out["answerable"] == 3 and out["accuracy_on_answerable"] == round(1 / 3, 3)
