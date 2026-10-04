"""Invented literature must never be evidence.

This build has no literature backend. `src/safety/stub_tools.py` fabricates
"Study on ..." titles and ids so the egress gate can be evaluated offline; the
graph used to wire that stub in as if it were real, which let synthesis cite it
as [P#]. These tests pin the three things that now prevent it.
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

import src.agents.graph as g
from src.agents import prompts
from src.safety import stub_tools
from tests.test_human_review import EVIDENCE, SUPPORTED
from tests.test_routing import REFUSAL, _cite

LIT_QUESTION = "What does the published literature say about metformin for this patient?"
FABRICATED = ("Study on", "stub", "pmid", "PMID", "NCT0", "doi", "[P")


@pytest.fixture
def run(monkeypatch):
    """The production graph on an in-memory checkpointer: one patient note, no
    guideline hits, and a fake model whose synthesis reply the test chooses."""
    box = {"reply": REFUSAL, "calls": []}
    monkeypatch.setattr(g, "patient_retrieval",
                        lambda s: {"patient_evidence": EVIDENCE, "node_trail": ["patient_retrieval"]})
    monkeypatch.setattr(g, "guideline_retrieval",
                        lambda s: {"guideline_evidence": [], "node_trail": ["guideline_retrieval"]})

    def chat_for(role, messages, **kw):
        box["calls"].append(role)
        box["prompt"] = messages[-1]["content"]
        return "{}" if role in ("verify", "concept") else box["reply"]

    monkeypatch.setattr(g, "chat_for", chat_for)
    graph = g._builder().compile(checkpointer=InMemorySaver())

    def ask(query, thread, reply=REFUSAL):
        box["reply"] = reply
        graph.invoke({"query": query, "subject_id": 1, "thread_id": thread},
                     config={"configurable": {"thread_id": thread}})
        return graph.get_state({"configurable": {"thread_id": thread}}).values
    ask.box = box
    return ask


def _everything_shown(st) -> str:
    """All text the API or UI could surface from a finished run."""
    sources = (st.get("patient_evidence") or []) + (st.get("guideline_evidence") or []) + (st.get("literature_evidence") or [])
    return " ".join([st.get("final_answer") or "", st.get("draft_answer") or ""]
                    + [c["claim"] for c in st.get("citations") or []]
                    + [s["label"] + " " + s["text"] for s in sources if s["label"].startswith("P")])


def test_literature_backend_is_off_unless_explicitly_enabled(monkeypatch):
    from src.safety import pubmed
    assert g.LITERATURE_TOOL is None                                          # the test (and default) configuration
    for value in ("", "none", "off", "stub", "stub_pubmed", "anything-else"):
        monkeypatch.setenv("LUMEN_LITERATURE_BACKEND", value)
        assert pubmed.configured_tool() is None and pubmed.backend_name() == "none", value
    monkeypatch.delenv("LUMEN_LITERATURE_BACKEND")
    assert pubmed.configured_tool() is None
    monkeypatch.setenv("LUMEN_LITERATURE_BACKEND", "PubMed")
    assert pubmed.configured_tool() is pubmed.search_literature and pubmed.backend_name() == "pubmed"


def test_pure_literature_question_says_unavailable_and_invents_nothing(run):
    st = run(LIT_QUESTION, "lit-1")
    assert st["query_type"] == "literature" and "literature_retrieval" in st["node_trail"]
    assert st["literature_evidence"] == [] and st["literature_unavailable"] is True
    assert g.LITERATURE_UNAVAILABLE in st["final_answer"] and REFUSAL in st["final_answer"]
    shown = _everything_shown(st)
    assert not any(marker in shown for marker in FABRICATED), shown
    assert "concept" not in run.box["calls"] and not st.get("egress_log")     # nothing was sent anywhere
    assert st["review_status"] == "auto_approved"                            # a known gap, not a review case
    assert "no published literature was retrieved" in run.box["prompt"]       # the model was told
    assert "VALID CITATION LABELS: S1" in run.box["prompt"]


def test_mixed_question_keeps_the_patient_evidence_and_marks_the_literature_gap(run):
    st = run(LIT_QUESTION, "lit-2", reply=SUPPORTED)
    assert st["final_answer"] == f"{SUPPORTED}\n\n{g.LITERATURE_UNAVAILABLE}"
    assert [(c["label"], c["verified"]) for c in st["citations"]] == [("S1", True)]   # the notice is not a claim
    assert st["review_status"] == "auto_approved" and st["literature_evidence"] == []


def test_a_model_that_cites_literature_anyway_is_not_believed(run):
    st = run(LIT_QUESTION, "lit-3", reply="A trial showed metformin lowers mortality [P1].")
    assert "[P1]" not in st["draft_answer"]                                   # stripped: P1 is not a valid label
    assert st["citations"][0]["verified"] is False and st["needs_human_review"] is True
    assert st["review_status"] == "pending"                                   # HITL still catches it


def test_stub_tool_output_never_becomes_evidence(monkeypatch):
    stub = stub_tools.search_literature("hyperkalemia management")
    assert stub["results"][0]["title"].startswith("Study on")                 # the stub does fabricate...
    assert g._to_literature_evidence(stub) == []                              # ...and none of it is evidence
    assert g._to_literature_evidence({"results": [{"title": "x", "pmid": "1"}]}) == []     # unnamed backend
    # wiring the stub in as the tool is still not enough
    monkeypatch.setattr(g, "LITERATURE_TOOL", stub_tools.search_literature)
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: '{"concept_query": "hyperkalemia management"}')
    out = g.literature_retrieval({"query": LIT_QUESTION, "query_type": "literature"})
    assert out["literature_evidence"] == []


def test_a_real_backend_would_still_be_citable():
    ev = g._to_literature_evidence({"source": "pubmed", "results": [{"pmid": "123", "title": "A real title"}]})
    assert [(e["label"], e["backend"]) for e in ev] == [("P1", "pubmed")]
    assert g._citable_literature({"literature_evidence": ev}) == ev


@pytest.mark.parametrize("backend", ["stub_pubmed", "", None])
def test_stub_literature_cannot_satisfy_verification(monkeypatch, backend):
    monkeypatch.setattr(g, "chat_for", lambda *a, **k: pytest.fail("verifier consulted for a stub source"))
    fake = {"chunk_id": -1, "source_type": "literature", "label": "P1", "text": "Study on statins (stub1)",
            "charttime": None, "note_type": "literature", "score": 0.0, "backend": backend}
    claim = "Study on statins shows benefit [P1]."
    out = g.verification({"literature_evidence": [fake], "draft_answer": claim, "citations": [_cite(claim, ["P1"])]})
    assert out["citations"][0]["verified"] is False and out["needs_human_review"] is True
    assert out["citations"][0]["verification_note"] == "no valid citation"


def test_guideline_evidence_is_untouched(monkeypatch):
    guideline = [{"chunk_id": 9, "source_type": "guideline", "note_type": "guideline", "charttime": None,
                  "score": 0.8, "label": "G1", "text": "Adults with hypertension should be offered an ACE inhibitor."}]
    state = {"query": "What do the guidelines recommend for hypertension?", "query_type": "guideline_check",
             "guideline_evidence": guideline}
    assert g.route_after_guidelines(state) == "synthesis"                     # literature is not consulted
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: "Guidelines advise an ACE inhibitor [G1].")
    out = g.synthesis(state)
    assert out["citations"][0]["label"] == "G1" and g.LITERATURE_UNAVAILABLE not in out["draft_answer"]
    # an empty guideline search falls back to literature, finds none, and announces nothing
    out = g.literature_retrieval({"query": state["query"], "query_type": "guideline_check"})
    assert out == {"literature_evidence": [], "literature_unavailable": False, "node_trail": ["literature_retrieval"]}


def test_notice_survives_approval_and_is_absent_after_rejection():
    base = {"literature_unavailable": True, "draft_answer": f"x\n\n{g.LITERATURE_UNAVAILABLE}"}
    approved = g.finalize({**base, "review_status": "reviewed", "final_answer": "x"})
    assert approved["final_answer"] == f"x\n\n{g.LITERATURE_UNAVAILABLE}"
    rejected = g.finalize({**base, "review_status": "rejected", "final_answer": g.REJECTED_ANSWER})
    assert "final_answer" not in rejected                                     # left exactly as the reviewer set it
    assert "final_answer" not in g.finalize({"review_status": "reviewed", "final_answer": "x"})


def test_prompt_note_appears_only_when_literature_was_asked_for():
    assert "no published literature" not in prompts.build_synthesis_prompt("q", EVIDENCE, [])
    assert "no published literature" in prompts.build_synthesis_prompt("q", EVIDENCE, [], literature_unavailable=True)


# --- the opt-in PubMed backend -------------------------------------------------
ESEARCH = b'{"esearchresult": {"idlist": ["111", "222"]}}'
EFETCH = b"""<PubmedArticleSet>
  <PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
    <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><Title>Heart Journal</Title></Journal>
    <ArticleTitle>Drug X in <i>heart</i> failure.</ArticleTitle>
    <Abstract><AbstractText Label="RESULTS">Mortality fell by 12 percent.</AbstractText></Abstract>
  </Article></MedlineCitation></PubmedArticle>
  <PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
    <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>Renal Review</Title></Journal>
    <ArticleTitle>A second paper.</ArticleTitle>
  </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def pubmed_http(monkeypatch):
    """PubMed's two endpoints, canned. Records every URL that would have left the machine."""
    from src.safety import pubmed
    urls = []

    class _Reply:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1): return self.body

    def urlopen(url, timeout=None, context=None):
        assert context is not None and context.verify_mode.name == "CERT_REQUIRED"
        urls.append(url)
        return _Reply(ESEARCH if "esearch" in url else EFETCH)

    monkeypatch.setattr(pubmed.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(g, "LITERATURE_TOOL", pubmed.search_literature)
    return urls


def test_pubmed_results_are_parsed_and_only_the_query_is_sent(pubmed_http):
    from src.safety import pubmed
    out = pubmed.search_literature("drug x heart failure")
    assert out["source"] == "pubmed" and [r["pmid"] for r in out["results"]] == ["111", "222"]
    assert out["results"][0] == {"pmid": "111", "title": "Drug X in heart failure.", "journal": "Heart Journal",
                                 "year": "2021", "abstract": "Mortality fell by 12 percent."}
    assert out["results"][1]["year"] == "2019" and out["results"][1]["abstract"] == ""
    assert len(pubmed_http) == 2 and all(u.startswith("https://eutils.ncbi.nlm.nih.gov/") for u in pubmed_http)
    from urllib.parse import parse_qs, urlsplit
    sent = [parse_qs(urlsplit(u).query) for u in pubmed_http]
    assert sent[0] == {"db": ["pubmed"], "tool": ["lumen"], "term": ["drug x heart failure"], "retmax": ["3"],
                       "retmode": ["json"], "sort": ["relevance"]}
    assert sent[1] == {"db": ["pubmed"], "tool": ["lumen"], "id": ["111,222"], "retmode": ["xml"]}


def test_enabled_backend_gives_real_citable_literature(run, pubmed_http, monkeypatch):
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: (
        '{"concept_query": "drug x heart failure"}' if role == "concept" else
        "{}" if role == "verify" else "Drug X lowered mortality in one study [P1]."))
    st = run(LIT_QUESTION, "lit-real")
    lit = st["literature_evidence"]
    assert [(e["label"], e["backend"], e["note_type"]) for e in lit] == [
        ("P1", "pubmed", "PubMed PMID 111"), ("P2", "pubmed", "PubMed PMID 222")]
    assert "Mortality fell by 12 percent." in lit[0]["text"] and "PMID 111" in lit[0]["text"]
    assert st["literature_unavailable"] is False and g.LITERATURE_EMPTY not in st["draft_answer"]
    assert st["citations"][0]["label"] == "P1"                                # a real source is a valid label
    assert [(r["tool"], r["allowed"]) for r in st["egress_log"]] == [("search_literature", True)]
    sent = "".join(pubmed_http)                                                # only the approved concept query went out
    assert len(pubmed_http) == 2 and "term=drug+x+heart+failure" in sent
    assert "subject" not in sent and "metformin" not in sent and "patient" not in sent


def test_a_query_carrying_patient_text_never_reaches_pubmed(run, pubmed_http, monkeypatch):
    leak = EVIDENCE[0]["text"]                                                # the model copies the patient note
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: (
        __import__("json").dumps({"concept_query": leak}) if role == "concept" else REFUSAL))
    st = run(LIT_QUESTION, "lit-leak")
    assert pubmed_http == []                                                  # nothing left the machine
    assert st["egress_log"] and not any(r["allowed"] for r in st["egress_log"])
    assert st["literature_evidence"] == [] and g.LITERATURE_EMPTY in st["final_answer"]


def test_backend_failure_degrades_to_an_explicit_notice(run, monkeypatch):
    def offline(query):
        raise TimeoutError("no network")
    monkeypatch.setattr(g, "LITERATURE_TOOL", offline)
    monkeypatch.setattr(g, "chat_for", lambda role, messages, **kw: (
        '{"concept_query": "drug x heart failure"}' if role == "concept" else REFUSAL))
    st = run(LIT_QUESTION, "lit-offline")
    assert st["literature_evidence"] == [] and st["review_status"] == "auto_approved"
    assert g.LITERATURE_EMPTY in st["final_answer"] and not st.get("errors")
