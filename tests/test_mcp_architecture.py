"""The optional MCP adapter must stay on the same data/service path as the API."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mcp_has_no_second_synthetic_corpus_or_hard_coded_lab_ids():
    server = (ROOT / "src/mcp_server/server.py").read_text()
    planes = (ROOT / "src/mcp_server/planes.py").read_text()
    assert "DEMO_NOTES" not in server + planes
    assert "DEMO_LABS" not in server + planes
    assert "DEMO_TIMELINE" not in server + planes
    assert "LAB_ITEMIDS" not in server


def test_mcp_reuses_retrieval_and_structured_lab_services():
    server = (ROOT / "src/mcp_server/server.py").read_text()
    assert "HybridRetriever" in server
    assert "LabResolver" in server
    assert "resolver.fetch" in server
