"""
Stub external tools
===================
Stand-ins for PubMed / ClinicalTrials.gov so the gate and its eval can run
with no network and no dependencies. Real MCP tools drop in behind the same
call_external() wrapper without changing the gate.
"""

def search_literature(query: str, max_results: int = 3) -> dict:
    return {"source": "stub_pubmed", "query": query,
            "results": [{"pmid": f"stub{i}", "title": f"Study on {query[:40]}"}
                        for i in range(1, max_results + 1)]}


def search_trials(condition: str, max_results: int = 3) -> dict:
    return {"source": "stub_clinicaltrials", "condition": condition,
            "results": [{"nct_id": f"NCT0000000{i}", "title": f"Trial for {condition[:40]}"}
                        for i in range(1, max_results + 1)]}