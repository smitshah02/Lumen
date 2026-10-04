"""
PubMed search — Lumen's one external tool, OFF unless explicitly enabled
========================================================================
    LUMEN_LITERATURE_BACKEND=pubmed

Unset (the default), the graph has no literature backend, makes no outbound
call, and tells the user literature retrieval is unavailable.

When enabled, exactly one thing leaves the machine: the short concept query
the egress gate has already approved ("sglt2 inhibitors heart failure"). It is
written by the local model from the user's QUESTION — never from retrieved
patient text — and src/safety/egress_gate.py blocks it if it contains an
identifier, a date, lab shorthand, or any span of the patient's notes. No
subject id, no note text, no answer and no API key is ever sent.

Uses NCBI E-utilities: esearch for ids, efetch for title and abstract.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT_S = 8
MAX_BYTES = 2_000_000       # an efetch of three abstracts is a few kilobytes
ABSTRACT_CHARS = 1500       # keeps three abstracts inside the synthesis context window


def backend_name() -> str:
    """'pubmed' when enabled, otherwise 'none'. Anything unrecognised is 'none'."""
    name = os.environ.get("LUMEN_LITERATURE_BACKEND", "").strip().lower()
    if name and name not in ("pubmed", "none", "off", "0"):
        logger.warning("LUMEN_LITERATURE_BACKEND=%r is not a known backend; literature stays off", name)
    return "pubmed" if name == "pubmed" else "none"


def configured_tool():
    """The literature search function, or None when no backend is enabled."""
    return search_literature if backend_name() == "pubmed" else None


def _tls() -> ssl.SSLContext:
    """Certificate verification is always on. certifi's CA bundle is used when
    installed, because a python.org macOS build ships no usable root store."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(endpoint: str, params: dict) -> bytes:
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode({'db': 'pubmed', 'tool': 'lumen', **params})}"
    with urllib.request.urlopen(url, timeout=TIMEOUT_S, context=_tls()) as response:   # fixed https host
        return response.read(MAX_BYTES)


def _text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def search_literature(query: str, max_results: int = 3) -> dict:
    """Top PubMed hits for a concept query, as {"source": "pubmed", "results": [...]}.
    Raises on a network or parse failure; the caller treats that as no results."""
    found = json.loads(_get("esearch.fcgi", {"term": query, "retmax": max_results,
                                             "retmode": "json", "sort": "relevance"}))
    ids = [i for i in found.get("esearchresult", {}).get("idlist", []) if str(i).isdigit()][:max_results]
    results = []
    if ids:
        root = ET.fromstring(_get("efetch.fcgi", {"id": ",".join(ids), "retmode": "xml"}))
        for article in root.iter("PubmedArticle"):
            pmid = _text(article.find("./MedlineCitation/PMID"))
            title = _text(article.find(".//ArticleTitle"))
            if not pmid or not title:
                continue
            date = article.find(".//JournalIssue/PubDate")
            year = (_text(date.find("Year")) or _text(date.find("MedlineDate"))[:4]) if date is not None else ""
            abstract = " ".join(_text(a) for a in article.iter("AbstractText"))
            results.append({"pmid": pmid, "title": title, "journal": _text(article.find(".//Journal/Title")),
                            "year": year, "abstract": abstract[:ABSTRACT_CHARS]})
    return {"source": "pubmed", "query": query, "results": results}
