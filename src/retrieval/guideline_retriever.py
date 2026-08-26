"""
Guideline Retrieval
===================
Retrieval over guideline_chunks. Separate from HybridRetriever because
the corpus differs structurally: no patient, no charttime, no BM25 arm
(guideline_chunks has only the HNSW index — see schema.py:185).

Pipeline: MedCPT vector search -> BGE cross-encoder rerank.

Guideline prose and clinical-note prose are stylistically different, so
do NOT assume the weights tuned on notes transfer here.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

from src.storage import engine
from src.retrieval.embeddings import MedCPTEmbedder
from src.retrieval.hybrid_retriever_v2 import BGEReranker, RetrievalResult

logger = logging.getLogger(__name__)

# Sentinels for the note-shaped fields RetrievalResult requires.
# rerank() only reads context_text/chunk_text, so these are never used
# for scoring — they exist to satisfy the dataclass.
_GUIDELINE_SENTINEL = dict(note_id=-1, subject_id=-1, hadm_id=None, note_type="guideline")


class GuidelineRetriever:
    """Vector search + rerank over indexed clinical guidelines."""

    def __init__(
        self,
        embedder: Optional[MedCPTEmbedder] = None,
        reranker: Optional[BGEReranker] = None,
        use_reranker: bool = True,
        vector_top_n: int = 20,
        top_k: int = 3,
    ):
        # Accept shared instances — on a 16GB machine we must not load
        # a second MedCPT/BGE pair alongside the HybridRetriever's.
        self.embedder = embedder or MedCPTEmbedder()
        self.use_reranker = use_reranker
        self.reranker = reranker if reranker is not None else (BGEReranker() if use_reranker else None)
        self.vector_top_n = vector_top_n
        self.top_k = top_k

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        qvec = self.embedder.embed_query(query)
        qlit = "[" + ",".join(str(float(x)) for x in qvec) + "]"

        sql = text("""
            SELECT chunk_id, source_file, section_title, chunk_index,
                   chunk_text, token_count,
                   1 - (embedding <=> CAST(:qvec AS vector)) AS vscore
            FROM guideline_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :top_n
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"qvec": qlit, "top_n": self.vector_top_n}).fetchall()

        if not rows:
            logger.warning("guideline search returned nothing — is guideline_chunks populated?")
            return []

        results: list[RetrievalResult] = []
        for r in rows:
            # Prepend the section title: it carries most of the topical
            # signal in guideline text and measurably helps the reranker.
            body = r.chunk_text
            titled = f"{r.section_title}\n{body}" if r.section_title else body
            results.append(RetrievalResult(
                chunk_id=r.chunk_id,
                chunk_index=r.chunk_index,
                chunk_text=body,
                context_text=titled,
                token_count=r.token_count or 0,
                vector_score=float(r.vscore),
                final_score=float(r.vscore),
                sources=["vector"],
                charttime=None,
                **_GUIDELINE_SENTINEL,
            ))

        if self.reranker is not None:
            results = self.reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]

        logger.debug(f"guideline search '{query[:40]}' -> {len(results)} hits")
        return results

    def source_of(self, chunk_id: int) -> dict:
        """Resolve a chunk back to its source file and section, for citations."""
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT source_file, section_title FROM guideline_chunks WHERE chunk_id = :cid"
            ), {"cid": chunk_id}).fetchone()
        return {"source_file": row.source_file, "section_title": row.section_title} if row else {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    gr = GuidelineRetriever()
    for q in ["management of hyperkalemia",
              "when to start dialysis in chronic kidney disease",
              "anticoagulation in atrial fibrillation"]:
        print("\n" + "=" * 70)
        print(f"  {q}")
        print("=" * 70)
        for i, r in enumerate(gr.search(q), 1):
            meta = gr.source_of(r.chunk_id)
            print(f"  [{i}] score={r.final_score:.3f}  {meta.get('source_file')}")
            print(f"      {meta.get('section_title')}")
            print(f"      {r.chunk_text[:150].strip()}...")