"""
MedCPT Embedding Module
=========================
Dual-encoder embeddings using MedCPT for biomedical/clinical text.

MedCPT is a contrastive-pretrained model from NCBI specifically designed
for biomedical retrieval. It has two separate encoders:
  - Article Encoder: for indexing documents (notes, chunks, guidelines)
  - Query Encoder: for encoding search queries at retrieval time

Both produce 768-dimensional vectors. Similarity is computed via dot product
or cosine similarity.

Usage:
    from src.retrieval.embeddings import MedCPTEmbedder

    embedder = MedCPTEmbedder()

    # Index-time: embed document chunks
    vectors = embedder.embed_documents(["Patient presents with...", "Labs show..."])

    # Query-time: embed a search query
    query_vec = embedder.embed_query("What medications is the patient on?")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# Default model paths — matches your project structure
DEFAULT_QUERY_MODEL = str(Path.home() / "Lumen" / "models" / "medcpt-query")
DEFAULT_ARTICLE_MODEL = str(Path.home() / "Lumen" / "models" / "medcpt-article")


class MedCPTEmbedder:
    """
    MedCPT dual-encoder for biomedical text embedding.

    Uses the Article Encoder for documents and the Query Encoder for queries.
    Supports batched encoding with configurable batch size.
    Automatically uses MPS (Apple Silicon GPU) if available, otherwise CPU.
    """

    def __init__(
        self,
        query_model_path: str = DEFAULT_QUERY_MODEL,
        article_model_path: str = DEFAULT_ARTICLE_MODEL,
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.max_length = max_length
        self.batch_size = batch_size

        # Auto-detect device: MPS (Apple Silicon) > CUDA > CPU
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (no GPU detected)")

        # Load Article Encoder (for documents)
        logger.info(f"Loading MedCPT Article Encoder from {article_model_path}...")
        t0 = time.time()
        self.article_tokenizer = AutoTokenizer.from_pretrained(article_model_path)
        self.article_model = AutoModel.from_pretrained(article_model_path).to(self.device)
        self.article_model.eval()
        logger.info(f"  Article Encoder loaded in {time.time() - t0:.1f}s")

        # Load Query Encoder (for search queries)
        logger.info(f"Loading MedCPT Query Encoder from {query_model_path}...")
        t0 = time.time()
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_path)
        self.query_model = AutoModel.from_pretrained(query_model_path).to(self.device)
        self.query_model.eval()
        logger.info(f"  Query Encoder loaded in {time.time() - t0:.1f}s")

        self.embedding_dim = 768

    @torch.no_grad()
    def _encode_batch(
        self, texts: list[str], tokenizer, model
    ) -> np.ndarray:
        """Encode a batch of texts and return normalized embeddings."""
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = model(**encoded)

        # MedCPT uses [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

        # L2 normalize for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # MPS can produce NaN for certain inputs after L2 normalize.
        # Retry on CPU so the caller gets a valid vector instead of zeros.
        if torch.isnan(embeddings).any():
            if self.device.type != "cpu":
                logger.warning(
                    f"NaN embeddings on {self.device} — retrying on CPU"
                )
                cpu_encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                cpu_outputs = model.to("cpu")(**cpu_encoded)
                embeddings = cpu_outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                model.to(self.device)  # move back for subsequent calls
            if torch.isnan(embeddings).any():
                logger.warning("NaN persists after CPU retry — replacing with zeros")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)

        return embeddings.cpu().numpy()

    def embed_documents(
        self, texts: list[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed document chunks using the Article Encoder.
        Use this at index time for notes, guidelines, etc.

        Returns: numpy array of shape (n_texts, 768)
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._encode_batch(batch, self.article_tokenizer, self.article_model)
            all_embeddings.append(embeddings)

            if show_progress and (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.info(f"  Embedded {min(i + self.batch_size, total)}/{total} chunks")

        result = np.vstack(all_embeddings)
        if show_progress:
            logger.info(f"  Embedded {total} chunks → shape {result.shape}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query using the Query Encoder.
        Use this at retrieval time.

        Returns: numpy array of shape (768,)
        """
        embeddings = self._encode_batch([query], self.query_tokenizer, self.query_model)
        return embeddings[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """
        Embed multiple queries at once.

        Returns: numpy array of shape (n_queries, 768)
        """
        return self._encode_batch(queries, self.query_tokenizer, self.query_model)
