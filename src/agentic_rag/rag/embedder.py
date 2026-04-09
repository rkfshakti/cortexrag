"""Sentence embedding using HuggingFace sentence-transformers.

The model is loaded once and cached for the lifetime of the :class:`Embedder`
instance.  The default ``all-MiniLM-L6-v2`` model is small (~80 MB), fast on
CPU, and produces 384-dimensional embeddings that work well with cosine
similarity.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from agentic_rag.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers model to produce dense vector embeddings.

    Parameters
    ----------
    settings:
        Application settings.  The ``embedding_model`` field controls which
        model is loaded.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.embedding_model
        self._model = None  # lazy-loaded on first use

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def model(self):
        """Return (and lazily initialise) the sentence-transformers model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install 'sentence-transformers': pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded.")
        return self._model

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single string.

        Parameters
        ----------
        text:
            Input text to embed.

        Returns
        -------
        list[float]
            1-D list of floats representing the dense embedding.
        """
        vector: np.ndarray = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of strings.

        Parameters
        ----------
        texts:
            Sequence of strings to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []

        logger.debug("Embedding batch of %d texts", len(texts))
        vectors: np.ndarray = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors (e.g. 384 for MiniLM)."""
        return self.model.get_sentence_embedding_dimension()
