"""ChromaDB-backed persistent vector store.

Documents are stored in a local SQLite + HNSW index managed by ChromaDB.
All data persists across restarts in the directory specified by
``settings.vector_store_path``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from cortexrag.config.settings import Settings, get_settings
from cortexrag.rag.document_loader import DocumentChunk
from cortexrag.rag.embedder import Embedder

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "knowledge_base"


@dataclass
class SearchResult:
    """A single retrieved document with its relevance score."""

    text: str
    source: str
    chunk_index: int
    score: float  # cosine similarity, higher = more relevant
    metadata: dict[str, Any]


class VectorStore:
    """Persistent vector store backed by ChromaDB.

    Parameters
    ----------
    settings:
        Application settings.
    embedder:
        Pre-instantiated :class:`~cortexrag.rag.embedder.Embedder`.  If
        omitted a new one is created using ``settings``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or Embedder(self._settings)
        self._client = None
        self._collection = None

    # ── Lazy initialisation ───────────────────────────────────────────────────

    @property
    def collection(self):
        """Return (and lazily create) the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._init_collection()
        return self._collection

    def _init_collection(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install 'chromadb': pip install chromadb") from exc

        logger.info("Initialising ChromaDB at %s", self._settings.vector_store_path)
        self._client = chromadb.PersistentClient(path=self._settings.vector_store_path)
        collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store ready — %d document(s) in collection.", collection.count()
        )
        return collection

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Embed and store a sequence of :class:`DocumentChunk` objects.

        Already-indexed chunks (same source + chunk_index) are skipped to
        avoid duplication.

        Parameters
        ----------
        chunks:
            Chunks to index.

        Returns
        -------
        int
            Number of *new* chunks actually added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        ids = [self._chunk_id(c) for c in chunks]

        # Check for existing documents
        existing = set(self.collection.get(ids=ids)["ids"])
        new_chunks = [c for c, doc_id in zip(chunks, ids) if doc_id not in existing]
        new_ids = [doc_id for doc_id in ids if doc_id not in existing]

        if not new_chunks:
            logger.info("All %d chunks are already indexed — skipping.", len(chunks))
            return 0

        new_texts = [c.text for c in new_chunks]
        embeddings = self._embedder.embed_batch(new_texts)
        metadatas = [
            {"source": c.source, "chunk_index": c.chunk_index, **c.metadata}
            for c in new_chunks
        ]

        self.collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=new_texts,
            metadatas=metadatas,
        )

        logger.info("Added %d new chunk(s) to the vector store.", len(new_chunks))
        return len(new_chunks)

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve the most semantically similar chunks for *query*.

        Parameters
        ----------
        query:
            User query string.
        top_k:
            Number of results to return.  Defaults to ``settings.retrieval_top_k``.

        Returns
        -------
        list[SearchResult]
            Results sorted by descending similarity score, filtered by
            ``settings.similarity_threshold``.
        """
        k = top_k or self._settings.retrieval_top_k
        count = self.collection.count()
        if count == 0:
            logger.warning("Vector store is empty — no documents to retrieve.")
            return []

        k = min(k, count)
        query_embedding = self._embedder.embed(query)

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity: 1 - dist/2 (produces [0, 1])
            score = 1.0 - dist / 2.0
            if score < self._settings.similarity_threshold:
                continue
            results.append(
                SearchResult(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=score,
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        logger.debug("Retrieved %d result(s) for query: %r", len(results), query[:80])
        return results

    def delete_source(self, source: str) -> int:
        """Remove all chunks whose source matches *source*.

        Returns the number of deleted documents.
        """
        existing = self.collection.get(where={"source": source})
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
            logger.info("Deleted %d chunk(s) from source: %s", len(ids), source)
        return len(ids)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete **all** documents from the collection (irreversible)."""
        logger.warning("Resetting the entire vector store.")
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._init_collection()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        """Derive a deterministic ID from source path and chunk index."""
        namespace = uuid.NAMESPACE_URL
        return str(uuid.uuid5(namespace, f"{chunk.source}::{chunk.chunk_index}"))
