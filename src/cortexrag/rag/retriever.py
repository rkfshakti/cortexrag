"""High-level retriever: thin façade over :class:`VectorStore`.

The retriever is the component the agent interacts with.  It formats search
results into a clean context string ready to be injected into an LLM prompt.
"""

from __future__ import annotations

import logging

from cortexrag.config.settings import Settings, get_settings
from cortexrag.rag.document_loader import DocumentLoader, DocumentChunk
from cortexrag.rag.embedder import Embedder
from cortexrag.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant document chunks for a query and format them as context.

    Parameters
    ----------
    settings:
        Application settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embedder = Embedder(self._settings)
        self._store = VectorStore(self._settings, self._embedder)
        self._loader = DocumentLoader(self._settings)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, path: str) -> int:
        """Load a file, chunk it, and add the chunks to the vector store.

        Returns the number of new chunks added.
        """
        chunks: list[DocumentChunk] = self._loader.load_file(path)
        return self._store.add_chunks(chunks)

    def ingest_directory(self, directory: str, *, recursive: bool = True) -> int:
        """Ingest all supported documents in *directory*.

        Returns the total number of new chunks added.
        """
        chunks: list[DocumentChunk] = self._loader.load_directory(
            directory, recursive=recursive
        )
        return self._store.add_chunks(chunks)

    def ingest_text(self, text: str, *, source: str = "<inline>") -> int:
        """Chunk and ingest an arbitrary string (e.g. a web page)."""
        chunks: list[DocumentChunk] = self._loader.load_text(text, source=source)
        return self._store.add_chunks(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        """Return the top-*k* most relevant chunks for *query*."""
        return self._store.search(query, top_k=top_k)

    def format_context(self, results: list[SearchResult]) -> str:
        """Render retrieved results as a numbered context block for the LLM.

        Example output::

            [1] Source: /data/documents/manual.pdf (chunk 3, score: 0.87)
            The system shall respond within 200 ms...

            [2] Source: /data/documents/faq.txt (chunk 1, score: 0.74)
            ...
        """
        if not results:
            return "No relevant documents found in the knowledge base."

        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            header = (
                f"[{i}] Source: {result.source} "
                f"(chunk {result.chunk_index}, score: {result.score:.2f})"
            )
            parts.append(f"{header}\n{result.text.strip()}")

        return "\n\n".join(parts)

    # ── Utility ───────────────────────────────────────────────────────────────

    def document_count(self) -> int:
        """Return the total number of indexed chunks."""
        return self._store.count()

    def delete_source(self, source: str) -> int:
        """Remove all chunks for *source* from the index."""
        return self._store.delete_source(source)

    def reset(self) -> None:
        """Delete all documents from the knowledge base (irreversible)."""
        self._store.reset()
