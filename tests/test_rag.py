"""Tests for the RAG sub-system (document loader, embedder, vector store, retriever)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.rag.document_loader import DocumentChunk, DocumentLoader
from agentic_rag.rag.embedder import Embedder
from agentic_rag.rag.vector_store import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chunk_size=200,
        chunk_overlap=20,
        retrieval_top_k=3,
        similarity_threshold=0.0,
        vector_store_path=tempfile.mkdtemp(),
    )


@pytest.fixture
def loader(settings: Settings) -> DocumentLoader:
    return DocumentLoader(settings)


# ── DocumentLoader ────────────────────────────────────────────────────────────


class TestDocumentLoader:
    def test_load_txt_file(self, loader: DocumentLoader, tmp_path: Path) -> None:
        txt = tmp_path / "sample.txt"
        txt.write_text("Hello world. " * 50, encoding="utf-8")
        chunks = loader.load_file(txt)
        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source == str(txt) for c in chunks)

    def test_load_markdown_file(self, loader: DocumentLoader, tmp_path: Path) -> None:
        md = tmp_path / "notes.md"
        md.write_text("# Title\n\nSome content here.\n", encoding="utf-8")
        chunks = loader.load_file(md)
        assert len(chunks) >= 1

    def test_unsupported_extension_raises(self, loader: DocumentLoader, tmp_path: Path) -> None:
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            loader.load_file(bad)

    def test_missing_file_raises(self, loader: DocumentLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.txt")

    def test_load_directory(self, loader: DocumentLoader, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text("Content " * 30, encoding="utf-8")
        chunks = loader.load_directory(tmp_path)
        assert len(chunks) >= 3

    def test_load_directory_missing_raises(self, loader: DocumentLoader) -> None:
        with pytest.raises(NotADirectoryError):
            loader.load_directory("/no/such/dir")

    def test_load_text(self, loader: DocumentLoader) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = loader.load_text(text, source="test-source")
        assert len(chunks) >= 1
        assert chunks[0].source == "test-source"
        assert chunks[0].chunk_index == 0

    def test_empty_text_returns_no_chunks(self, loader: DocumentLoader) -> None:
        chunks = loader.load_text("   ", source="empty")
        assert chunks == []

    def test_chunk_indices_are_sequential(self, loader: DocumentLoader) -> None:
        text = "Word " * 500
        chunks = loader.load_text(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))


# ── Embedder ──────────────────────────────────────────────────────────────────


class TestEmbedder:
    """Use a mocked sentence-transformers model to avoid downloading weights."""

    def _make_embedder(self, settings: Settings) -> Embedder:
        embedder = Embedder(settings)
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: (
            __import__("numpy").ones(384) if isinstance(texts, str)
            else __import__("numpy").ones((len(texts), 384))
        )
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder._model = mock_model
        return embedder

    def test_embed_returns_list_of_floats(self, settings: Settings) -> None:
        embedder = self._make_embedder(settings)
        vec = embedder.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_batch_returns_list_of_lists(self, settings: Settings) -> None:
        embedder = self._make_embedder(settings)
        vecs = embedder.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_embed_batch_empty_returns_empty(self, settings: Settings) -> None:
        embedder = self._make_embedder(settings)
        assert embedder.embed_batch([]) == []

    def test_dimension_property(self, settings: Settings) -> None:
        embedder = self._make_embedder(settings)
        assert embedder.dimension == 384


# ── VectorStore ───────────────────────────────────────────────────────────────


class TestVectorStore:
    def _make_store(self, settings: Settings) -> VectorStore:
        """Build a VectorStore with a fully mocked embedder and ChromaDB."""
        import numpy as np

        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = np.ones(384).tolist()
        embedder.embed_batch.side_effect = lambda texts: [np.ones(384).tolist() for _ in texts]

        store = VectorStore(settings, embedder)

        # Replace ChromaDB collection with an in-memory mock
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        store._collection = mock_collection
        return store

    def test_add_chunks_calls_collection_add(self, settings: Settings) -> None:
        store = self._make_store(settings)
        chunks = [DocumentChunk(text="hello", source="test.txt", chunk_index=0)]
        added = store.add_chunks(chunks)
        assert added == 1
        store._collection.add.assert_called_once()

    def test_add_chunks_skips_duplicates(self, settings: Settings) -> None:
        store = self._make_store(settings)
        from agentic_rag.rag.vector_store import VectorStore as VS

        chunk = DocumentChunk(text="hello", source="test.txt", chunk_index=0)
        existing_id = VS._chunk_id(chunk)
        store._collection.get.return_value = {"ids": [existing_id]}

        added = store.add_chunks([chunk])
        assert added == 0
        store._collection.add.assert_not_called()

    def test_search_empty_store_returns_empty(self, settings: Settings) -> None:
        store = self._make_store(settings)
        store._collection.count.return_value = 0
        results = store.search("anything")
        assert results == []

    def test_count_delegates_to_collection(self, settings: Settings) -> None:
        store = self._make_store(settings)
        store._collection.count.return_value = 42
        assert store.count() == 42
