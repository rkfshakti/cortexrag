"""Document loader: reads files from disk and splits them into chunks.

Supported formats
-----------------
- Plain text  (``.txt``)
- Markdown    (``.md``, ``.markdown``)
- PDF         (``.pdf``) via *pypdf*

Each resulting :class:`DocumentChunk` carries the source path and chunk index
so the retriever can cite origins in the final answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from cortexrag.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single text chunk produced by the document loader."""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:60].replace("\n", " ")
        return f"DocumentChunk(source={self.source!r}, index={self.chunk_index}, text={preview!r}...)"


class DocumentLoader:
    """Loads documents from individual files or entire directories and chunks them.

    Parameters
    ----------
    settings:
        Application settings. Defaults to the singleton.
    """

    _SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".pdf"})

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> list[DocumentChunk]:
        """Load and chunk a single file.

        Parameters
        ----------
        path:
            Absolute or relative path to the file.

        Returns
        -------
        list[DocumentChunk]
            Ordered list of text chunks from the file.
        """
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self._SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {ext!r}. "
                f"Supported: {sorted(self._SUPPORTED_EXTENSIONS)}"
            )

        logger.info("Loading %s", path)
        raw_text = self._read_file(path)
        return self._chunk(raw_text, source=str(path))

    def load_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> list[DocumentChunk]:
        """Load all supported documents from a directory.

        Parameters
        ----------
        directory:
            Path to the directory.
        recursive:
            If ``True`` (default), descend into sub-directories.

        Returns
        -------
        list[DocumentChunk]
            Concatenated chunks from all discovered files.
        """
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        files: list[Path] = [
            p
            for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in self._SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning("No supported files found in %s", directory)
            return []

        chunks: list[DocumentChunk] = []
        for file_path in sorted(files):
            try:
                chunks.extend(self.load_file(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s — %s", file_path.name, exc)

        logger.info("Loaded %d chunks from %d file(s) in %s", len(chunks), len(files), directory)
        return chunks

    def load_text(self, text: str, *, source: str = "<inline>") -> list[DocumentChunk]:
        """Chunk an already-loaded string.

        Useful for ingesting dynamically generated content or web pages.
        """
        return self._chunk(text, source=source)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_file(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install 'pypdf' to load PDF files: pip install pypdf") from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def _chunk(self, text: str, *, source: str) -> list[DocumentChunk]:
        if not text.strip():
            logger.warning("Empty document: %s", source)
            return []

        raw_chunks: Sequence[str] = self._splitter.split_text(text)
        return [
            DocumentChunk(text=chunk, source=source, chunk_index=i)
            for i, chunk in enumerate(raw_chunks)
        ]
