"""Top-level main.py — convenience wrapper for direct execution.

Usage::

    python main.py --help
    python main.py chat
    python main.py ingest data/documents/
"""

from agentic_rag.main import app

if __name__ == "__main__":
    app()
