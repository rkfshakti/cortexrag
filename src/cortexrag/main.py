"""Command-line interface for the Agentic RAG system.

Commands
--------
::

    agentic-rag ingest  <path>          # index documents
    agentic-rag chat                    # interactive text chat
    agentic-rag voice                   # interactive voice chat (STT + TTS)
    agentic-rag ask    "<question>"     # single text query
    agentic-rag status                  # show current configuration & counts

All commands honour the .env file in the current working directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from cortexrag.config.settings import get_settings

app = typer.Typer(
    name="cortexrag",
    help="CortexRAG — Agentic RAG pipeline with local STT & TTS (QWEN model).",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ── Logging setup ─────────────────────────────────────────────────────────────


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        help="File or directory to ingest into the knowledge base.",
        exists=True,
    ),
    recursive: bool = typer.Option(True, help="Descend into sub-directories."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index documents into the knowledge base."""
    _configure_logging(verbose)
    from cortexrag.agent.rag_agent import RAGAgent

    agent = RAGAgent()

    with console.status(f"[bold green]Ingesting {path} …"):
        if path.is_dir():
            added = agent.ingest_directory(str(path), recursive=recursive)
        else:
            added = agent.ingest_file(str(path))

    console.print(
        Panel(
            f"[green]✓[/green] Indexed [bold]{added}[/bold] new chunk(s).\n"
            f"Total chunks in knowledge base: [bold]{agent.document_count()}[/bold]",
            title="Ingestion Complete",
            border_style="green",
        )
    )


@app.command()
def chat(
    tts: bool = typer.Option(False, "--tts", help="Speak responses aloud."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start an interactive text-based chat session."""
    _configure_logging(verbose)
    from cortexrag.agent.rag_agent import RAGAgent

    agent = RAGAgent(enable_tts=tts)
    settings = get_settings()

    console.print(
        Panel(
            f"Model: [cyan]{settings.llm_model}[/cyan] @ [dim]{settings.llm_base_url}[/dim]\n"
            f"Knowledge base: [cyan]{agent.document_count()}[/cyan] chunk(s) indexed\n"
            f"TTS: [{'green]on' if tts else 'red]off'}[/]\n\n"
            "[dim]Type your question and press Enter.  Type [bold]exit[/bold] or [bold]quit[/bold] to stop.[/dim]",
            title="[bold]Agentic RAG — Text Chat[/bold]",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        with console.status("[dim]Thinking …[/dim]"):
            response = agent.query(user_input)

        _print_response(response)


@app.command()
def voice(
    no_tts: bool = typer.Option(False, "--no-tts", help="Disable spoken responses."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start an interactive voice chat session (STT + TTS)."""
    _configure_logging(verbose)
    from cortexrag.agent.rag_agent import RAGAgent

    agent = RAGAgent(enable_stt=True, enable_tts=not no_tts)
    settings = get_settings()

    console.print(
        Panel(
            f"Model: [cyan]{settings.llm_model}[/cyan] @ [dim]{settings.llm_base_url}[/dim]\n"
            f"STT model: [cyan]{settings.whisper_model}[/cyan]\n"
            f"TTS voice: [cyan]{settings.tts_voice}[/cyan]\n\n"
            "[dim]Press [bold]Ctrl+C[/bold] to stop.[/dim]",
            title="[bold]Agentic RAG — Voice Chat[/bold]",
            border_style="magenta",
        )
    )

    turn = 0
    while True:
        try:
            turn += 1
            console.print(f"\n[bold magenta][Turn {turn}][/bold magenta] [dim]Listening … (speak now)[/dim]")

            with console.status("[dim]Listening …[/dim]"):
                response = agent.listen_and_respond()

            if response.transcription:
                console.print(f"[bold cyan]You (transcribed):[/bold cyan] {response.transcription}")

            _print_response(response)

        except KeyboardInterrupt:
            console.print("\n[dim]Voice session ended.[/dim]")
            break


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the agent."),
    tts: bool = typer.Option(False, "--tts", help="Speak the response aloud."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask a single question and print the answer."""
    _configure_logging(verbose)
    from cortexrag.agent.rag_agent import RAGAgent

    agent = RAGAgent(enable_tts=tts)
    with console.status("[dim]Thinking …[/dim]"):
        response = agent.query(question)
    _print_response(response)


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Display current configuration and knowledge-base statistics."""
    _configure_logging(verbose)
    settings = get_settings()

    from cortexrag.llm.client import LLMClient
    from cortexrag.rag.retriever import Retriever

    llm = LLMClient(settings)
    retriever = Retriever(settings)

    table = Table(title="Agentic RAG Status", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    # LLM
    reachable = llm.health_check()
    table.add_row("LLM Server", settings.llm_base_url)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row(
        "Server Status",
        "[green]✓ reachable[/green]" if reachable else "[red]✗ unreachable[/red]",
    )
    # RAG
    table.add_row("Vector Store", settings.vector_store_path)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Indexed Chunks", str(retriever.document_count()))
    table.add_row("Retrieval Top-K", str(settings.retrieval_top_k))
    # STT
    table.add_row("Whisper Model", settings.whisper_model)
    table.add_row("Whisper Device", settings.whisper_device)
    # TTS
    table.add_row("TTS Voice", settings.tts_voice)

    console.print(table)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_response(response) -> None:
    from cortexrag.agent.rag_agent import AgentResponse

    r: AgentResponse = response
    retrieval_badge = (
        "[green]RAG[/green]" if r.used_retrieval else "[blue]Direct[/blue]"
    )

    console.print(f"\n[bold green]Assistant[/bold green] [{retrieval_badge}]:")
    console.print(Markdown(r.answer))

    if r.used_retrieval and r.retrieved_docs:
        sources = sorted({doc.source for doc in r.retrieved_docs})
        console.print(f"\n[dim]Sources: {', '.join(sources)}[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    app()


if __name__ == "__main__":
    main()
