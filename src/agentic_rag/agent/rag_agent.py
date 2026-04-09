"""Agentic RAG orchestrator.

Architecture (ReAct-style loop)
--------------------------------
::

    User Input (text or speech)
           │
    ┌──────▼──────────────────────────┐
    │  1. OBSERVE                      │
    │     STT → plain text query       │
    └──────┬──────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │  2. THINK                        │
    │     LLM classifies the query:    │
    │     RETRIEVE vs DIRECT           │
    └──────┬──────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │  3. ACT (conditional)            │
    │     If RETRIEVE → vector search  │
    │     Return top-k chunks          │
    └──────┬──────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │  4. RESPOND                      │
    │     LLM generates answer with    │
    │     (or without) retrieved ctx   │
    └──────┬──────────────────────────┘
           │
    ┌──────▼──────────────────────────┐
    │  5. SPEAK (optional)             │
    │     TTS → audio playback         │
    └─────────────────────────────────┘

The agent exposes both a synchronous text interface
(:meth:`RAGAgent.query`) and a full voice interface
(:meth:`RAGAgent.listen_and_respond`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.llm.client import LLMClient
from agentic_rag.rag.retriever import Retriever
from agentic_rag.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_RETRIEVAL_DECISION_PROMPT = """\
You are a query classifier. Decide whether the following query should be \
answered using a private knowledge base (RETRIEVE) or from general knowledge (DIRECT).

Rules:
 - RETRIEVE if the query asks about specific documents, company data, policies, \
manuals, or facts that are unlikely to be in general training data.
 - DIRECT if the query is conversational, a general knowledge question, \
a coding question, or a greeting.

Query: {query}

Respond with exactly one word — either RETRIEVE or DIRECT:"""

_RAG_ANSWER_PROMPT = """\
You are a helpful AI assistant. Use the context below to answer the question.
If the context does not contain enough information, say so and answer from \
general knowledge where appropriate.

--- Retrieved Context ---
{context}
--- End Context ---

Question: {question}

Answer:"""

_DIRECT_ANSWER_PROMPT = """\
You are a helpful AI assistant. Answer the following question concisely and \
accurately.

Question: {question}

Answer:"""


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """Structured output from a single agent turn."""

    answer: str
    query: str
    used_retrieval: bool
    retrieved_docs: list[SearchResult] = field(default_factory=list)
    transcription: str = ""  # populated in voice mode

    def __str__(self) -> str:
        return self.answer


# ── Agent ─────────────────────────────────────────────────────────────────────


class RAGAgent:
    """Orchestrates STT → classification → retrieval → LLM → TTS.

    Parameters
    ----------
    settings:
        Application settings.  Defaults to the singleton.
    enable_tts:
        If ``True``, spoken responses are played automatically via TTS.
    enable_stt:
        If ``True``, the :meth:`listen_and_respond` method is available.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        enable_tts: bool = False,
        enable_stt: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = LLMClient(self._settings)
        self._retriever = Retriever(self._settings)
        self._enable_tts = enable_tts
        self._enable_stt = enable_stt

        # Lazy-loaded to avoid importing heavy dependencies unless needed
        self._stt = None
        self._tts = None

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, user_input: str) -> AgentResponse:
        """Process a text query through the full agentic RAG pipeline.

        Parameters
        ----------
        user_input:
            Plain-text question or statement from the user.

        Returns
        -------
        AgentResponse
            Contains the answer, retrieval decision, and source documents.
        """
        user_input = user_input.strip()
        if not user_input:
            return AgentResponse(
                answer="I didn't receive any input. Please try again.",
                query=user_input,
                used_retrieval=False,
            )

        # Step 1: Decide whether retrieval is needed
        use_retrieval = self._should_retrieve(user_input)
        logger.info("Retrieval decision for %r → %s", user_input[:60], use_retrieval)

        retrieved: list[SearchResult] = []

        if use_retrieval:
            # Step 2: Retrieve relevant chunks
            retrieved = self._retriever.retrieve(user_input)

            if retrieved:
                context = self._retriever.format_context(retrieved)
                prompt = _RAG_ANSWER_PROMPT.format(context=context, question=user_input)
            else:
                # Knowledge base is empty or has no relevant docs → fall back
                logger.info("No relevant docs found; falling back to direct answer.")
                prompt = _DIRECT_ANSWER_PROMPT.format(question=user_input)
                use_retrieval = False
        else:
            prompt = _DIRECT_ANSWER_PROMPT.format(question=user_input)

        # Step 3: Generate answer
        llm_response = self._llm.chat(
            user_input=prompt,
            system_prompt=self._settings.llm_system_prompt,
        )
        answer = llm_response.content.strip()

        # Step 4: Optionally speak the response
        if self._enable_tts and answer:
            self._get_tts().speak(answer)

        return AgentResponse(
            answer=answer,
            query=user_input,
            used_retrieval=use_retrieval,
            retrieved_docs=retrieved,
        )

    def listen_and_respond(self) -> AgentResponse:
        """Record microphone input, transcribe, and respond.

        Requires ``enable_stt=True`` when constructing the agent.

        Returns
        -------
        AgentResponse
            Same as :meth:`query` with ``transcription`` populated.
        """
        if not self._enable_stt:
            raise RuntimeError(
                "STT is disabled. Instantiate RAGAgent with enable_stt=True."
            )

        stt = self._get_stt()
        result = stt.record_and_transcribe()

        if not result:
            response = AgentResponse(
                answer="I couldn't understand the audio. Please try again.",
                query="",
                used_retrieval=False,
                transcription="",
            )
            if self._enable_tts:
                self._get_tts().speak(response.answer)
            return response

        logger.info("Transcription: %r", result.text)
        response = self.query(result.text)
        response.transcription = result.text
        return response

    # ── Ingestion convenience methods ─────────────────────────────────────────

    def ingest_file(self, path: str) -> int:
        """Add a document file to the knowledge base.  Returns chunks added."""
        return self._retriever.ingest_file(path)

    def ingest_directory(self, directory: str, *, recursive: bool = True) -> int:
        """Add all supported documents in a directory. Returns chunks added."""
        return self._retriever.ingest_directory(directory, recursive=recursive)

    def document_count(self) -> int:
        """Return total indexed chunks."""
        return self._retriever.document_count()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _should_retrieve(self, query: str) -> bool:
        """Ask the LLM whether the query requires knowledge-base retrieval."""
        # Short-circuit: if the knowledge base is empty, never retrieve
        if self._retriever.document_count() == 0:
            return False

        prompt = _RETRIEVAL_DECISION_PROMPT.format(query=query)
        try:
            decision = self._llm.chat(
                user_input=prompt,
                system_prompt="You are a precise classifier. Respond with one word only.",
            ).content.strip().upper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval decision LLM call failed (%s); defaulting to DIRECT.", exc)
            return False

        return "RETRIEVE" in decision

    def _get_stt(self):
        if self._stt is None:
            from agentic_rag.stt.speech_to_text import SpeechToText

            self._stt = SpeechToText(self._settings)
        return self._stt

    def _get_tts(self):
        if self._tts is None:
            from agentic_rag.tts.text_to_speech import TextToSpeech

            self._tts = TextToSpeech(self._settings)
        return self._tts
