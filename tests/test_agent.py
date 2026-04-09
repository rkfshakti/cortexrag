"""Tests for the RAGAgent orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.agent.rag_agent import AgentResponse, RAGAgent
from agentic_rag.config.settings import Settings
from agentic_rag.rag.vector_store import SearchResult


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:1234",
        llm_model="test-model",
        vector_store_path="/tmp/test_chroma",
        retrieval_top_k=2,
        similarity_threshold=0.0,
    )


def _make_agent(settings: Settings) -> RAGAgent:
    """Return a RAGAgent with all heavy dependencies mocked out."""
    agent = RAGAgent(settings)

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm_response = MagicMock()
    mock_llm_response.content = "This is a test answer."
    mock_llm.chat.return_value = mock_llm_response
    agent._llm = mock_llm

    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.document_count.return_value = 5
    mock_retriever.retrieve.return_value = [
        SearchResult(
            text="Relevant document text.",
            source="/data/doc.txt",
            chunk_index=0,
            score=0.85,
            metadata={},
        )
    ]
    mock_retriever.format_context.return_value = "[1] Relevant document text."
    agent._retriever = mock_retriever

    return agent


class TestRAGAgentQuery:
    def test_returns_agent_response(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        # Force retrieval path
        agent._llm.chat.side_effect = [
            MagicMock(content="RETRIEVE"),  # decision call
            MagicMock(content="Final answer."),  # answer call
        ]
        result = agent.query("What is the policy?")
        assert isinstance(result, AgentResponse)
        assert result.answer == "Final answer."
        assert result.query == "What is the policy?"

    def test_direct_path_when_decision_is_direct(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._llm.chat.side_effect = [
            MagicMock(content="DIRECT"),  # decision call
            MagicMock(content="General answer."),  # answer call
        ]
        result = agent.query("What is 2 + 2?")
        assert result.used_retrieval is False
        assert result.answer == "General answer."
        # Retriever should NOT have been invoked
        agent._retriever.retrieve.assert_not_called()

    def test_retrieval_path_populates_docs(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._llm.chat.side_effect = [
            MagicMock(content="RETRIEVE"),
            MagicMock(content="Retrieved answer."),
        ]
        result = agent.query("Tell me about the document.")
        assert result.used_retrieval is True
        assert len(result.retrieved_docs) >= 1

    def test_empty_input_returns_gentle_error(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        result = agent.query("   ")
        assert "didn't receive" in result.answer.lower()
        assert result.used_retrieval is False

    def test_empty_knowledge_base_skips_retrieval(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._retriever.document_count.return_value = 0
        agent._llm.chat.return_value = MagicMock(content="Direct fallback.")

        result = agent.query("Any question")
        assert result.used_retrieval is False
        # _should_retrieve should short-circuit without calling the LLM for decision
        # Only one LLM call: the answer generation
        assert agent._llm.chat.call_count == 1

    def test_retrieval_decision_failure_falls_back_to_direct(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        decision_call = [True]

        def side_effect(user_input, **_):
            if decision_call[0]:
                decision_call[0] = False
                raise RuntimeError("LLM unreachable")
            return MagicMock(content="Fallback answer.")

        agent._llm.chat.side_effect = side_effect
        result = agent.query("What's in the docs?")
        assert result.used_retrieval is False

    def test_no_docs_found_falls_back_to_direct(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._llm.chat.side_effect = [
            MagicMock(content="RETRIEVE"),
            MagicMock(content="No context answer."),
        ]
        agent._retriever.retrieve.return_value = []

        result = agent.query("Something obscure")
        assert result.used_retrieval is False


class TestRAGAgentVoice:
    def test_listen_and_respond_without_stt_enabled_raises(self, settings: Settings) -> None:
        agent = RAGAgent(settings, enable_stt=False)
        with pytest.raises(RuntimeError, match="STT is disabled"):
            agent.listen_and_respond()

    def test_listen_and_respond_with_empty_transcription(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._enable_stt = True

        mock_stt = MagicMock()
        from agentic_rag.stt.speech_to_text import TranscriptionResult

        mock_stt.record_and_transcribe.return_value = TranscriptionResult(text="")
        agent._stt = mock_stt

        result = agent.listen_and_respond()
        assert "couldn't understand" in result.answer.lower()

    def test_listen_and_respond_success(self, settings: Settings) -> None:
        agent = _make_agent(settings)
        agent._enable_stt = True
        agent._llm.chat.side_effect = [
            MagicMock(content="DIRECT"),
            MagicMock(content="Voice answer."),
        ]

        mock_stt = MagicMock()
        from agentic_rag.stt.speech_to_text import TranscriptionResult

        mock_stt.record_and_transcribe.return_value = TranscriptionResult(
            text="Hello from voice", language="en"
        )
        agent._stt = mock_stt

        result = agent.listen_and_respond()
        assert result.transcription == "Hello from voice"
        assert result.answer == "Voice answer."
