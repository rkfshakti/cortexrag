"""Tests for the LLM HTTP client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.llm.client import LLMClient, LLMClientError, LLMResponse


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:1234",
        llm_model="test-model",
        llm_system_prompt="You are a test assistant.",
        llm_timeout=10,
    )


@pytest.fixture
def client(settings: Settings) -> LLMClient:
    return LLMClient(settings)


class TestLLMClientExtractContent:
    """Unit tests for the static response-parsing helper."""

    def test_response_key(self, client: LLMClient) -> None:
        assert client._extract_content({"response": "hello"}) == "hello"

    def test_output_key(self, client: LLMClient) -> None:
        assert client._extract_content({"output": "world"}) == "world"

    def test_text_key(self, client: LLMClient) -> None:
        assert client._extract_content({"text": "foo"}) == "foo"

    def test_content_key(self, client: LLMClient) -> None:
        assert client._extract_content({"content": "bar"}) == "bar"

    def test_openai_choices_format(self, client: LLMClient) -> None:
        data = {"choices": [{"message": {"content": "openai answer"}}]}
        assert client._extract_content(data) == "openai answer"

    def test_message_wrapper(self, client: LLMClient) -> None:
        data = {"message": {"content": "wrapped"}}
        assert client._extract_content(data) == "wrapped"

    def test_raw_string(self, client: LLMClient) -> None:
        assert client._extract_content("just a string") == "just a string"

    def test_unknown_format_returns_str(self, client: LLMClient) -> None:
        data = {"unknown_key": "value"}
        result = client._extract_content(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestLLMClientChat:
    """Integration-style tests with mocked HTTP."""

    def test_successful_chat(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello there!"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response):
            result: LLMResponse = client.chat("Hi")

        assert result.content == "Hello there!"
        assert result.model == "test-model"

    def test_connect_error_raises_client_error(self, client: LLMClient) -> None:
        with patch.object(client._http, "post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(LLMClientError, match="Cannot connect"):
                client.chat("Hi")

    def test_timeout_raises_client_error(self, client: LLMClient) -> None:
        with patch.object(client._http, "post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(LLMClientError, match="timed out"):
                client.chat("Hi")

    def test_http_status_error_raises_client_error(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        error = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)
        mock_response.raise_for_status.side_effect = error

        with patch.object(client._http, "post", return_value=mock_response):
            with pytest.raises(LLMClientError, match="HTTP 500"):
                client.chat("Hi")

    def test_system_prompt_override(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response) as mock_post:
            client.chat("test", system_prompt="Custom prompt")

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["system_prompt"] == "Custom prompt"

    def test_model_override(self, client: LLMClient) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response) as mock_post:
            client.chat("test", model="other-model")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "other-model"
