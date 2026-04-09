"""HTTP client for the locally running LLM server.

The server exposes the API format shown below (as used with LM Studio or a
compatible local inference server):

    POST /api/v1/chat
    {
        "model": "qwen3.5-4b",
        "system_prompt": "...",
        "input": "user message here"
    }

The client handles multiple response envelope formats so it stays compatible
with slight server variations without requiring configuration changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentic_rag.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured wrapper around a raw LLM API response."""

    content: str
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMClientError(Exception):
    """Raised when the LLM API returns an error or cannot be reached."""


class LLMClient:
    """Synchronous HTTP client for the local LLM inference server.

    Parameters
    ----------
    settings:
        Application settings. Defaults to the singleton returned by
        :func:`~agentic_rag.config.settings.get_settings`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.Client(
            base_url=self._settings.llm_base_url,
            timeout=self._settings.llm_timeout,
            headers={"Content-Type": "application/json"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM server and return the response.

        Parameters
        ----------
        user_input:
            The user message / query.
        system_prompt:
            Optional override for the system prompt.  Falls back to the value
            in application settings.
        model:
            Optional model override.  Falls back to ``settings.llm_model``.

        Returns
        -------
        LLMResponse
            Parsed response object with a populated ``content`` field.

        Raises
        ------
        LLMClientError
            On HTTP errors, connection failures, or unparseable responses.
        """
        payload: dict[str, Any] = {
            "model": model or self._settings.llm_model,
            "system_prompt": system_prompt or self._settings.llm_system_prompt,
            "input": user_input,
        }

        logger.debug("LLM request → model=%s input=%r", payload["model"], user_input[:120])

        try:
            response = self._http.post("/api/v1/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMClientError(
                f"Cannot connect to LLM server at {self._settings.llm_base_url}. "
                "Make sure the local server is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError("LLM request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(
                f"LLM server returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            raise LLMClientError(
                f"Could not parse LLM response as JSON: {response.text[:300]}"
            ) from exc

        content = self._extract_content(data)
        logger.debug("LLM response ← %r", content[:120])

        return LLMResponse(
            content=content,
            raw=data,
            model=data.get("model", payload["model"]),
            usage=data.get("usage", {}),
        )

    def health_check(self) -> bool:
        """Return ``True`` if the LLM server can be reached, ``False`` otherwise."""
        try:
            self._http.get("/", timeout=5)
            return True
        except httpx.RequestError:
            return False

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Try multiple response envelope formats and return extracted text.

        Supported formats (attempted in order):
        - ``{"response": "..."}``              — common custom format
        - ``{"output": "..."}``                — another common variant
        - ``{"text": "..."}``
        - ``{"content": "..."}``
        - OpenAI-compatible ``{"choices": [{"message": {"content": "..."}}]}``
        - ``{"message": {"content": "..."}}``  — simplified OpenAI wrapper
        - Raw string response
        """
        if isinstance(data, str):
            return data

        if "response" in data and isinstance(data["response"], str):
            return data["response"]

        # LM Studio / custom server: output is a list of message objects
        # e.g. {"output": [{"type": "message", "content": "..."}]}
        if "output" in data:
            output = data["output"]
            if isinstance(output, str):
                return output
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and "content" in first:
                    return str(first["content"])
                if isinstance(first, str):
                    return first

        if "text" in data and isinstance(data["text"], str):
            return data["text"]

        if "content" in data and isinstance(data["content"], str):
            return data["content"]

        # OpenAI-compatible: choices[0].message.content
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or first
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])
                if "text" in first:
                    return str(first["text"])

        # Single message wrapper
        msg = data.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])

        logger.warning("Unknown LLM response format, returning raw JSON string.")
        return str(data)
