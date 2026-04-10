"""Application settings loaded from environment variables / .env file.

All settings can be overridden via environment variables prefixed with
``CORTEXRAG_``, e.g. ``CORTEXRAG_LLM_MODEL=my-model``.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for every sub-system."""

    model_config = SettingsConfigDict(
        env_prefix="CORTEXRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_base_url: str = Field(
        default="http://192.168.68.113:1234",
        description="Base URL of the local LLM server",
    )
    llm_model: str = Field(
        default="qwen3.5-4b",
        description="Model identifier sent in every API request",
    )
    llm_system_prompt: str = Field(
        default=(
            "You are a helpful AI assistant with access to a knowledge base. "
            "Answer concisely and accurately using the provided context when available. "
            "If you are unsure about something, say so."
        ),
        description="Default system prompt for the LLM",
    )
    llm_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="HTTP request timeout in seconds",
    )

    # ── RAG ───────────────────────────────────────────────────────────────────
    vector_store_path: str = Field(
        default="./data/chroma_db",
        description="Directory to persist the ChromaDB vector store",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum character length of each document chunk",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Character overlap between consecutive chunks",
    )
    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of documents to retrieve per query",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for retrieved documents",
    )

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
        default="base",
        description="Whisper model variant (smaller = faster, larger = more accurate)",
    )
    whisper_device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device to run Whisper on",
    )
    whisper_compute_type: Literal["int8", "float16", "float32"] = Field(
        default="int8",
        description="Quantisation type for Whisper weights",
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_voice: str = Field(
        default="en-US-AriaNeural",
        description="Edge TTS voice name (run `edge-tts --list-voices` to see all)",
    )
    tts_rate: str = Field(
        default="+0%",
        description="Speech rate adjustment, e.g. +10% or -5%",
    )
    tts_volume: str = Field(
        default="+0%",
        description="Speech volume adjustment, e.g. +10% or -5%",
    )

    # ── Audio recording ───────────────────────────────────────────────────────
    audio_sample_rate: int = Field(
        default=16000,
        description="Recording sample rate in Hz",
    )
    audio_silence_duration: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Seconds of silence that triggers end-of-speech detection",
    )
    audio_max_duration: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Maximum single recording duration in seconds",
    )

    @field_validator("llm_base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance."""
    return Settings()
