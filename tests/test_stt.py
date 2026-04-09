"""Tests for the STT module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.stt.speech_to_text import SpeechToText, TranscriptionResult


@pytest.fixture
def settings() -> Settings:
    return Settings(
        whisper_model="base",
        whisper_device="cpu",
        whisper_compute_type="int8",
        audio_sample_rate=16000,
        audio_silence_duration=1.0,
        audio_max_duration=10,
    )


@pytest.fixture
def stt(settings: Settings) -> SpeechToText:
    return SpeechToText(settings)


class TestTranscriptionResult:
    def test_bool_true_on_text(self) -> None:
        r = TranscriptionResult(text="hello world")
        assert bool(r) is True

    def test_bool_false_on_empty(self) -> None:
        r = TranscriptionResult(text="   ")
        assert bool(r) is False


class TestSpeechToText:
    def _attach_mock_model(self, stt: SpeechToText, text: str = "test transcription") -> None:
        """Attach a mock faster-whisper model to the STT instance."""
        mock_segment = MagicMock()
        mock_segment.text = text
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        stt._model = mock_model

    def test_transcribe_file_returns_result(self, stt: SpeechToText, tmp_path: Path) -> None:
        import soundfile as sf

        self._attach_mock_model(stt, "hello from whisper")

        # Write a short silent WAV
        wav = tmp_path / "test.wav"
        sf.write(str(wav), np.zeros(16000), 16000)

        result = stt.transcribe_file(wav)
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello from whisper"
        assert result.language == "en"

    def test_transcribe_missing_file_raises(self, stt: SpeechToText) -> None:
        self._attach_mock_model(stt)
        with pytest.raises(FileNotFoundError):
            stt.transcribe_file("/no/such/file.wav")

    def test_model_lazy_load(self, stt: SpeechToText) -> None:
        assert stt._model is None

    def test_record_and_transcribe_cleans_up(self, stt: SpeechToText, tmp_path: Path) -> None:
        self._attach_mock_model(stt, "cleaned up")

        fake_audio = tmp_path / "temp.wav"
        import soundfile as sf
        sf.write(str(fake_audio), np.zeros(16000), 16000)

        with patch.object(stt, "record", return_value=fake_audio):
            result = stt.record_and_transcribe()

        assert result.text == "cleaned up"
        # Temp file should be removed
        assert not fake_audio.exists()
