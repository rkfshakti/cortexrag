"""Tests for the TTS module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.tts.text_to_speech import TextToSpeech


@pytest.fixture
def settings() -> Settings:
    return Settings(
        tts_voice="en-US-AriaNeural",
        tts_rate="+0%",
        tts_volume="+0%",
    )


@pytest.fixture
def tts(settings: Settings) -> TextToSpeech:
    return TextToSpeech(settings)


class TestTextToSpeech:
    def test_synthesize_empty_text_raises(self, tts: TextToSpeech) -> None:
        with pytest.raises(ValueError, match="empty"):
            tts.synthesize("   ")

    def test_synthesize_creates_file(self, tts: TextToSpeech, tmp_path: Path) -> None:
        output = tmp_path / "out.mp3"

        async def fake_save(path: str) -> None:
            Path(path).write_bytes(b"fake-mp3-data")

        mock_communicate = MagicMock()
        mock_communicate.save = fake_save

        with patch("agentic_rag.tts.text_to_speech.edge_tts", create=True):
            with patch(
                "agentic_rag.tts.text_to_speech.TextToSpeech._async_synthesize",
                new_callable=AsyncMock,
                side_effect=lambda text, path: Path(path).write_bytes(b"mp3"),
            ):
                result = tts.synthesize("Hello world", output_path=output)

        assert result == output

    def test_synthesize_uses_temp_file_when_no_path(self, tts: TextToSpeech) -> None:
        with patch(
            "agentic_rag.tts.text_to_speech.TextToSpeech._async_synthesize",
            new_callable=AsyncMock,
            side_effect=lambda text, path: path.write_bytes(b"mp3"),
        ):
            result = tts.synthesize("test audio")

        assert result.suffix == ".mp3"
        # Clean up
        result.unlink(missing_ok=True)

    def test_speak_calls_synthesize_and_play(self, tts: TextToSpeech, tmp_path: Path) -> None:
        fake_mp3 = tmp_path / "speak.mp3"
        fake_mp3.write_bytes(b"mp3data")

        with patch.object(tts, "synthesize", return_value=fake_mp3) as mock_synth:
            with patch.object(TextToSpeech, "_play_audio") as mock_play:
                tts.speak("Hello!")

        mock_synth.assert_called_once_with("Hello!")
        mock_play.assert_called_once_with(fake_mp3)

    def test_speak_empty_text_is_noop(self, tts: TextToSpeech) -> None:
        with patch.object(tts, "synthesize") as mock_synth:
            tts.speak("")
        mock_synth.assert_not_called()
