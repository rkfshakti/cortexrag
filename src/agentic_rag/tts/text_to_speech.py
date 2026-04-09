"""Text-to-Speech using Microsoft Edge TTS.

*edge-tts* calls Microsoft's free streaming TTS service (the same engine that
powers the Edge browser's Read Aloud feature) without requiring an API key.

Workflow
--------
1. :meth:`TextToSpeech.synthesize` — convert text to an MP3 byte stream or
   save it to a file.
2. :meth:`TextToSpeech.speak` — synthesize and immediately play the audio
   through the default output device (uses *pygame* for playback).
3. :meth:`TextToSpeech.list_voices` — list all available Azure Neural voices.

All synthesis calls are *async* under the hood; the public interface wraps
them with ``asyncio.run`` so callers need not manage an event loop.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from agentic_rag.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Converts text to audio using Microsoft Edge TTS.

    Parameters
    ----------
    settings:
        Application settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(self, text: str, *, output_path: str | Path | None = None) -> Path:
        """Convert *text* to an MP3 file.

        Parameters
        ----------
        text:
            Text to synthesize.  Long text is handled automatically by the
            Edge TTS service.
        output_path:
            Destination path for the MP3.  If ``None``, a temporary file is
            created.

        Returns
        -------
        Path
            Path to the synthesized MP3 file.
        """
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = Path(tmp.name)
        else:
            output_path = Path(output_path)

        start = time.perf_counter()
        asyncio.run(self._async_synthesize(text, output_path))
        elapsed = time.perf_counter() - start
        logger.debug("Synthesis took %.2fs → %s", elapsed, output_path)
        return output_path

    def speak(self, text: str) -> None:
        """Synthesize *text* and play it through the default audio output.

        Blocks until playback is complete.

        Parameters
        ----------
        text:
            Text to speak aloud.
        """
        if not text.strip():
            return

        audio_path = self.synthesize(text)
        try:
            self._play_audio(audio_path)
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def list_voices() -> list[dict]:
        """Return a list of all available Azure Neural voice descriptors.

        Each item contains at least ``ShortName``, ``Gender``, and ``Locale``.
        """
        return asyncio.run(TextToSpeech._async_list_voices())

    # ── Private async helpers ─────────────────────────────────────────────────

    async def _async_synthesize(self, text: str, output_path: Path) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError("Install 'edge-tts': pip install edge-tts") from exc

        communicate = edge_tts.Communicate(
            text=text,
            voice=self._settings.tts_voice,
            rate=self._settings.tts_rate,
            volume=self._settings.tts_volume,
        )
        await communicate.save(str(output_path))

    @staticmethod
    async def _async_list_voices() -> list[dict]:
        try:
            import edge_tts
        except ImportError as exc:
            raise ImportError("Install 'edge-tts': pip install edge-tts") from exc

        voices = await edge_tts.list_voices()
        return list(voices)

    # ── Audio playback ────────────────────────────────────────────────────────

    @staticmethod
    def _play_audio(path: Path) -> None:
        """Play an MP3/WAV file using pygame.mixer (blocks until done)."""
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Install 'pygame': pip install pygame") from exc

        pygame.mixer.init()
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            # Block until playback finishes
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            pygame.mixer.quit()
