"""Speech-to-Text using *faster-whisper* (local, no API key required).

Workflow
--------
1. :meth:`SpeechToText.record` — capture microphone audio with automatic
   silence-based end detection (VAD) using energy thresholding.
2. :meth:`SpeechToText.transcribe_file` — transcribe a pre-recorded WAV/MP3.
3. :meth:`SpeechToText.record_and_transcribe` — convenience method combining
   both steps.

The Whisper model is loaded lazily on first use and cached.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from cortexrag.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_ENERGY_THRESHOLD = 0.01  # RMS amplitude below which audio is considered silence


@dataclass
class TranscriptionResult:
    """Result of a Whisper transcription."""

    text: str
    language: str = "en"
    duration_seconds: float = 0.0
    segments: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class SpeechToText:
    """Records microphone input and transcribes it with faster-whisper.

    Parameters
    ----------
    settings:
        Application settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None  # lazy-loaded

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self):
        """Return (and lazily load) the faster-whisper model."""
        if self._model is None:
            logger.info(
                "Loading Whisper model: %s on %s (%s)",
                self._settings.whisper_model,
                self._settings.whisper_device,
                self._settings.whisper_compute_type,
            )
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "Install 'faster-whisper': pip install faster-whisper"
                ) from exc

            self._model = WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
            logger.info("Whisper model loaded.")
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, *, output_path: str | Path | None = None) -> Path:
        """Record from the default microphone with automatic silence detection.

        Recording stops after :attr:`~Settings.audio_silence_duration` seconds
        of silence *or* after :attr:`~Settings.audio_max_duration` seconds
        total.

        Parameters
        ----------
        output_path:
            Where to save the WAV file.  If ``None`` a temporary file is used.

        Returns
        -------
        Path
            Path to the saved WAV file.
        """
        sample_rate = self._settings.audio_sample_rate
        silence_limit = self._settings.audio_silence_duration
        max_duration = self._settings.audio_max_duration
        frame_duration = 0.1  # seconds per analysis frame
        frame_samples = int(sample_rate * frame_duration)

        logger.info("Recording … speak now. Silence for %.1fs stops recording.", silence_limit)

        frames: list[np.ndarray] = []
        silent_frames = 0
        max_silent_frames = int(silence_limit / frame_duration)
        max_total_frames = int(max_duration / frame_duration)

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while len(frames) < max_total_frames:
                chunk, _ = stream.read(frame_samples)
                frames.append(chunk.copy())
                rms = float(np.sqrt(np.mean(chunk**2)))

                if rms < _ENERGY_THRESHOLD:
                    silent_frames += 1
                else:
                    silent_frames = 0  # reset on speech activity

                # Only stop after at least 0.5 s of audio has been captured
                min_frames = int(0.5 / frame_duration)
                if len(frames) > min_frames and silent_frames >= max_silent_frames:
                    break

        audio = np.concatenate(frames, axis=0)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
        else:
            output_path = Path(output_path)

        sf.write(str(output_path), audio, sample_rate)
        duration = len(audio) / sample_rate
        logger.info("Recorded %.1f seconds → %s", duration, output_path)
        return output_path

    def transcribe_file(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe a pre-recorded audio file.

        Parameters
        ----------
        audio_path:
            Path to a WAV, MP3, or any format supported by ffmpeg/soundfile.

        Returns
        -------
        TranscriptionResult
            Transcribed text with metadata.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.debug("Transcribing %s", audio_path)
        start = time.perf_counter()

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            language=None,  # auto-detect
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = list(segments_iter)
        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.perf_counter() - start

        logger.info(
            "Transcription complete in %.2fs — language=%s text=%r",
            elapsed,
            info.language,
            text[:80],
        )

        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=info.duration,
            segments=[{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        )

    def record_and_transcribe(self) -> TranscriptionResult:
        """Record microphone input, transcribe it, then clean up the temp file.

        Returns
        -------
        TranscriptionResult
            Transcription of the recorded speech.
        """
        audio_path = self.record()
        try:
            return self.transcribe_file(audio_path)
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
