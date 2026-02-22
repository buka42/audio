"""Transcription module with word-level timestamps for Polish audio."""

import logging
import os
from typing import List, Optional

import numpy as np

from .models import TranscriptionResult, WordSegment

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    model_size: str = "base",
    language: str = "pl",
) -> TranscriptionResult:
    """Transcribe audio using faster-whisper with word-level timestamps.

    Args:
        audio: Mono float32 audio at 16kHz.
        sample_rate: Sample rate (should be 16000 for Whisper).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Language code.

    Returns:
        TranscriptionResult with word-level timestamps.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )
        raise

    logger.info("Loading Whisper model '%s' for language '%s'...", model_size, language)

    # Use CPU with int8 for compatibility
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    logger.info("Transcribing audio (%.1f seconds)...", len(audio) / sample_rate)

    segments_gen, info = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=200,
            speech_pad_ms=100,
        ),
    )

    words: List[WordSegment] = []
    full_text_parts = []

    for segment in segments_gen:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words.append(WordSegment(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    confidence=w.probability if hasattr(w, "probability") else 1.0,
                ))

    full_text = " ".join(full_text_parts)
    logger.info(
        "Transcription complete: %d words, detected language: %s (prob: %.2f)",
        len(words),
        info.language,
        info.language_probability,
    )

    return TranscriptionResult(
        text=full_text,
        words=words,
        language=info.language,
    )


def transcribe_audio_stub(
    audio: np.ndarray,
    sample_rate: int,
    words_data: Optional[List[dict]] = None,
) -> TranscriptionResult:
    """Stub transcription for testing without Whisper model.

    Args:
        audio: Audio data (not used in stub).
        sample_rate: Sample rate.
        words_data: Optional list of word dicts with 'word', 'start', 'end', 'confidence'.

    Returns:
        TranscriptionResult from provided data.
    """
    if words_data is None:
        words_data = []

    words = [
        WordSegment(
            word=w["word"],
            start=w["start"],
            end=w["end"],
            confidence=w.get("confidence", 1.0),
        )
        for w in words_data
    ]

    full_text = " ".join(w.word for w in words)

    return TranscriptionResult(
        text=full_text,
        words=words,
        language="pl",
    )
