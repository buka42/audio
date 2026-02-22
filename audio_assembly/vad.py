"""Voice Activity Detection and silence/pause detection."""

import logging
from typing import List, Tuple

import numpy as np

from .models import SilenceSegment, VoiceSegment

logger = logging.getLogger(__name__)


def detect_voice_activity(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 30,
    energy_threshold: float = 0.01,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 50,
) -> List[VoiceSegment]:
    """Detect voice activity using energy-based VAD.

    Uses RMS energy with adaptive thresholding for reliable detection.

    Args:
        audio: Mono float32 audio samples.
        sample_rate: Sample rate of audio.
        frame_duration_ms: Frame size in ms for analysis.
        energy_threshold: Base energy threshold for speech detection.
        min_speech_duration_ms: Minimum speech segment duration.
        min_silence_duration_ms: Minimum silence segment duration.

    Returns:
        List of VoiceSegment with speech/silence classification.
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    n_frames = len(audio) // frame_size

    if n_frames == 0:
        return [VoiceSegment(start=0.0, end=len(audio) / sample_rate, is_speech=True)]

    # Calculate RMS energy per frame
    frame_energies = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        frame_energies[i] = np.sqrt(np.mean(frame**2))

    # Adaptive threshold: use a percentile of energy as noise floor
    noise_floor = np.percentile(frame_energies[frame_energies > 0], 10) if np.any(frame_energies > 0) else 0.001
    adaptive_threshold = max(energy_threshold, noise_floor * 3.0)

    # Classify frames
    is_speech = frame_energies > adaptive_threshold

    # Smooth: apply minimum duration constraints
    is_speech = _smooth_vad(
        is_speech,
        frame_duration_ms,
        min_speech_duration_ms,
        min_silence_duration_ms,
    )

    # Convert to segments
    segments = _frames_to_segments(is_speech, frame_duration_ms, sample_rate, len(audio))

    logger.info(
        "VAD: %d segments (%d speech, %d silence), threshold=%.4f",
        len(segments),
        sum(1 for s in segments if s.is_speech),
        sum(1 for s in segments if not s.is_speech),
        adaptive_threshold,
    )

    return segments


def detect_silences(
    voice_segments: List[VoiceSegment],
    min_silence_sec: float = 0.1,
) -> List[SilenceSegment]:
    """Extract silence segments from VAD results.

    Args:
        voice_segments: VAD segments.
        min_silence_sec: Minimum silence duration to report.

    Returns:
        List of SilenceSegment.
    """
    silences = []
    for seg in voice_segments:
        if not seg.is_speech:
            duration = seg.end - seg.start
            if duration >= min_silence_sec:
                silences.append(SilenceSegment(start=seg.start, end=seg.end))

    logger.info("Found %d silence segments >= %.1fs", len(silences), min_silence_sec)
    return silences


def detect_long_pauses(
    silences: List[SilenceSegment],
    threshold_sec: float = 3.0,
) -> List[SilenceSegment]:
    """Find pauses longer than threshold.

    Args:
        silences: All silence segments.
        threshold_sec: Minimum duration to consider a "long pause".

    Returns:
        List of long pause segments.
    """
    long_pauses = [s for s in silences if s.duration >= threshold_sec]
    logger.info(
        "Found %d long pauses (>= %.1fs), total %.1fs",
        len(long_pauses),
        threshold_sec,
        sum(p.duration for p in long_pauses),
    )
    return long_pauses


def extract_room_tone(
    audio: np.ndarray,
    sample_rate: int,
    voice_segments: List[VoiceSegment],
    duration_ms: int = 500,
) -> np.ndarray:
    """Extract a room tone sample from a quiet segment.

    Finds the quietest non-speech segment and extracts a sample.

    Args:
        audio: Full audio array.
        sample_rate: Sample rate.
        voice_segments: VAD segments.
        duration_ms: Desired room tone duration in ms.

    Returns:
        Room tone audio sample.
    """
    target_samples = int(sample_rate * duration_ms / 1000)

    # Find silence segments long enough
    silence_segments = []
    for seg in voice_segments:
        if not seg.is_speech:
            seg_samples = int((seg.end - seg.start) * sample_rate)
            if seg_samples >= target_samples:
                silence_segments.append(seg)

    if not silence_segments:
        # Fallback: generate very quiet noise
        logger.warning("No suitable silence segment for room tone, generating synthetic.")
        noise_level = np.percentile(np.abs(audio), 5) * 0.5
        return np.random.randn(target_samples).astype(np.float32) * noise_level

    # Pick the quietest one
    best_seg = None
    best_energy = float("inf")
    for seg in silence_segments:
        start_idx = int(seg.start * sample_rate)
        end_idx = min(start_idx + target_samples, len(audio))
        if end_idx - start_idx < target_samples:
            continue
        chunk = audio[start_idx:end_idx]
        energy = np.sqrt(np.mean(chunk**2))
        if energy < best_energy:
            best_energy = energy
            best_seg = seg

    if best_seg is None:
        best_seg = silence_segments[0]

    start_idx = int(best_seg.start * sample_rate)
    room_tone = audio[start_idx : start_idx + target_samples].copy()

    # Ensure it's the right length
    if len(room_tone) < target_samples:
        # Tile to fill
        repeats = (target_samples // len(room_tone)) + 1
        room_tone = np.tile(room_tone, repeats)[:target_samples]

    logger.info("Extracted room tone: %.3fs from %.2fs", duration_ms / 1000, best_seg.start)
    return room_tone


def _smooth_vad(
    is_speech: np.ndarray,
    frame_ms: int,
    min_speech_ms: int,
    min_silence_ms: int,
) -> np.ndarray:
    """Smooth VAD decisions by enforcing minimum durations."""
    min_speech_frames = max(1, min_speech_ms // frame_ms)
    min_silence_frames = max(1, min_silence_ms // frame_ms)

    result = is_speech.copy()

    # Remove short speech segments
    in_segment = False
    seg_start = 0
    for i in range(len(result)):
        if result[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not result[i] and in_segment:
            if i - seg_start < min_speech_frames:
                result[seg_start:i] = False
            in_segment = False
    if in_segment and len(result) - seg_start < min_speech_frames:
        result[seg_start:] = False

    # Remove short silence segments
    in_segment = False
    seg_start = 0
    for i in range(len(result)):
        if not result[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif result[i] and in_segment:
            if i - seg_start < min_silence_frames:
                result[seg_start:i] = True
            in_segment = False

    return result


def _frames_to_segments(
    is_speech: np.ndarray,
    frame_ms: int,
    sample_rate: int,
    total_samples: int,
) -> List[VoiceSegment]:
    """Convert frame-level decisions to time-based segments."""
    segments = []
    if len(is_speech) == 0:
        return segments

    current_speech = is_speech[0]
    seg_start = 0.0
    frame_sec = frame_ms / 1000.0
    total_duration = total_samples / sample_rate

    for i in range(1, len(is_speech)):
        if is_speech[i] != current_speech:
            seg_end = i * frame_sec
            segments.append(VoiceSegment(
                start=seg_start,
                end=seg_end,
                is_speech=bool(current_speech),
            ))
            seg_start = seg_end
            current_speech = is_speech[i]

    # Final segment
    segments.append(VoiceSegment(
        start=seg_start,
        end=total_duration,
        is_speech=bool(current_speech),
    ))

    return segments
