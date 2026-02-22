"""Audio renderer: applies edits with crossfade and soften modes."""

import logging
from typing import List, Optional

import numpy as np

from .models import Config, EditCandidate, EditStatus, SoftenMode
from .vad import extract_room_tone

logger = logging.getLogger(__name__)


def render_audio(
    audio: np.ndarray,
    sample_rate: int,
    edits: List[EditCandidate],
    config: Config,
    room_tone: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render edited audio by applying all edit candidates.

    Applies edits in reverse order (from end to start) to maintain
    correct time references. Each edit is masked according to soften_mode.

    Args:
        audio: Original audio samples (float32).
        sample_rate: Sample rate.
        edits: List of edit candidates to apply.
        config: Processing configuration.
        room_tone: Pre-extracted room tone sample (for ROOMTONE mode).

    Returns:
        Processed audio array.
    """
    # Only apply edits with APPLIED status
    applied_edits = [e for e in edits if e.status == EditStatus.APPLIED]

    if not applied_edits:
        logger.info("No edits to apply.")
        return audio.copy()

    # Sort by start time descending (apply from end to preserve indices)
    applied_edits.sort(key=lambda e: e.start, reverse=True)

    result = audio.copy()
    crossfade_samples = int(config.crossfade_ms * sample_rate / 1000)
    soften_samples = int(config.soften_ms * sample_rate / 1000)

    for edit in applied_edits:
        start_sample = int(edit.start * sample_rate)
        end_sample = int(edit.end * sample_rate)

        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(result), end_sample)

        if start_sample >= end_sample:
            continue

        if config.soften_mode == SoftenMode.ROOMTONE:
            result = _apply_roomtone_edit(
                result, start_sample, end_sample,
                sample_rate, crossfade_samples, soften_samples,
                room_tone,
            )
        elif config.soften_mode == SoftenMode.FADE:
            result = _apply_fade_edit(
                result, start_sample, end_sample,
                sample_rate, crossfade_samples, soften_samples,
            )
        else:  # CUT
            result = _apply_cut_edit(
                result, start_sample, end_sample,
                sample_rate, crossfade_samples,
            )

    logger.info(
        "Rendered %d edits. Original: %d samples, Result: %d samples (%.1fs removed)",
        len(applied_edits),
        len(audio),
        len(result),
        (len(audio) - len(result)) / sample_rate,
    )

    return result


def _apply_cut_edit(
    audio: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    crossfade_samples: int,
) -> np.ndarray:
    """Apply a hard cut with crossfade at the splice point.

    Removes the segment [start, end) and crossfades the junction.
    """
    cf = min(crossfade_samples, start, len(audio) - end)
    if cf <= 0:
        # No room for crossfade, just concatenate
        return np.concatenate([audio[:start], audio[end:]])

    # Get segments around the cut
    before = audio[:start + cf]
    after = audio[end - cf:]

    # Create crossfade
    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)

    # Overlap region
    overlap = before[-cf:] * fade_out + after[:cf] * fade_in

    # Assemble
    result = np.concatenate([
        audio[:start],
        overlap,
        after[cf:],
    ])

    return result


def _apply_fade_edit(
    audio: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    crossfade_samples: int,
    soften_samples: int,
) -> np.ndarray:
    """Apply edit with fade out/in around the cut point.

    Fades out before the cut, removes the segment, fades in after.
    """
    cf = min(crossfade_samples, start, len(audio) - end)
    fade_len = min(soften_samples, start, len(audio) - end)

    if fade_len <= 0 or cf <= 0:
        return _apply_cut_edit(audio, start, end, sample_rate, crossfade_samples)

    result = audio.copy()

    # Apply fade-out before the cut
    fade_out_start = max(0, start - fade_len)
    fade_out = np.linspace(1.0, 0.0, start - fade_out_start, dtype=np.float32)
    result[fade_out_start:start] *= fade_out

    # Apply fade-in after the cut
    fade_in_end = min(len(result), end + fade_len)
    fade_in = np.linspace(0.0, 1.0, fade_in_end - end, dtype=np.float32)
    result[end:fade_in_end] *= fade_in

    # Now do the cut with crossfade
    return _apply_cut_edit(result, start, end, sample_rate, cf)


def _apply_roomtone_edit(
    audio: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    crossfade_samples: int,
    soften_samples: int,
    room_tone: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply edit by replacing removed segment with room tone.

    Instead of a hard cut, replaces the removed content with a short
    room tone segment, making the edit less audible.
    """
    # Duration of room tone to insert (short, just for smoothing)
    insert_duration = min(soften_samples, end - start)
    cf = min(crossfade_samples, start, len(audio) - end, insert_duration // 2)

    if cf <= 0:
        return _apply_cut_edit(audio, start, end, sample_rate, crossfade_samples)

    # Prepare room tone fill
    if room_tone is not None and len(room_tone) > 0:
        # Use actual room tone
        if len(room_tone) >= insert_duration:
            fill = room_tone[:insert_duration].copy()
        else:
            repeats = (insert_duration // len(room_tone)) + 1
            fill = np.tile(room_tone, repeats)[:insert_duration].copy()
    else:
        # Fallback: use very quiet noise matching surrounding level
        nearby_start = max(0, start - sample_rate)
        nearby_end = min(len(audio), end + sample_rate)
        nearby = audio[nearby_start:nearby_end]
        noise_level = np.percentile(np.abs(nearby), 10) * 0.3
        fill = np.random.randn(insert_duration).astype(np.float32) * noise_level

    # Crossfade into room tone
    if cf > 0 and cf <= len(fill):
        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)

        # Blend start of fill with end of pre-edit audio
        pre_cf = audio[start:start + cf]
        fill[:cf] = pre_cf * fade_out + fill[:cf] * fade_in

        # Blend end of fill with start of post-edit audio
        post_cf = audio[end:end + cf] if end + cf <= len(audio) else audio[end:]
        actual_cf = min(cf, len(post_cf), len(fill))
        if actual_cf > 0:
            fade_out2 = np.linspace(1.0, 0.0, actual_cf, dtype=np.float32)
            fade_in2 = np.linspace(0.0, 1.0, actual_cf, dtype=np.float32)
            fill[-actual_cf:] = fill[-actual_cf:] * fade_out2 + post_cf[:actual_cf] * fade_in2

    # Assemble: before + room_tone_fill + after
    result = np.concatenate([
        audio[:start],
        fill,
        audio[end:],
    ])

    return result


def apply_offset_preservation(
    original_audio: np.ndarray,
    processed_segment: np.ndarray,
    sample_rate: int,
    start_offset_sec: Optional[float],
    end_offset_sec: Optional[float],
    original_sr: int,
) -> np.ndarray:
    """Combine processed segment with unprocessed parts if offsets are set.

    If the user specified start/end offsets, only the segment within those
    bounds was processed. This function reassembles the full audio.

    Args:
        original_audio: Full original audio at original SR.
        processed_segment: Processed audio segment at original SR.
        sample_rate: Sample rate of both arrays.
        start_offset_sec: Start of processed region.
        end_offset_sec: End of processed region.
        original_sr: Original sample rate.

    Returns:
        Full audio with processed segment inserted.
    """
    if start_offset_sec is None and end_offset_sec is None:
        return processed_segment

    parts = []

    if start_offset_sec is not None and start_offset_sec > 0:
        start_sample = int(start_offset_sec * sample_rate)
        parts.append(original_audio[:start_sample])

    parts.append(processed_segment)

    if end_offset_sec is not None:
        end_sample = int(end_offset_sec * sample_rate)
        if end_sample < len(original_audio):
            parts.append(original_audio[end_sample:])

    return np.concatenate(parts)
