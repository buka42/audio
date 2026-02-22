"""Audio I/O: loading, normalization, and export."""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def load_audio(
    filepath: str,
    target_sr: int = 16000,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[np.ndarray, int, dict]:
    """Load an audio file and return mono float32 samples at target_sr.

    Returns:
        (samples, sample_rate, metadata) where metadata includes original info.
    """
    info = sf.info(filepath)
    metadata = {
        "original_sr": info.samplerate,
        "original_channels": info.channels,
        "original_subtype": info.subtype,
        "original_format": info.format,
        "duration_sec": info.duration,
        "frames": info.frames,
    }
    logger.info(
        "Loading %s: %.1fs, %dHz, %d ch, %s",
        filepath,
        info.duration,
        info.samplerate,
        info.channels,
        info.subtype,
    )

    start_frame = None
    frames_to_read = None
    if start_sec is not None:
        start_frame = int(start_sec * info.samplerate)
    if end_sec is not None:
        end_frame = int(end_sec * info.samplerate)
        if start_frame is not None:
            frames_to_read = end_frame - start_frame
        else:
            frames_to_read = end_frame

    data, sr = sf.read(
        filepath,
        start=start_frame or 0,
        stop=(start_frame or 0) + frames_to_read if frames_to_read else None,
        dtype="float32",
    )

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        data = _resample(data, sr, target_sr)

    return data, target_sr, metadata


def load_audio_original(
    filepath: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[np.ndarray, int, dict]:
    """Load audio at original sample rate for export quality."""
    info = sf.info(filepath)
    metadata = {
        "original_sr": info.samplerate,
        "original_channels": info.channels,
        "original_subtype": info.subtype,
        "original_format": info.format,
        "duration_sec": info.duration,
    }

    start_frame = None
    frames_to_read = None
    if start_sec is not None:
        start_frame = int(start_sec * info.samplerate)
    if end_sec is not None:
        end_frame = int(end_sec * info.samplerate)
        if start_frame is not None:
            frames_to_read = end_frame - start_frame
        else:
            frames_to_read = end_frame

    data, sr = sf.read(
        filepath,
        start=start_frame or 0,
        stop=(start_frame or 0) + frames_to_read if frames_to_read else None,
        dtype="float32",
    )

    return data, sr, metadata


def export_audio(
    data: np.ndarray,
    sample_rate: int,
    filepath: str,
    subtype: Optional[str] = None,
) -> str:
    """Export audio data to file.

    Args:
        data: Audio samples as float32 numpy array.
        sample_rate: Sample rate.
        filepath: Output file path.
        subtype: Audio subtype (e.g., 'PCM_24' for 24-bit WAV).

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    ext = os.path.splitext(filepath)[1].lower()
    if subtype is None:
        if ext == ".wav":
            subtype = "PCM_24"
        elif ext == ".flac":
            subtype = "PCM_24"

    # Clip to prevent clipping distortion
    data = np.clip(data, -1.0, 1.0)

    sf.write(filepath, data, sample_rate, subtype=subtype)
    logger.info("Exported audio to %s (%d samples, %d Hz)", filepath, len(data), sample_rate)
    return os.path.abspath(filepath)


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy."""
    from scipy.signal import resample

    if orig_sr == target_sr:
        return data
    num_samples = int(len(data) * target_sr / orig_sr)
    resampled = resample(data, num_samples)
    return resampled.astype(np.float32)


def get_duration(filepath: str) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(filepath)
    return info.duration


def samples_to_seconds(n_samples: int, sample_rate: int) -> float:
    """Convert sample count to seconds."""
    return n_samples / sample_rate


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to sample count."""
    return int(seconds * sample_rate)
