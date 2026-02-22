"""Demo tests with synthetic audio files.

Creates two synthetic Polish audio recordings and processes them
through both Mode A and Mode B to demonstrate the pipeline.
"""

import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_assembly.models import Config, ProcessingMode, SoftenMode
from audio_assembly.pipeline import process_audio
from audio_assembly.report import format_report_summary


def generate_tone(freq: float, duration: float, sr: int = 48000, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_silence(duration: float, sr: int = 48000, noise_level: float = 0.001) -> np.ndarray:
    """Generate silence with very low background noise (room tone)."""
    n_samples = int(sr * duration)
    return (np.random.randn(n_samples) * noise_level).astype(np.float32)


def generate_speech_like(duration: float, sr: int = 48000, amplitude: float = 0.3) -> np.ndarray:
    """Generate speech-like audio (multi-frequency noise burst)."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)

    # Combine multiple frequencies to simulate speech formants
    signal = np.zeros(n_samples, dtype=np.float32)
    for freq in [150, 300, 600, 1200, 2400]:
        signal += np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)

    # Add some noise for naturalness
    signal += np.random.randn(n_samples).astype(np.float32) * 0.1

    # Normalize and apply amplitude
    signal = signal / np.max(np.abs(signal)) * amplitude

    # Apply envelope (smooth attack/release)
    env_samples = min(int(0.01 * sr), n_samples // 4)
    if env_samples > 0:
        signal[:env_samples] *= np.linspace(0, 1, env_samples)
        signal[-env_samples:] *= np.linspace(1, 0, env_samples)

    return signal


def create_demo_file_1(filepath: str, sr: int = 48000):
    """Create Demo File 1: Interview with fillers and long pauses.

    Simulates a Polish interview with:
    - Speech segments
    - Filler sounds (yyy, eee)
    - Long pauses (>3s)
    - Short natural pauses

    Timeline (approx):
    0.0-1.5s   speech ("Dzień dobry")
    1.5-1.8s   short pause
    1.8-2.5s   filler ("yyy")
    2.5-2.7s   short pause
    2.7-4.5s   speech ("chciałem powiedzieć")
    4.5-8.0s   LONG PAUSE (3.5s)
    8.0-10.0s  speech ("że to jest ważne")
    10.0-10.3s short pause
    10.3-11.0s filler ("eee")
    11.0-11.2s short pause
    11.2-13.0s speech ("bardzo dziękuję")
    """
    parts = []

    # Speech 1
    parts.append(generate_speech_like(1.5, sr, 0.35))
    # Short pause
    parts.append(generate_silence(0.3, sr))
    # Filler "yyy" - lower amplitude, single frequency
    parts.append(generate_tone(180, 0.7, sr, 0.15))
    # Short pause
    parts.append(generate_silence(0.2, sr))
    # Speech 2
    parts.append(generate_speech_like(1.8, sr, 0.30))
    # LONG PAUSE
    parts.append(generate_silence(3.5, sr))
    # Speech 3
    parts.append(generate_speech_like(2.0, sr, 0.35))
    # Short pause
    parts.append(generate_silence(0.3, sr))
    # Filler "eee"
    parts.append(generate_tone(220, 0.7, sr, 0.12))
    # Short pause
    parts.append(generate_silence(0.2, sr))
    # Speech 4
    parts.append(generate_speech_like(1.8, sr, 0.32))

    audio = np.concatenate(parts)
    sf.write(filepath, audio, sr, subtype='PCM_24')

    # Return word-level timestamps for stub transcription
    words = [
        {"word": "Dzień", "start": 0.0, "end": 0.7, "confidence": 0.95},
        {"word": "dobry", "start": 0.7, "end": 1.5, "confidence": 0.95},
        {"word": "yyy", "start": 1.8, "end": 2.5, "confidence": 0.90},
        {"word": "chciałem", "start": 2.7, "end": 3.5, "confidence": 0.92},
        {"word": "powiedzieć", "start": 3.5, "end": 4.5, "confidence": 0.90},
        {"word": "że", "start": 8.0, "end": 8.3, "confidence": 0.95},
        {"word": "to", "start": 8.3, "end": 8.6, "confidence": 0.96},
        {"word": "jest", "start": 8.6, "end": 9.0, "confidence": 0.95},
        {"word": "ważne", "start": 9.0, "end": 10.0, "confidence": 0.94},
        {"word": "eee", "start": 10.3, "end": 11.0, "confidence": 0.88},
        {"word": "bardzo", "start": 11.2, "end": 12.0, "confidence": 0.93},
        {"word": "dziękuję", "start": 12.0, "end": 13.0, "confidence": 0.95},
    ]

    return words


def create_demo_file_2(filepath: str, sr: int = 48000):
    """Create Demo File 2: Interview with repetitions and restarts.

    Simulates a Polish interview with:
    - Immediate word repetitions ("to to")
    - Phrase repetitions
    - Sentence restarts

    Timeline (approx):
    0.0-1.0s   speech ("Myślę")
    1.0-1.3s   short pause
    1.3-1.7s   speech ("to") - first occurrence
    1.7-2.1s   speech ("to") - repetition
    2.1-3.5s   speech ("jest ważne")
    3.5-3.8s   short pause
    3.8-5.0s   speech ("Chciałem") - incomplete
    5.0-5.3s   short pause
    5.3-5.8s   speech ("znaczy") - restart marker
    5.8-8.0s   speech ("chodzi mi o to że") - restarted sentence
    8.0-8.3s   pause
    8.3-10.0s  speech ("to jest dobre")
    """
    parts = []

    # Speech 1 "Myślę"
    parts.append(generate_speech_like(1.0, sr, 0.33))
    # Short pause
    parts.append(generate_silence(0.3, sr))
    # "to" (first)
    parts.append(generate_speech_like(0.4, sr, 0.30))
    # "to" (repetition)
    parts.append(generate_speech_like(0.4, sr, 0.30))
    # "jest ważne"
    parts.append(generate_speech_like(1.4, sr, 0.35))
    # Short pause
    parts.append(generate_silence(0.3, sr))
    # "Chciałem" (incomplete sentence)
    parts.append(generate_speech_like(1.2, sr, 0.28))
    # Short pause
    parts.append(generate_silence(0.3, sr))
    # "znaczy" (restart marker)
    parts.append(generate_speech_like(0.5, sr, 0.25))
    # "chodzi mi o to że" (restarted)
    parts.append(generate_speech_like(2.2, sr, 0.35))
    # Pause
    parts.append(generate_silence(0.3, sr))
    # "to jest dobre"
    parts.append(generate_speech_like(1.7, sr, 0.33))

    audio = np.concatenate(parts)
    sf.write(filepath, audio, sr, subtype='PCM_24')

    # Word-level timestamps for stub transcription
    words = [
        {"word": "Myślę", "start": 0.0, "end": 1.0, "confidence": 0.93},
        {"word": "to", "start": 1.3, "end": 1.7, "confidence": 0.95},
        {"word": "to", "start": 1.7, "end": 2.1, "confidence": 0.95},
        {"word": "jest", "start": 2.1, "end": 2.7, "confidence": 0.94},
        {"word": "ważne", "start": 2.7, "end": 3.5, "confidence": 0.92},
        {"word": "Chciałem", "start": 3.8, "end": 5.0, "confidence": 0.90},
        {"word": "znaczy", "start": 5.3, "end": 5.8, "confidence": 0.91},
        {"word": "chodzi", "start": 5.8, "end": 6.3, "confidence": 0.93},
        {"word": "mi", "start": 6.3, "end": 6.5, "confidence": 0.95},
        {"word": "o", "start": 6.5, "end": 6.7, "confidence": 0.96},
        {"word": "to", "start": 6.7, "end": 6.9, "confidence": 0.95},
        {"word": "że", "start": 6.9, "end": 7.2, "confidence": 0.94},
        {"word": "to", "start": 8.3, "end": 8.7, "confidence": 0.95},
        {"word": "jest", "start": 8.7, "end": 9.2, "confidence": 0.94},
        {"word": "dobre", "start": 9.2, "end": 10.0, "confidence": 0.93},
    ]

    return words


def run_demo():
    """Run the full demo with both files and both modes."""
    output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_base, exist_ok=True)

    print("=" * 60)
    print("  AUDIO ASSEMBLY - Demo z plikami syntetycznymi")
    print("=" * 60)
    print()

    # ---- Demo 1: Mode A (fillers + pauses) ----
    print("▶ Demo 1: Tryb A - Czyszczenie wypełniaczy i pauz")
    print("-" * 50)

    demo1_path = os.path.join(output_base, "demo1_fillers_pauses.wav")
    demo1_words = create_demo_file_1(demo1_path)
    print(f"  Utworzono plik demo: {demo1_path}")

    config_a = Config(
        mode=ProcessingMode.CLEAN_FILLERS_AND_PAUSES,
        silence_threshold_sec=3.0,
        crossfade_ms=60,
        soften_mode=SoftenMode.ROOMTONE,
        soften_ms=120,
        max_pause_after_edit_sec=0.4,
    )

    report_a = process_audio(
        demo1_path,
        os.path.join(output_base, "demo1_results"),
        config_a,
        stub_transcription=demo1_words,
    )

    summary_a = format_report_summary(report_a)
    print(summary_a)
    print()

    # ---- Demo 2: Mode B (repetitions) ----
    print("▶ Demo 2: Tryb B - Usuwanie powtórzeń")
    print("-" * 50)

    demo2_path = os.path.join(output_base, "demo2_repetitions.wav")
    demo2_words = create_demo_file_2(demo2_path)
    print(f"  Utworzono plik demo: {demo2_path}")

    config_b = Config(
        mode=ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL,
        crossfade_ms=60,
        soften_mode=SoftenMode.ROOMTONE,
        soften_ms=120,
        confidence_threshold=0.75,
    )

    report_b = process_audio(
        demo2_path,
        os.path.join(output_base, "demo2_results"),
        config_b,
        stub_transcription=demo2_words,
    )

    summary_b = format_report_summary(report_b)
    print(summary_b)
    print()

    # ---- Demo 3: Mode A with FADE soften ----
    print("▶ Demo 3: Tryb A z soften_mode=FADE")
    print("-" * 50)

    config_a_fade = Config(
        mode=ProcessingMode.CLEAN_FILLERS_AND_PAUSES,
        silence_threshold_sec=3.0,
        crossfade_ms=80,
        soften_mode=SoftenMode.FADE,
        soften_ms=150,
        max_pause_after_edit_sec=0.4,
    )

    report_a_fade = process_audio(
        demo1_path,
        os.path.join(output_base, "demo1_results_fade"),
        config_a_fade,
        stub_transcription=demo1_words,
    )

    summary_a_fade = format_report_summary(report_a_fade)
    print(summary_a_fade)
    print()

    # ---- Summary ----
    print("=" * 60)
    print("  PODSUMOWANIE")
    print("=" * 60)
    print(f"  Demo 1 (Tryb A): usunięto {report_a.total_removed_sec:.1f}s "
          f"({report_a.fillers_removed} wypełniaczy, {report_a.pauses_shortened} pauz)")
    print(f"  Demo 2 (Tryb B): usunięto {report_b.total_removed_sec:.1f}s "
          f"({report_b.repetitions_removed} powtórzeń)")
    print(f"  Demo 3 (Tryb A/FADE): usunięto {report_a_fade.total_removed_sec:.1f}s")
    print()
    print(f"  Pliki wyjściowe w: {output_base}/")
    print()

    # Print EDL sample
    edl_path = os.path.join(output_base, "demo1_results", "demo1_fillers_pauses_modeA_edl.json")
    if os.path.exists(edl_path):
        print("▶ Przykładowy EDL/JSON:")
        print("-" * 50)
        with open(edl_path, "r") as f:
            edl_data = json.load(f)
        print(json.dumps(edl_data, indent=2, ensure_ascii=False)[:2000])
        print()


if __name__ == "__main__":
    run_demo()
