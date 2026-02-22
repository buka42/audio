"""Main processing pipeline for Audio Assembly."""

import logging
import os
import time
from typing import List, Optional

import numpy as np

from .audio_io import (
    export_audio,
    load_audio,
    load_audio_original,
    seconds_to_samples,
)
from .fillers import detect_fillers, detect_long_pauses_edits
from .models import (
    Config,
    EditCandidate,
    ProcessingMode,
    ProcessingReport,
    TranscriptionResult,
)
from .renderer import apply_offset_preservation, render_audio
from .repetitions import detect_repetitions
from .report import (
    format_report_summary,
    generate_report,
    save_edl_json,
    save_report,
    save_transcript,
)
from .transcription import transcribe_audio, transcribe_audio_stub
from .vad import (
    detect_silences,
    detect_voice_activity,
    extract_room_tone,
)

logger = logging.getLogger(__name__)


def process_audio(
    input_file: str,
    output_dir: str,
    config: Config,
    stub_transcription: Optional[List[dict]] = None,
) -> ProcessingReport:
    """Run the full audio processing pipeline.

    Pipeline steps:
    1. Load audio
    2. VAD + silence detection
    3. Transcription (PL, word-level timestamps)
    4. Build edit candidates
    5. Apply edits per selected mode
    6. Render audio with crossfade/soften
    7. Save output files + report

    Args:
        input_file: Path to input audio file.
        output_dir: Directory for output files.
        config: Processing configuration.
        stub_transcription: Optional stub data for testing without Whisper.

    Returns:
        ProcessingReport with all results.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(input_file))[0]
    mode_suffix = config.mode.value

    # Output file paths
    output_audio = os.path.join(output_dir, f"{basename}_mode{mode_suffix}.wav")
    output_edl = os.path.join(output_dir, f"{basename}_mode{mode_suffix}_edl.json")
    output_transcript = os.path.join(output_dir, f"{basename}_mode{mode_suffix}_transcript.txt")
    output_report = os.path.join(output_dir, f"{basename}_mode{mode_suffix}_report.json")

    # ---- Step 1: Load audio ----
    logger.info("Step 1: Loading audio...")
    audio_analysis, analysis_sr, metadata = load_audio(
        input_file,
        target_sr=config.analysis_sample_rate,
        start_sec=config.start_offset_sec,
        end_sec=config.end_offset_sec,
    )

    original_duration = len(audio_analysis) / analysis_sr
    logger.info("Loaded %.1fs of audio for analysis at %dHz", original_duration, analysis_sr)

    # Also load at export quality
    audio_export, export_sr, _ = load_audio(
        input_file,
        target_sr=config.export_sample_rate,
        start_sec=config.start_offset_sec,
        end_sec=config.end_offset_sec,
    )

    # ---- Step 2: VAD + silence detection ----
    logger.info("Step 2: Voice activity detection...")
    voice_segments = detect_voice_activity(
        audio_analysis,
        analysis_sr,
        min_silence_duration_ms=config.min_gap_to_edit_ms,
    )
    silences = detect_silences(voice_segments, min_silence_sec=config.min_gap_to_edit_ms / 1000)

    # Extract room tone for ROOMTONE soften mode
    room_tone_analysis = extract_room_tone(audio_analysis, analysis_sr, voice_segments)
    room_tone_export = extract_room_tone(audio_export, export_sr, voice_segments)

    # ---- Step 3: Transcription ----
    logger.info("Step 3: Transcription...")
    if stub_transcription is not None:
        transcription = transcribe_audio_stub(audio_analysis, analysis_sr, stub_transcription)
    else:
        transcription = transcribe_audio(
            audio_analysis,
            analysis_sr,
            model_size=config.whisper_model,
            language="pl",
        )

    logger.info("Transcribed %d words", len(transcription.words))

    # ---- Step 4: Build edit candidates ----
    logger.info("Step 4: Building edit candidates...")
    edits: List[EditCandidate] = []

    if config.mode == ProcessingMode.CLEAN_FILLERS_AND_PAUSES:
        # Mode A: fillers + pauses
        filler_edits = detect_fillers(transcription, config)
        pause_edits = detect_long_pauses_edits(silences, config)
        edits.extend(filler_edits)
        edits.extend(pause_edits)

    elif config.mode == ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL:
        # Mode B: repetitions + restarts
        repetition_edits = detect_repetitions(transcription, config)
        edits.extend(repetition_edits)

    # Sort by time
    edits.sort(key=lambda e: e.start)

    logger.info(
        "Built %d edit candidates (%d applied, %d review, %d skipped)",
        len(edits),
        sum(1 for e in edits if e.status.value == "applied"),
        sum(1 for e in edits if e.status.value == "review"),
        sum(1 for e in edits if e.status.value == "skipped"),
    )

    # ---- Step 5 & 6: Apply edits and render ----
    logger.info("Step 5-6: Rendering audio...")
    processed_audio = render_audio(
        audio_export,
        export_sr,
        edits,
        config,
        room_tone=room_tone_export,
    )

    # If offsets were used, reassemble with original unprocessed parts
    if config.start_offset_sec is not None or config.end_offset_sec is not None:
        full_original, full_sr, _ = load_audio(
            input_file, target_sr=config.export_sample_rate
        )
        processed_audio = apply_offset_preservation(
            full_original,
            processed_audio,
            export_sr,
            config.start_offset_sec,
            config.end_offset_sec,
            full_sr,
        )

    processed_duration = len(processed_audio) / export_sr

    # ---- Step 7: Save outputs ----
    logger.info("Step 7: Saving outputs...")

    export_audio(processed_audio, export_sr, output_audio)

    save_edl_json(
        edits,
        output_edl,
        metadata={
            "input_file": input_file,
            "mode": config.mode.name,
            "config": {
                "silence_threshold_sec": config.silence_threshold_sec,
                "min_gap_to_edit_ms": config.min_gap_to_edit_ms,
                "crossfade_ms": config.crossfade_ms,
                "soften_mode": config.soften_mode.value,
                "soften_ms": config.soften_ms,
                "max_pause_after_edit_sec": config.max_pause_after_edit_sec,
            },
        },
    )

    save_transcript(transcription, edits, output_transcript)

    # Generate report
    report = generate_report(
        input_file=input_file,
        output_file=output_audio,
        edl_file=output_edl,
        transcript_file=output_transcript,
        mode=config.mode.name,
        original_duration=original_duration,
        processed_duration=processed_duration,
        edits=edits,
    )

    save_report(report, output_report)

    elapsed = time.time() - start_time
    logger.info("Processing complete in %.1fs", elapsed)

    return report
