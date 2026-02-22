"""Report and EDL/JSON generation."""

import json
import logging
import os
from typing import List, Optional

from .models import (
    EditCandidate,
    EditStatus,
    EditType,
    ProcessingReport,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


def generate_report(
    input_file: str,
    output_file: str,
    edl_file: str,
    transcript_file: str,
    mode: str,
    original_duration: float,
    processed_duration: float,
    edits: List[EditCandidate],
) -> ProcessingReport:
    """Generate a processing report from edit candidates.

    Args:
        input_file: Input audio file path.
        output_file: Output audio file path.
        edl_file: EDL/JSON file path.
        transcript_file: Transcript file path.
        mode: Processing mode name.
        original_duration: Original audio duration in seconds.
        processed_duration: Processed audio duration in seconds.
        edits: All edit candidates (applied and not).

    Returns:
        ProcessingReport with all statistics.
    """
    applied = [e for e in edits if e.status == EditStatus.APPLIED]

    fillers = [e for e in edits if e.edit_type == EditType.FILLER_REMOVAL]
    fillers_applied = [e for e in applied if e.edit_type == EditType.FILLER_REMOVAL]

    pauses = [e for e in applied if e.edit_type == EditType.PAUSE_SHORTENING]

    repetitions = [
        e for e in edits
        if e.edit_type in (EditType.REPETITION_REMOVAL, EditType.SENTENCE_RESTART_REMOVAL)
    ]
    repetitions_applied = [
        e for e in applied
        if e.edit_type in (EditType.REPETITION_REMOVAL, EditType.SENTENCE_RESTART_REMOVAL)
    ]

    total_removed = sum(e.duration for e in applied)

    report = ProcessingReport(
        input_file=input_file,
        output_file=output_file,
        edl_file=edl_file,
        transcript_file=transcript_file,
        mode=mode,
        original_duration_sec=original_duration,
        processed_duration_sec=processed_duration,
        total_removed_sec=total_removed,
        fillers_detected=len(fillers),
        fillers_removed=len(fillers_applied),
        pauses_shortened=len(pauses),
        repetitions_detected=len(repetitions),
        repetitions_removed=len(repetitions_applied),
        edits=edits,
    )

    return report


def save_edl_json(
    edits: List[EditCandidate],
    filepath: str,
    metadata: Optional[dict] = None,
) -> str:
    """Save edit decision list as JSON.

    Args:
        edits: All edit candidates.
        filepath: Output file path.
        metadata: Optional metadata to include.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    edl = {
        "version": "1.0",
        "format": "audio_assembly_edl",
        "metadata": metadata or {},
        "edits": [
            {
                "type": e.edit_type.value,
                "start_sec": round(e.start, 3),
                "end_sec": round(e.end, 3),
                "duration_sec": round(e.duration, 3),
                "confidence": round(e.confidence, 3),
                "status": e.status.value,
                "description": e.description,
                "original_text": e.original_text,
                "replacement_text": e.replacement_text,
            }
            for e in edits
        ],
        "summary": {
            "total_edits": len(edits),
            "applied": sum(1 for e in edits if e.status == EditStatus.APPLIED),
            "review": sum(1 for e in edits if e.status == EditStatus.REVIEW),
            "skipped": sum(1 for e in edits if e.status == EditStatus.SKIPPED),
            "total_duration_removed_sec": round(
                sum(e.duration for e in edits if e.status == EditStatus.APPLIED), 3
            ),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(edl, f, ensure_ascii=False, indent=2)

    logger.info("Saved EDL to %s", filepath)
    return os.path.abspath(filepath)


def save_transcript(
    transcription: TranscriptionResult,
    edits: List[EditCandidate],
    filepath: str,
) -> str:
    """Save transcription with edit markers.

    Words that were edited are marked with [REMOVED] tags.

    Args:
        transcription: Full transcription.
        edits: Applied edits.
        filepath: Output file path.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    applied = [e for e in edits if e.status == EditStatus.APPLIED]
    review = [e for e in edits if e.status == EditStatus.REVIEW]

    lines = []
    lines.append("=== TRANSKRYPCJA Z OZNACZENIAMI EDYCJI ===\n")
    lines.append(f"Język: {transcription.language}\n")
    lines.append("")

    for word in transcription.words:
        # Check if this word falls within any edit
        in_applied = any(
            e.start <= word.start and word.end <= e.end
            for e in applied
        )
        in_review = any(
            e.start <= word.start and word.end <= e.end
            for e in review
        )

        timestamp = f"[{_format_time(word.start)}-{_format_time(word.end)}]"

        if in_applied:
            lines.append(f"{timestamp} ~~{word.word}~~ [USUNIĘTO]")
        elif in_review:
            lines.append(f"{timestamp} {word.word} [DO PRZEGLĄDU]")
        else:
            lines.append(f"{timestamp} {word.word}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved transcript to %s", filepath)
    return os.path.abspath(filepath)


def save_report(report: ProcessingReport, filepath: str) -> str:
    """Save processing report as JSON.

    Args:
        report: Processing report.
        filepath: Output file path.

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info("Saved report to %s", filepath)
    return os.path.abspath(filepath)


def format_report_summary(report: ProcessingReport) -> str:
    """Format a human-readable summary of the processing report."""
    lines = [
        "╔══════════════════════════════════════════╗",
        "║      RAPORT MONTAŻU AUDIO                ║",
        "╚══════════════════════════════════════════╝",
        "",
        f"  Plik wejściowy:   {report.input_file}",
        f"  Plik wyjściowy:   {report.output_file}",
        f"  Tryb:             {report.mode}",
        "",
        f"  Czas oryginału:   {_format_duration(report.original_duration_sec)}",
        f"  Czas po montażu:  {_format_duration(report.processed_duration_sec)}",
        f"  Usunięto:         {_format_duration(report.total_removed_sec)}",
        "",
    ]

    if report.fillers_detected > 0 or report.fillers_removed > 0:
        lines.extend([
            f"  Wypełniacze wykryte:  {report.fillers_detected}",
            f"  Wypełniacze usunięte: {report.fillers_removed}",
        ])

    if report.pauses_shortened > 0:
        lines.append(f"  Pauzy skrócone:       {report.pauses_shortened}")

    if report.repetitions_detected > 0 or report.repetitions_removed > 0:
        lines.extend([
            f"  Powtórzenia wykryte:  {report.repetitions_detected}",
            f"  Powtórzenia usunięte: {report.repetitions_removed}",
        ])

    lines.extend([
        "",
        f"  Plik EDL:         {report.edl_file}",
        f"  Transkrypcja:     {report.transcript_file}",
        "",
    ])

    # Show review items if any
    review_edits = [e for e in report.edits if e.status == EditStatus.REVIEW]
    if review_edits:
        lines.append(f"  ⚠ Elementy do przeglądu: {len(review_edits)}")
        for e in review_edits[:5]:
            lines.append(f"    - [{_format_time(e.start)}] {e.description}")
        if len(review_edits) > 5:
            lines.append(f"    ... i {len(review_edits) - 5} więcej")
        lines.append("")

    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def _format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS.s."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:04.1f}s"
    return f"{minutes}m {secs:04.1f}s"
