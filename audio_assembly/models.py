"""Data models and configuration for Audio Assembly."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProcessingMode(Enum):
    CLEAN_FILLERS_AND_PAUSES = "A"
    REMOVE_REPETITIONS_CONTEXTUAL = "B"


class SoftenMode(Enum):
    CUT = "CUT"
    FADE = "FADE"
    ROOMTONE = "ROOMTONE"


class EditType(Enum):
    FILLER_REMOVAL = "filler_removal"
    PAUSE_SHORTENING = "pause_shortening"
    REPETITION_REMOVAL = "repetition_removal"
    SENTENCE_RESTART_REMOVAL = "sentence_restart_removal"


class EditStatus(Enum):
    APPLIED = "applied"
    REVIEW = "review"
    SKIPPED = "skipped"


@dataclass
class Config:
    """Processing configuration with all user-adjustable parameters."""
    mode: ProcessingMode = ProcessingMode.CLEAN_FILLERS_AND_PAUSES
    silence_threshold_sec: float = 3.0
    min_gap_to_edit_ms: int = 120
    crossfade_ms: int = 60
    soften_mode: SoftenMode = SoftenMode.ROOMTONE
    soften_ms: int = 120
    max_pause_after_edit_sec: float = 0.4
    start_offset_sec: Optional[float] = None
    end_offset_sec: Optional[float] = None
    strict_mode: bool = False
    confidence_threshold: float = 0.75
    analysis_sample_rate: int = 16000
    export_sample_rate: int = 48000
    whisper_model: str = "base"
    context_window_sec: float = 15.0


@dataclass
class WordSegment:
    """A single word with timing information from transcription."""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class VoiceSegment:
    """A voice activity segment."""
    start: float
    end: float
    is_speech: bool = True


@dataclass
class SilenceSegment:
    """A detected silence/pause segment."""
    start: float
    end: float
    duration: float = 0.0

    def __post_init__(self):
        self.duration = self.end - self.start


@dataclass
class EditCandidate:
    """A candidate edit operation."""
    edit_type: EditType
    start: float
    end: float
    confidence: float = 1.0
    status: EditStatus = EditStatus.APPLIED
    description: str = ""
    original_text: str = ""
    replacement_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Full transcription result with word-level timestamps."""
    text: str = ""
    words: list = field(default_factory=list)
    language: str = "pl"


@dataclass
class ProcessingReport:
    """Report of all processing operations."""
    input_file: str = ""
    output_file: str = ""
    edl_file: str = ""
    transcript_file: str = ""
    mode: str = ""
    original_duration_sec: float = 0.0
    processed_duration_sec: float = 0.0
    total_removed_sec: float = 0.0
    fillers_detected: int = 0
    fillers_removed: int = 0
    pauses_shortened: int = 0
    repetitions_detected: int = 0
    repetitions_removed: int = 0
    edits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "edl_file": self.edl_file,
            "transcript_file": self.transcript_file,
            "mode": self.mode,
            "original_duration_sec": round(self.original_duration_sec, 3),
            "processed_duration_sec": round(self.processed_duration_sec, 3),
            "total_removed_sec": round(self.total_removed_sec, 3),
            "fillers_detected": self.fillers_detected,
            "fillers_removed": self.fillers_removed,
            "pauses_shortened": self.pauses_shortened,
            "repetitions_detected": self.repetitions_detected,
            "repetitions_removed": self.repetitions_removed,
            "edits": [
                {
                    "type": e.edit_type.value,
                    "start": round(e.start, 3),
                    "end": round(e.end, 3),
                    "duration": round(e.duration, 3),
                    "confidence": round(e.confidence, 3),
                    "status": e.status.value,
                    "description": e.description,
                    "original_text": e.original_text,
                }
                for e in self.edits
            ],
        }
