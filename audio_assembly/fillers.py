"""Mode A: Clean fillers and pauses for Polish audio."""

import logging
import re
from typing import List

from .models import (
    Config,
    EditCandidate,
    EditStatus,
    EditType,
    SilenceSegment,
    TranscriptionResult,
    WordSegment,
)

logger = logging.getLogger(__name__)

# Polish filler words - categorized by aggressiveness
# Always remove: hesitation sounds
FILLER_ALWAYS = {
    "yyy", "eee", "yy", "ee", "em", "ymm", "ym", "umm", "um",
    "aaa", "aa", "mmm", "mm", "ehm", "öhm", "hm",
}

# Remove cautiously: discourse markers that CAN be fillers
FILLER_CAUTIOUS = {
    "noo", "nooo",  # prolonged "no" as hesitation
    "no",  # can be confirmation - check context
}

# Remove optionally: discourse markers
FILLER_OPTIONAL = {
    "w sensie", "jakby", "tzn", "znaczy", "tak jakby",
    "powiedzmy", "wiesz", "nie wiem",
}

# Sentence restart markers
RESTART_MARKERS = {
    "znaczy", "to znaczy", "to jest", "chodzi mi o to",
    "w sensie", "właściwie",
}


def detect_fillers(
    transcription: TranscriptionResult,
    config: Config,
) -> List[EditCandidate]:
    """Detect filler words and hesitations in transcription.

    Args:
        transcription: Word-level transcription result.
        config: Processing configuration.

    Returns:
        List of edit candidates for filler removal.
    """
    edits = []
    words = transcription.words

    for i, word in enumerate(words):
        normalized = word.word.lower().strip().rstrip(".,!?…")

        # Check always-remove fillers
        if normalized in FILLER_ALWAYS:
            edits.append(EditCandidate(
                edit_type=EditType.FILLER_REMOVAL,
                start=word.start,
                end=word.end,
                confidence=0.95,
                status=EditStatus.APPLIED,
                description=f"Filler word: '{word.word}'",
                original_text=word.word,
            ))
            continue

        # Check cautious fillers with context
        if normalized in FILLER_CAUTIOUS:
            confidence = _evaluate_filler_context(words, i, normalized)
            status = EditStatus.APPLIED if confidence >= config.confidence_threshold else EditStatus.REVIEW
            if config.strict_mode and status == EditStatus.REVIEW:
                status = EditStatus.SKIPPED
            edits.append(EditCandidate(
                edit_type=EditType.FILLER_REMOVAL,
                start=word.start,
                end=word.end,
                confidence=confidence,
                status=status,
                description=f"Cautious filler: '{word.word}' (conf: {confidence:.2f})",
                original_text=word.word,
            ))
            continue

        # Check multi-word fillers
        if i < len(words) - 1:
            bigram = f"{normalized} {words[i + 1].word.lower().strip().rstrip('.,!?…')}"
            if bigram in FILLER_OPTIONAL:
                edits.append(EditCandidate(
                    edit_type=EditType.FILLER_REMOVAL,
                    start=word.start,
                    end=words[i + 1].end,
                    confidence=0.70,
                    status=EditStatus.APPLIED if 0.70 >= config.confidence_threshold else EditStatus.REVIEW,
                    description=f"Optional filler phrase: '{bigram}'",
                    original_text=bigram,
                ))

        # Check for prolonged syllables (hesitation-like)
        if _is_prolonged_hesitation(word, words, i):
            edits.append(EditCandidate(
                edit_type=EditType.FILLER_REMOVAL,
                start=word.start,
                end=word.end,
                confidence=0.85,
                status=EditStatus.APPLIED,
                description=f"Prolonged hesitation: '{word.word}'",
                original_text=word.word,
            ))

    logger.info("Detected %d filler candidates", len(edits))
    return edits


def detect_long_pauses_edits(
    silences: List[SilenceSegment],
    config: Config,
) -> List[EditCandidate]:
    """Generate edit candidates for long pauses that should be shortened.

    Args:
        silences: Detected silence segments.
        config: Processing configuration.

    Returns:
        List of edit candidates for pause shortening.
    """
    edits = []
    for silence in silences:
        if silence.duration >= config.silence_threshold_sec:
            # We want to keep max_pause_after_edit_sec of the pause
            keep_sec = config.max_pause_after_edit_sec
            # Cut from the middle, keeping some at start and end
            keep_start = keep_sec / 2
            keep_end = keep_sec / 2

            edit_start = silence.start + keep_start
            edit_end = silence.end - keep_end

            if edit_end > edit_start:
                edits.append(EditCandidate(
                    edit_type=EditType.PAUSE_SHORTENING,
                    start=edit_start,
                    end=edit_end,
                    confidence=1.0,
                    status=EditStatus.APPLIED,
                    description=(
                        f"Shorten pause from {silence.duration:.1f}s "
                        f"to {keep_sec:.1f}s"
                    ),
                ))

    logger.info("Detected %d long pauses to shorten", len(edits))
    return edits


def _evaluate_filler_context(
    words: List[WordSegment],
    idx: int,
    normalized: str,
) -> float:
    """Evaluate whether a cautious filler word is actually a filler based on context.

    Returns confidence score 0.0-1.0.
    """
    # "no" at the start of a sentence or after a long pause is likely a confirmation
    if normalized == "no":
        # Check if it's a standalone confirmation
        next_gap = 0.0

        if idx == 0:
            # First word in transcript = beginning of turn, treat as after long pause
            return 0.3  # Likely a confirmation, don't remove

        prev_gap = words[idx].start - words[idx - 1].end
        if idx < len(words) - 1:
            next_gap = words[idx + 1].start - words[idx].end

        # If there's a significant pause before and it's at turn boundary
        if prev_gap > 0.5:
            return 0.3  # Likely a confirmation, don't remove

        # If followed by continuation without pause, more likely a filler
        if next_gap < 0.2:
            return 0.85

        return 0.5  # Uncertain

    # Prolonged "noo" is almost always a filler
    if normalized in ("noo", "nooo"):
        return 0.9

    return 0.7


def _is_prolonged_hesitation(
    word: WordSegment,
    words: List[WordSegment],
    idx: int,
) -> bool:
    """Check if a word represents a prolonged hesitation (stretched syllable).

    Heuristic: if a short word has unusually long duration, it's likely a hesitation.
    """
    normalized = word.word.lower().strip()
    duration = word.end - word.start

    # Very short words (1-3 chars) lasting > 0.5s are suspicious
    if len(normalized) <= 3 and duration > 0.5:
        # But not if it's a real word with meaning
        real_short_words = {
            "ja", "ty", "on", "my", "wy", "to", "te", "ta", "tu",
            "do", "od", "na", "po", "za", "ze", "we", "co", "bo",
            "że", "by", "i", "a", "o", "w", "z", "u", "już", "nie",
            "tak", "ale", "jak", "gdy",
        }
        if normalized not in real_short_words:
            return True

    # Repeated vowel patterns suggest hesitation
    if re.match(r'^[aeiouyę]+$', normalized) and duration > 0.3:
        return True

    return False
