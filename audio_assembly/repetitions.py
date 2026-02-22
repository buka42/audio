"""Mode B: Remove repetitions and sentence restarts contextually."""

import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .models import (
    Config,
    EditCandidate,
    EditStatus,
    EditType,
    TranscriptionResult,
    WordSegment,
)

logger = logging.getLogger(__name__)

# Polish restart/correction markers
RESTART_MARKERS = {
    "znaczy", "to znaczy", "to jest", "chodzi mi o to",
    "w sensie", "właściwie", "a raczej", "raczej",
    "przepraszam", "to znaczy że",
}


def detect_repetitions(
    transcription: TranscriptionResult,
    config: Config,
) -> List[EditCandidate]:
    """Detect repetitions and sentence restarts in transcription.

    Analyzes the transcription in context windows and identifies:
    1. Immediate word repetitions ("to to", "ja ja")
    2. Phrase repetitions after short pauses
    3. Sentence restarts with correction markers

    Args:
        transcription: Word-level transcription.
        config: Processing configuration.

    Returns:
        List of edit candidates for repetition removal.
    """
    edits = []
    words = transcription.words

    if len(words) < 2:
        return edits

    # Pass 1: Immediate word repetitions
    edits.extend(_detect_immediate_repetitions(words, config))

    # Pass 2: Phrase repetitions in context windows
    edits.extend(_detect_phrase_repetitions(words, config))

    # Pass 3: Sentence restarts
    edits.extend(_detect_sentence_restarts(words, config))

    # Deduplicate overlapping edits (keep higher confidence)
    edits = _deduplicate_edits(edits)

    logger.info("Detected %d repetition/restart candidates", len(edits))
    return edits


def _detect_immediate_repetitions(
    words: List[WordSegment],
    config: Config,
) -> List[EditCandidate]:
    """Detect immediate word repetitions like 'to to', 'ja ja'."""
    edits = []
    i = 0

    while i < len(words) - 1:
        w1 = words[i].word.lower().strip().rstrip(".,!?…")
        w2 = words[i + 1].word.lower().strip().rstrip(".,!?…")

        if w1 == w2 and len(w1) > 0:
            # Check gap between words
            gap = words[i + 1].start - words[i].end
            if gap < 1.0:  # Within 1 second
                # Check if this is a rhetorical/intentional repetition
                is_rhetorical = _is_rhetorical_repetition(words, i)

                confidence = 0.90 if not is_rhetorical else 0.40
                status = EditStatus.APPLIED if confidence >= config.confidence_threshold else EditStatus.REVIEW

                if config.strict_mode and status == EditStatus.REVIEW:
                    status = EditStatus.SKIPPED

                # Remove the first occurrence (keep the second, often more natural)
                edits.append(EditCandidate(
                    edit_type=EditType.REPETITION_REMOVAL,
                    start=words[i].start,
                    end=words[i].end,
                    confidence=confidence,
                    status=status,
                    description=f"Immediate repetition: '{words[i].word} {words[i + 1].word}'",
                    original_text=f"{words[i].word} {words[i + 1].word}",
                    replacement_text=words[i + 1].word,
                ))

                # Check for triple repetition
                if i + 2 < len(words):
                    w3 = words[i + 2].word.lower().strip().rstrip(".,!?…")
                    if w3 == w1:
                        # Remove first two, keep last
                        edits[-1].end = words[i + 1].end
                        edits[-1].confidence = 0.95
                        edits[-1].description = f"Triple repetition: '{w1}' x3"
                        i += 3
                        continue

                i += 2
                continue

        i += 1

    return edits


def _detect_phrase_repetitions(
    words: List[WordSegment],
    config: Config,
) -> List[EditCandidate]:
    """Detect phrase-level repetitions within context windows.

    Example: 'ja chciałem... ja chciałem powiedzieć...'
    """
    edits = []
    window_sec = config.context_window_sec

    i = 0
    while i < len(words):
        # Build a context window starting from word i
        window_end_time = words[i].start + window_sec
        window_words = []
        j = i
        while j < len(words) and words[j].start < window_end_time:
            window_words.append(j)
            j += 1

        if len(window_words) < 4:
            i += 1
            continue

        # Look for repeated n-grams (2-5 words)
        for ngram_size in range(2, min(6, len(window_words) // 2 + 1)):
            for start_a in range(len(window_words) - ngram_size * 2 + 1):
                idx_a = [window_words[start_a + k] for k in range(ngram_size)]
                text_a = " ".join(words[wi].word.lower().strip().rstrip(".,!?…") for wi in idx_a)

                # Search for matching n-gram after the first one
                for start_b in range(start_a + ngram_size, len(window_words) - ngram_size + 1):
                    idx_b = [window_words[start_b + k] for k in range(ngram_size)]
                    text_b = " ".join(words[wi].word.lower().strip().rstrip(".,!?…") for wi in idx_b)

                    similarity = SequenceMatcher(None, text_a, text_b).ratio()

                    if similarity >= 0.85:
                        # Found a repetition - check gap
                        gap = words[idx_b[0]].start - words[idx_a[-1]].end
                        if gap > 5.0:
                            continue  # Too far apart, probably intentional

                        # Determine which version to keep (prefer the more complete one)
                        keep_b = _is_more_complete(words, idx_a, idx_b)

                        if keep_b:
                            # Remove version A
                            edit_start = words[idx_a[0]].start
                            edit_end = words[idx_a[-1]].end
                            removed_text = " ".join(words[wi].word for wi in idx_a)
                        else:
                            # Remove version B
                            edit_start = words[idx_b[0]].start
                            edit_end = words[idx_b[-1]].end
                            removed_text = " ".join(words[wi].word for wi in idx_b)

                        confidence = similarity * 0.9
                        status = (
                            EditStatus.APPLIED
                            if confidence >= config.confidence_threshold
                            else EditStatus.REVIEW
                        )
                        if config.strict_mode and status == EditStatus.REVIEW:
                            status = EditStatus.SKIPPED

                        edits.append(EditCandidate(
                            edit_type=EditType.REPETITION_REMOVAL,
                            start=edit_start,
                            end=edit_end,
                            confidence=confidence,
                            status=status,
                            description=f"Phrase repetition: '{text_a}' (sim: {similarity:.2f})",
                            original_text=removed_text,
                        ))

        i += 1

    return edits


def _detect_sentence_restarts(
    words: List[WordSegment],
    config: Config,
) -> List[EditCandidate]:
    """Detect sentence restarts where speaker starts over.

    Example: 'Chciałem... znaczy, chodzi mi o to, że...'
    """
    edits = []

    for i, word in enumerate(words):
        normalized = word.word.lower().strip().rstrip(".,!?…")

        # Check if this word is a restart marker
        is_marker = normalized in RESTART_MARKERS

        # Also check bigrams
        if not is_marker and i + 1 < len(words):
            bigram = f"{normalized} {words[i + 1].word.lower().strip().rstrip('.,!?…')}"
            is_marker = bigram in RESTART_MARKERS

        if not is_marker:
            continue

        # Look backward for the incomplete sentence fragment
        fragment_start = _find_fragment_start(words, i)
        if fragment_start is None:
            continue

        # Look forward for the restarted sentence
        restart_end = _find_restart_end(words, i)
        if restart_end is None:
            continue

        # Compare fragment before marker with sentence after marker
        before_text = " ".join(
            w.word.lower().strip() for w in words[fragment_start:i]
        )
        after_text = " ".join(
            w.word.lower().strip() for w in words[i + 1 : restart_end + 1]
        )

        if not before_text or not after_text:
            continue

        # Check similarity - the restart should contain similar content
        similarity = SequenceMatcher(None, before_text, after_text).ratio()

        if similarity < 0.3:
            continue  # Too different, not a restart

        # Remove the incomplete fragment + restart marker
        edit_start = words[fragment_start].start
        edit_end = words[i].end  # Include the marker word

        # Include marker bigram if applicable
        if i + 1 < len(words):
            bigram = f"{normalized} {words[i + 1].word.lower().strip().rstrip('.,!?…')}"
            if bigram in RESTART_MARKERS:
                edit_end = words[i + 1].end

        confidence = min(0.95, 0.5 + similarity * 0.5)
        status = (
            EditStatus.APPLIED
            if confidence >= config.confidence_threshold
            else EditStatus.REVIEW
        )
        if config.strict_mode and status == EditStatus.REVIEW:
            status = EditStatus.SKIPPED

        removed_words = " ".join(w.word for w in words[fragment_start:i + 1])
        edits.append(EditCandidate(
            edit_type=EditType.SENTENCE_RESTART_REMOVAL,
            start=edit_start,
            end=edit_end,
            confidence=confidence,
            status=status,
            description=f"Sentence restart at '{word.word}' (sim: {similarity:.2f})",
            original_text=removed_words,
        ))

    return edits


def _is_rhetorical_repetition(
    words: List[WordSegment],
    idx: int,
) -> bool:
    """Heuristic to detect intentional rhetorical repetitions.

    Rhetorical repetitions are often:
    - Emphatic ("tak, tak!", "nie, nie!")
    - Part of a pattern (3+ repetitions with consistent rhythm)
    - In an exclamatory context
    """
    word = words[idx].word.lower().strip()

    # Common intentional repetitions in Polish
    emphatic_words = {"tak", "nie", "dobrze", "okej", "ok", "proszę", "chodź"}
    if word in emphatic_words:
        return True

    # Check for exclamation context
    if idx + 1 < len(words) and "!" in words[idx + 1].word:
        return True

    return False


def _is_more_complete(
    words: List[WordSegment],
    idx_a: List[int],
    idx_b: List[int],
) -> bool:
    """Determine if version B is more complete than version A.

    Checks what follows each version to decide which is more complete.
    Returns True if B should be kept (A should be removed).
    """
    # If B is followed by more words, B is likely the complete version
    last_b = idx_b[-1]
    last_a = idx_a[-1]

    # Check how much speech follows each version
    a_continuation = 0
    for i in range(last_a + 1, min(last_a + 5, len(words))):
        if words[i].start - words[i - 1].end < 0.5:
            a_continuation += 1
        else:
            break

    b_continuation = 0
    for i in range(last_b + 1, min(last_b + 5, len(words))):
        if words[i].start - words[i - 1].end < 0.5:
            b_continuation += 1
        else:
            break

    # Version with more continuation is more complete
    return b_continuation >= a_continuation


def _find_fragment_start(
    words: List[WordSegment],
    marker_idx: int,
    max_lookback: int = 8,
) -> Optional[int]:
    """Find the start of an incomplete sentence fragment before a restart marker."""
    if marker_idx == 0:
        return None

    # Look back for a natural sentence boundary (long pause or punctuation)
    for i in range(marker_idx - 1, max(0, marker_idx - max_lookback) - 1, -1):
        # Check for significant pause before this word
        if i > 0:
            gap = words[i].start - words[i - 1].end
            if gap > 0.8:  # Long pause suggests sentence boundary
                return i

        # Check for sentence-ending punctuation
        prev_word = words[i].word.strip()
        if prev_word.endswith(('.', '!', '?')):
            return i + 1

    # Default: start from max_lookback words before marker
    return max(0, marker_idx - max_lookback)


def _find_restart_end(
    words: List[WordSegment],
    marker_idx: int,
    max_lookahead: int = 10,
) -> Optional[int]:
    """Find the end of the restarted sentence after a restart marker."""
    if marker_idx >= len(words) - 1:
        return None

    # Look forward for the end of the restarted phrase
    for i in range(marker_idx + 1, min(len(words), marker_idx + max_lookahead + 1)):
        # Sentence end
        if words[i].word.strip().endswith(('.', '!', '?')):
            return i

        # Long pause
        if i + 1 < len(words):
            gap = words[i + 1].start - words[i].end
            if gap > 1.0:
                return i

    return min(len(words) - 1, marker_idx + max_lookahead)


def _deduplicate_edits(edits: List[EditCandidate]) -> List[EditCandidate]:
    """Remove overlapping edit candidates, keeping higher confidence ones."""
    if not edits:
        return edits

    # Sort by start time
    edits.sort(key=lambda e: (e.start, -e.confidence))

    result = []
    for edit in edits:
        # Check if this overlaps with any existing edit
        overlaps = False
        for existing in result:
            if edit.start < existing.end and edit.end > existing.start:
                # Overlap detected - keep the one with higher confidence
                if edit.confidence > existing.confidence:
                    result.remove(existing)
                    result.append(edit)
                overlaps = True
                break

        if not overlaps:
            result.append(edit)

    return sorted(result, key=lambda e: e.start)
