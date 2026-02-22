"""Unit tests for Audio Assembly modules."""

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_assembly.models import (
    Config,
    EditCandidate,
    EditStatus,
    EditType,
    ProcessingMode,
    SoftenMode,
    WordSegment,
    TranscriptionResult,
)
from audio_assembly.vad import detect_voice_activity, detect_silences, extract_room_tone
from audio_assembly.fillers import detect_fillers, detect_long_pauses_edits
from audio_assembly.repetitions import detect_repetitions
from audio_assembly.renderer import render_audio
from audio_assembly.report import generate_report, save_edl_json, format_report_summary
from audio_assembly.audio_io import load_audio, export_audio


def test_vad_basic():
    """Test VAD on simple speech + silence pattern."""
    sr = 16000
    # Create speech-like segment (noise) + silence + speech
    speech1 = np.random.randn(sr * 1).astype(np.float32) * 0.3
    silence = np.random.randn(sr * 2).astype(np.float32) * 0.001
    speech2 = np.random.randn(sr * 1).astype(np.float32) * 0.3

    audio = np.concatenate([speech1, silence, speech2])
    segments = detect_voice_activity(audio, sr)

    assert len(segments) > 0
    # Should have both speech and silence segments
    has_speech = any(s.is_speech for s in segments)
    has_silence = any(not s.is_speech for s in segments)
    assert has_speech, "Should detect speech"
    assert has_silence, "Should detect silence"
    print("  ✓ test_vad_basic passed")


def test_detect_silences():
    """Test silence detection from VAD segments."""
    from audio_assembly.models import VoiceSegment
    segments = [
        VoiceSegment(start=0.0, end=2.0, is_speech=True),
        VoiceSegment(start=2.0, end=5.5, is_speech=False),
        VoiceSegment(start=5.5, end=8.0, is_speech=True),
        VoiceSegment(start=8.0, end=8.3, is_speech=False),
    ]

    silences = detect_silences(segments, min_silence_sec=0.5)
    assert len(silences) == 1  # Only the 3.5s silence
    assert abs(silences[0].duration - 3.5) < 0.01
    print("  ✓ test_detect_silences passed")


def test_detect_fillers():
    """Test filler detection in Polish text."""
    config = Config(confidence_threshold=0.5)
    transcription = TranscriptionResult(
        text="Dzień dobry yyy chciałem eee powiedzieć",
        words=[
            WordSegment(word="Dzień", start=0.0, end=0.5, confidence=0.95),
            WordSegment(word="dobry", start=0.5, end=1.0, confidence=0.95),
            WordSegment(word="yyy", start=1.2, end=1.9, confidence=0.90),
            WordSegment(word="chciałem", start=2.0, end=2.8, confidence=0.92),
            WordSegment(word="eee", start=3.0, end=3.6, confidence=0.88),
            WordSegment(word="powiedzieć", start=3.8, end=4.5, confidence=0.91),
        ],
    )

    edits = detect_fillers(transcription, config)
    filler_words = [e.original_text for e in edits if e.edit_type == EditType.FILLER_REMOVAL]

    assert "yyy" in filler_words, "Should detect 'yyy' as filler"
    assert "eee" in filler_words, "Should detect 'eee' as filler"
    assert "Dzień" not in filler_words, "Should not detect real words as fillers"
    print("  ✓ test_detect_fillers passed")


def test_detect_fillers_no_false_positive():
    """Test that meaningful 'no' is not aggressively removed."""
    config = Config(confidence_threshold=0.75)
    transcription = TranscriptionResult(
        text="no dobrze zgadzam się",
        words=[
            # "no" at start of turn after long pause -> likely confirmation
            WordSegment(word="no", start=5.0, end=5.3, confidence=0.95),
            WordSegment(word="dobrze", start=5.3, end=5.8, confidence=0.95),
            WordSegment(word="zgadzam", start=5.8, end=6.3, confidence=0.93),
            WordSegment(word="się", start=6.3, end=6.5, confidence=0.95),
        ],
    )

    edits = detect_fillers(transcription, config)
    applied_fillers = [
        e for e in edits
        if e.edit_type == EditType.FILLER_REMOVAL and e.status == EditStatus.APPLIED
    ]

    # "no" at start (after implied long pause) should NOT be auto-removed
    no_edits = [e for e in applied_fillers if "no" in e.original_text.lower()]
    assert len(no_edits) == 0, "'no' as confirmation should not be auto-removed"
    print("  ✓ test_detect_fillers_no_false_positive passed")


def test_detect_repetitions_immediate():
    """Test immediate word repetition detection."""
    config = Config(
        mode=ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL,
        confidence_threshold=0.75,
    )
    transcription = TranscriptionResult(
        text="myślę to to jest ważne",
        words=[
            WordSegment(word="myślę", start=0.0, end=0.5, confidence=0.93),
            WordSegment(word="to", start=0.6, end=0.8, confidence=0.95),
            WordSegment(word="to", start=0.8, end=1.0, confidence=0.95),
            WordSegment(word="jest", start=1.0, end=1.3, confidence=0.94),
            WordSegment(word="ważne", start=1.3, end=1.8, confidence=0.92),
        ],
    )

    edits = detect_repetitions(transcription, config)
    rep_edits = [e for e in edits if e.edit_type == EditType.REPETITION_REMOVAL]

    assert len(rep_edits) > 0, "Should detect 'to to' repetition"
    assert any("to" in e.original_text for e in rep_edits)
    print("  ✓ test_detect_repetitions_immediate passed")


def test_detect_sentence_restart():
    """Test sentence restart detection."""
    config = Config(
        mode=ProcessingMode.REMOVE_REPETITIONS_CONTEXTUAL,
        confidence_threshold=0.5,
    )
    transcription = TranscriptionResult(
        text="Chciałem znaczy chodzi mi o to",
        words=[
            WordSegment(word="Chciałem", start=0.0, end=0.8, confidence=0.90),
            WordSegment(word="znaczy", start=1.0, end=1.5, confidence=0.91),
            WordSegment(word="chodzi", start=1.5, end=2.0, confidence=0.93),
            WordSegment(word="mi", start=2.0, end=2.2, confidence=0.95),
            WordSegment(word="o", start=2.2, end=2.3, confidence=0.96),
            WordSegment(word="to", start=2.3, end=2.5, confidence=0.95),
        ],
    )

    edits = detect_repetitions(transcription, config)
    restart_edits = [e for e in edits if e.edit_type == EditType.SENTENCE_RESTART_REMOVAL]

    assert len(restart_edits) > 0, "Should detect sentence restart at 'znaczy'"
    print("  ✓ test_detect_sentence_restart passed")


def test_render_cut():
    """Test audio rendering with CUT mode."""
    sr = 48000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.3
    original_len = len(audio)

    config = Config(soften_mode=SoftenMode.CUT, crossfade_ms=30)
    edits = [
        EditCandidate(
            edit_type=EditType.FILLER_REMOVAL,
            start=2.0, end=3.0,
            confidence=0.95,
            status=EditStatus.APPLIED,
        ),
    ]

    result = render_audio(audio, sr, edits, config)

    # Result should be shorter by approximately 1 second
    expected_len = original_len - sr  # minus ~1 second
    assert abs(len(result) - expected_len) < sr * 0.1, (
        f"Expected ~{expected_len} samples, got {len(result)}"
    )
    print("  ✓ test_render_cut passed")


def test_render_roomtone():
    """Test audio rendering with ROOMTONE mode."""
    sr = 48000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.3
    room_tone = np.random.randn(sr).astype(np.float32) * 0.001

    config = Config(soften_mode=SoftenMode.ROOMTONE, crossfade_ms=30, soften_ms=60)
    edits = [
        EditCandidate(
            edit_type=EditType.FILLER_REMOVAL,
            start=2.0, end=3.0,
            confidence=0.95,
            status=EditStatus.APPLIED,
        ),
    ]

    result = render_audio(audio, sr, edits, config, room_tone=room_tone)

    # ROOMTONE inserts short room tone instead of hard cut
    assert len(result) < len(audio), "Result should be shorter"
    print("  ✓ test_render_roomtone passed")


def test_render_skips_review():
    """Test that renderer skips REVIEW-status edits."""
    sr = 48000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.3
    original_len = len(audio)

    config = Config(soften_mode=SoftenMode.CUT, crossfade_ms=30)
    edits = [
        EditCandidate(
            edit_type=EditType.FILLER_REMOVAL,
            start=2.0, end=3.0,
            confidence=0.5,
            status=EditStatus.REVIEW,
        ),
    ]

    result = render_audio(audio, sr, edits, config)

    assert len(result) == original_len, "REVIEW edits should not be applied"
    print("  ✓ test_render_skips_review passed")


def test_audio_io_roundtrip():
    """Test loading and saving audio."""
    sr = 48000
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.3

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr, subtype='PCM_24')
        filepath = f.name

    try:
        loaded, loaded_sr, meta = load_audio(filepath, target_sr=sr)
        assert loaded_sr == sr
        assert abs(len(loaded) - len(audio)) < 10  # Allow tiny rounding
        assert meta["original_sr"] == sr

        # Export
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            export_audio(loaded, sr, f2.name)
            assert os.path.exists(f2.name)
            info = sf.info(f2.name)
            assert info.samplerate == sr
            os.unlink(f2.name)
    finally:
        os.unlink(filepath)

    print("  ✓ test_audio_io_roundtrip passed")


def test_edl_json():
    """Test EDL/JSON generation."""
    edits = [
        EditCandidate(
            edit_type=EditType.FILLER_REMOVAL,
            start=1.5, end=2.2,
            confidence=0.95,
            status=EditStatus.APPLIED,
            description="Filler: yyy",
            original_text="yyy",
        ),
        EditCandidate(
            edit_type=EditType.PAUSE_SHORTENING,
            start=5.0, end=8.0,
            confidence=1.0,
            status=EditStatus.APPLIED,
            description="Long pause shortened",
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        filepath = f.name

    try:
        import json
        save_edl_json(edits, filepath, metadata={"test": True})
        assert os.path.exists(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert len(data["edits"]) == 2
        assert data["edits"][0]["type"] == "filler_removal"
        assert data["summary"]["applied"] == 2
        assert data["summary"]["total_duration_removed_sec"] > 0
    finally:
        os.unlink(filepath)

    print("  ✓ test_edl_json passed")


def test_report_generation():
    """Test report generation and formatting."""
    edits = [
        EditCandidate(
            edit_type=EditType.FILLER_REMOVAL,
            start=1.0, end=1.5,
            confidence=0.95,
            status=EditStatus.APPLIED,
        ),
        EditCandidate(
            edit_type=EditType.PAUSE_SHORTENING,
            start=5.0, end=8.0,
            confidence=1.0,
            status=EditStatus.APPLIED,
        ),
    ]

    report = generate_report(
        input_file="test.wav",
        output_file="test_output.wav",
        edl_file="test_edl.json",
        transcript_file="test_transcript.txt",
        mode="CLEAN_FILLERS_AND_PAUSES",
        original_duration=60.0,
        processed_duration=56.5,
        edits=edits,
    )

    assert report.fillers_removed == 1
    assert report.pauses_shortened == 1
    assert report.total_removed_sec > 0

    summary = format_report_summary(report)
    assert "Wypełniacze usunięte" in summary
    assert "Pauzy skrócone" in summary
    print("  ✓ test_report_generation passed")


def test_room_tone_extraction():
    """Test room tone extraction from audio."""
    sr = 16000
    # Create audio with clear silence segment
    speech = np.random.randn(sr * 2).astype(np.float32) * 0.3
    silence = np.random.randn(sr * 2).astype(np.float32) * 0.001
    audio = np.concatenate([speech, silence, speech])

    from audio_assembly.models import VoiceSegment
    segments = [
        VoiceSegment(start=0.0, end=2.0, is_speech=True),
        VoiceSegment(start=2.0, end=4.0, is_speech=False),
        VoiceSegment(start=4.0, end=6.0, is_speech=True),
    ]

    room_tone = extract_room_tone(audio, sr, segments, duration_ms=500)

    assert len(room_tone) == int(sr * 0.5)
    # Room tone should be quiet
    assert np.sqrt(np.mean(room_tone**2)) < 0.01
    print("  ✓ test_room_tone_extraction passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("  UNIT TESTS")
    print("=" * 60)
    print()

    tests = [
        test_vad_basic,
        test_detect_silences,
        test_detect_fillers,
        test_detect_fillers_no_false_positive,
        test_detect_repetitions_immediate,
        test_detect_sentence_restart,
        test_render_cut,
        test_render_roomtone,
        test_render_skips_review,
        test_audio_io_roundtrip,
        test_edl_json,
        test_report_generation,
        test_room_tone_extraction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
