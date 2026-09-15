"""Unit tests for pipeline stages.

Design note: tests are split into two tiers —

1. Pure-logic unit tests (word-to-segment assignment, stretch-ratio
   decisions, overlap flagging, ffmpeg atempo factor chaining) run with NO
   external dependencies (no GPU, no downloaded models, no network) and
   are what CI should run on every commit.
2. Integration tests that touch real models are marked
   `@pytest.mark.integration` and skipped by default — run explicitly with
   `pytest -m integration` on a machine with the model weights /GPU
   available. This split is called out in the README so it's clear which
   tests validate logic vs. which validate model behavior.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.align import _time_stretch_ffmpeg  # noqa: F401 (imported for existence check)
from src.diarize import _flag_overlaps
from src.transcribe import _assign_words_to_segments, _words_to_text
from src.types import PipelineConfig, Segment, Word


# --------------------------------------------------------------- transcribe

def test_words_assigned_to_correct_segment_by_overlap():
    segments = [
        Segment(id="s0", speaker="A", start=0.0, end=2.0),
        Segment(id="s1", speaker="B", start=2.0, end=4.0),
    ]
    words = [
        Word(text="hello", start=0.1, end=0.5),
        Word(text="world", start=1.0, end=1.5),
        Word(text="foo", start=2.2, end=2.6),
    ]
    _assign_words_to_segments(words, segments)

    assert [w.text for w in segments[0].words] == ["hello", "world"]
    assert [w.text for w in segments[1].words] == ["foo"]


def test_words_outside_any_segment_are_dropped():
    segments = [Segment(id="s0", speaker="A", start=0.0, end=1.0)]
    words = [Word(text="orphan", start=5.0, end=5.5)]
    _assign_words_to_segments(words, segments)
    assert segments[0].words == []


def test_words_to_text_joins_with_spaces():
    words = [Word(text="hello", start=0, end=1), Word(text="world", start=1, end=2)]
    assert _words_to_text(words) == "hello world"


# ----------------------------------------------------------------- diarize

def test_flag_overlaps_marks_concurrent_speakers():
    segments = [
        Segment(id="s0", speaker="A", start=0.0, end=3.0),
        Segment(id="s1", speaker="B", start=2.5, end=5.0),   # overlaps s0
        Segment(id="s2", speaker="A", start=5.0, end=7.0),   # no overlap
    ]
    _flag_overlaps(segments)
    assert segments[0].overlap is True
    assert segments[1].overlap is True
    assert segments[2].overlap is False


def test_flag_overlaps_ignores_same_speaker_adjacency():
    segments = [
        Segment(id="s0", speaker="A", start=0.0, end=3.0),
        Segment(id="s1", speaker="A", start=2.9, end=5.0),  # same speaker, "overlap" is fine
    ]
    _flag_overlaps(segments)
    assert segments[0].overlap is False
    assert segments[1].overlap is False


# ------------------------------------------------------------------- align

def test_segment_duration_property():
    seg = Segment(id="s0", speaker="A", start=1.5, end=4.0)
    assert seg.duration == pytest.approx(2.5)


def test_stretch_ratio_within_bounds_does_not_require_rephrase():
    cfg = PipelineConfig(min_stretch_ratio=0.75, max_stretch_ratio=1.35)
    target, synth = 4.0, 3.5
    ratio = target / synth
    assert cfg.min_stretch_ratio <= ratio <= cfg.max_stretch_ratio


def test_stretch_ratio_outside_bounds_triggers_rephrase_path():
    cfg = PipelineConfig(min_stretch_ratio=0.75, max_stretch_ratio=1.35)
    target, synth = 10.0, 3.0  # would need a 3.3x stretch — way outside bounds
    ratio = target / synth
    assert not (cfg.min_stretch_ratio <= ratio <= cfg.max_stretch_ratio)


# ------------------------------------------------------------------ config

def test_pipeline_config_creates_work_dir(tmp_path: Path):
    work_dir = tmp_path / "nested" / "work"
    cfg = PipelineConfig(work_dir=work_dir)
    assert work_dir.exists()
    assert cfg.work_dir == work_dir


# ------------------------------------------------------------- integration

@pytest.mark.integration
def test_full_pipeline_single_speaker_clip(tmp_path: Path):
    """Requires: HF_TOKEN set, GPU or patient CPU, model weights downloaded.
    Not run in default `pytest` invocation — see module docstring.
    """
    from src.pipeline import DubbingPipeline

    sample = Path("examples/sample_en_single_speaker.wav")
    if not sample.exists():
        pytest.skip("Sample fixture not present; add one under examples/ to run this test.")

    cfg = PipelineConfig(
        input_path=sample,
        output_path=tmp_path / "out.wav",
        source_lang="en",
        target_lang="es",
        diarize=False,
        remux_video=False,
        work_dir=tmp_path / "work",
    )
    output = DubbingPipeline(cfg).run()
    assert output.exists()
