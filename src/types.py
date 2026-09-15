"""Shared dataclasses / typed structures passed between pipeline stages.

Keeping these in one module (rather than letting each stage invent its own
dict shape) is what makes `pipeline.py` legible: every stage has a clear
input/output contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Word:
    """A single word with timing, as produced by faster-whisper."""
    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Segment:
    """One utterance: a diarized speaker turn that has been transcribed,
    (optionally) translated, and (eventually) re-synthesized.

    This object is mutated / enriched as it flows through the pipeline
    stages rather than being re-wrapped at each step — diarize.py creates
    it, transcribe.py fills `text`/`words`, translate.py fills
    `translated_text`, clone_tts.py fills `synth_audio_path`, align.py
    fills `aligned_audio_path`.
    """
    id: str
    speaker: str
    start: float                       # seconds, in ORIGINAL audio timeline
    end: float
    text: str = ""
    words: list[Word] = field(default_factory=list)
    translated_text: str = ""
    synth_audio_path: Optional[Path] = None
    aligned_audio_path: Optional[Path] = None
    synth_duration: float = 0.0        # duration of raw TTS output, seconds
    target_duration: float = 0.0       # = end - start
    overlap: bool = False              # True if this segment overlaps a neighbor

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerProfile:
    """Reference audio + metadata used to clone one speaker's voice."""
    speaker_id: str
    reference_clip_path: Path
    reference_duration: float
    embedding: Optional[list[float]] = None  # populated for eval (resemblyzer)


@dataclass
class PipelineConfig:
    """All tunables in one place. Defaults match the spec's tech-stack table."""

    # I/O
    input_path: Path = Path("input.mp4")
    work_dir: Path = Path("outputs/tmp")
    output_path: Path = Path("outputs/dubbed.mp4")

    # Languages (ISO 639-1, e.g. "en", "de", "es")
    source_lang: str = "en"
    target_lang: str = "es"

    # Stage toggles
    diarize: bool = True
    remux_video: bool = True
    preserve_background: bool = False   # Demucs source separation (stretch goal)

    # STT
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "float16"

    # Diarization
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Translation
    translation_backend: str = "nllb"   # "nllb" | "gpt"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    gpt_model: str = "gpt-4o-mini"

    # TTS / voice cloning
    tts_backend: str = "xtts"           # "xtts" | "f5"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    min_reference_seconds: float = 6.0

    # Time alignment
    max_stretch_ratio: float = 1.35     # beyond this, we re-phrase instead of stretching
    min_stretch_ratio: float = 0.75
    segment_tolerance_ms: float = 300.0  # success-criteria target from spec

    # Misc
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.output_path = Path(self.output_path)
        self.input_path = Path(self.input_path)
        self.work_dir.mkdir(parents=True, exist_ok=True)
