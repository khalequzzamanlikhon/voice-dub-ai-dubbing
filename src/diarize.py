"""[2] Speaker Diarization

Uses pyannote.audio 3.1's pretrained pipeline to produce speaker-labeled
time segments. Requires a Hugging Face token with access to the gated
`pyannote/speaker-diarization-3.1` model (accept the license on the model
page, then set HF_TOKEN in .env).

Overlap handling (see spec §4 "Decision points"): pyannote emits
overlapping turns when two speakers talk simultaneously. Rather than
silently dropping or merging these, we flag them (`Segment.overlap = True`)
so downstream stages can decide: v1 behavior is "keep both segments, dub
sequentially, log a warning" — see docs/evaluation.md for why we didn't
attempt concurrent-speech synthesis for a portfolio-scope project.
"""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from src.types import Segment

_pipeline_cache = None


def _load_pipeline(device: str = "cuda"):
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    from pyannote.audio import Pipeline
    import torch

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. pyannote/speaker-diarization-3.1 is a gated "
            "model — accept its license on Hugging Face and set HF_TOKEN "
            "in your .env (see .env.example)."
        )

    logger.info("Loading pyannote speaker-diarization-3.1 pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        pipeline.to(torch.device(device))

    _pipeline_cache = pipeline
    return pipeline


def diarize(
    audio_path: Path,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Segment]:
    """Run diarization and return coarse speaker-turn segments (pre-transcription).

    These segments have empty `text` — transcribe.py fills that in next,
    ideally re-using these speaker boundaries to avoid a second diarization
    pass inside faster-whisper.
    """
    pipeline = _load_pipeline(device)

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    logger.info(f"Diarizing {audio_path}...")
    diarization = pipeline(str(audio_path), **kwargs)

    raw_turns = list(diarization.itertracks(yield_label=True))
    segments: list[Segment] = []
    for i, (turn, _, speaker) in enumerate(raw_turns):
        segments.append(
            Segment(
                id=f"seg_{i:04d}",
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            )
        )

    _flag_overlaps(segments)

    n_speakers = len({s.speaker for s in segments})
    logger.info(f"Diarization found {len(segments)} turns across {n_speakers} speaker(s).")
    return segments


def _flag_overlaps(segments: list[Segment]) -> None:
    """Mark segments whose time window overlaps a different speaker's turn."""
    segments.sort(key=lambda s: s.start)
    for i, seg in enumerate(segments):
        for other in segments[max(0, i - 3): i + 3]:
            if other is seg or other.speaker == seg.speaker:
                continue
            if seg.start < other.end and other.start < seg.end:
                seg.overlap = True
                break


def single_speaker_fallback(duration: float) -> list[Segment]:
    """When diarization is disabled (single-speaker mode, M4 in the
    milestones), return one segment spanning the whole clip. transcribe.py
    will subdivide it using whisper's own segment boundaries."""
    return [Segment(id="seg_0000", speaker="SPEAKER_00", start=0.0, end=duration)]
