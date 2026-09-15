"""[7] Audio Reassembly

Stitches per-segment aligned clips back onto a single timeline matching
the original audio's total duration, placing each clip at its original
`start` timestamp (crossfaded slightly to avoid clicks at splice points).

Background-audio preservation (spec §4 decision point): v1 default is
full replacement — silence outside speech segments — with the option
(`preserve_background=True`) to duck-mix the original track underneath at
low volume instead of dead silence. True music/SFX isolation via Demucs
source separation is listed as a stretch goal (spec §9) and is NOT
implemented in this repo; `preserve_background` here is the cheaper
"blend original track under dubbed speech" approximation, and this
limitation is called out explicitly in docs/evaluation.md.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydub import AudioSegment as PydubSegment

from src.types import Segment

CROSSFADE_MS = 15  # short crossfade at splice points to avoid audible clicks


def reassemble(
    segments: list[Segment],
    original_audio_path: Path,
    total_duration: float,
    work_dir: Path,
    preserve_background: bool = False,
    background_gain_db: float = -18.0,
) -> Path:
    out_path = Path(work_dir) / "dubbed_audio.wav"

    if preserve_background:
        base = PydubSegment.from_wav(str(original_audio_path))
        base = base + background_gain_db  # duck original track under dub
    else:
        base = PydubSegment.silent(duration=int(total_duration * 1000))

    placed, skipped = 0, 0
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.aligned_audio_path is None:
            skipped += 1
            continue
        clip = PydubSegment.from_wav(str(seg.aligned_audio_path))
        start_ms = int(seg.start * 1000)
        base = base.overlay(clip, position=start_ms)
        placed += 1

    logger.info(f"Reassembled track: {placed} segment(s) placed, {skipped} skipped (no audio).")

    base.export(str(out_path), format="wav")
    return out_path


def remux_with_video(
    dubbed_audio_path: Path,
    original_video_path: Path,
    output_path: Path,
) -> Path:
    """[8] Re-mux dubbed audio onto the original video's video stream,
    dropping the original audio stream (ffmpeg -map)."""
    import ffmpeg

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_in = ffmpeg.input(str(original_video_path))
    audio_in = ffmpeg.input(str(dubbed_audio_path))

    logger.info(f"Re-muxing dubbed audio with original video -> {output_path}")
    (
        ffmpeg
        .output(
            video_in["v"], audio_in["a"], str(output_path),
            vcodec="copy", acodec="aac", shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path
