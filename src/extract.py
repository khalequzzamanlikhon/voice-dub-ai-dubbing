"""[1] Audio Extraction

Pulls a mono 16kHz WAV track out of an arbitrary input video/audio file.
16kHz mono is the format faster-whisper and pyannote both expect natively,
so we standardize here rather than re-resampling at every downstream stage.
"""
from __future__ import annotations

from pathlib import Path

import ffmpeg
from loguru import logger

TARGET_SR = 16000


def extract_audio(input_path: Path, work_dir: Path) -> Path:
    """Extract a mono 16kHz WAV from `input_path` into `work_dir/audio.wav`.

    Works whether the input is already audio-only (mp3/wav/m4a) or a video
    container (mp4/mov/mkv) — ffmpeg demuxes the audio stream either way.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "audio.wav"

    logger.info(f"Extracting audio: {input_path} -> {out_path}")
    (
        ffmpeg
        .input(str(input_path))
        .output(
            str(out_path),
            ac=1,               # mono
            ar=TARGET_SR,       # 16kHz
            acodec="pcm_s16le",
            vn=None,             # no video
        )
        .overwrite_output()
        .run(quiet=True)
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output for {input_path}")

    logger.info(f"Extracted audio: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    info = ffmpeg.probe(str(path))
    return float(info["format"]["duration"])


def has_video_stream(path: Path) -> bool:
    """True if the input container has a video stream (so we know whether
    a re-mux step at the end of the pipeline is even possible)."""
    info = ffmpeg.probe(str(path))
    return any(s["codec_type"] == "video" for s in info["streams"])
