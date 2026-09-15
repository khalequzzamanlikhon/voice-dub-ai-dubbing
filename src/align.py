"""[6] Time-Alignment / Stretching

Decision point from spec §4: what do we do when translated speech is much
longer/shorter than the original slot?

Strategy (three-tier):
  1. If synth_duration is within [min_stretch_ratio, max_stretch_ratio] of
     target_duration, time-stretch with pyrubberband (pitch-preserving) to
     fit exactly.
  2. If the required stretch would exceed those ratios, DON'T degrade audio
     quality by over-stretching — instead flag the segment for
     re-phrasing: ask the translation LLM to redo the line targeting a
     shorter/longer syllable count, then re-synthesize once. This is only
     available when translation_backend="gpt" (NLLB has no
     re-prompting mechanism); NLLB-backend pipelines fall through to
     tier 3.
  3. If re-phrasing isn't available or still doesn't fit, accept drift:
     stretch to the nearest allowed ratio (clamped) and let the segment
     start slightly early/run slightly long. We log the residual drift in
     ms so it shows up in the evaluation report (spec §7 "Timing drift").

This mirrors real dubbing-industry practice: professional dubs re-write
lines to fit timing constraints far more often than they time-stretch
audio, because stretched speech beyond ~±25-35% sounds artificial.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.types import PipelineConfig, Segment


def align_segments(
    segments: list[Segment],
    config: PipelineConfig,
    retranslate_fn=None,
    resynthesize_fn=None,
) -> list[Segment]:
    """Time-stretch (or re-phrase + re-synthesize) each segment to fit its
    original time slot. Fills `aligned_audio_path`.

    `retranslate_fn(segment, target_seconds) -> str` and
    `resynthesize_fn(segment) -> Path` are injected by pipeline.py so this
    module doesn't need to import translate.py/clone_tts.py directly
    (keeps stages decoupled and independently testable).
    """
    out_dir = Path(config.work_dir) / "aligned_segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_drift_ms = 0.0
    n_stretched = n_rephrased = n_drifted = 0

    for seg in segments:
        if seg.synth_audio_path is None:
            continue

        seg.target_duration = seg.duration
        ratio = seg.target_duration / max(seg.synth_duration, 1e-6)

        if config.min_stretch_ratio <= ratio <= config.max_stretch_ratio:
            out_path = out_dir / f"{seg.id}.wav"
            _time_stretch(seg.synth_audio_path, out_path, ratio)
            seg.aligned_audio_path = out_path
            n_stretched += 1
            continue

        # Tier 2: re-phrase, if we have the hook for it
        if retranslate_fn is not None and resynthesize_fn is not None:
            logger.info(
                f"{seg.id}: required stretch ratio {ratio:.2f} outside "
                f"[{config.min_stretch_ratio}, {config.max_stretch_ratio}] — re-phrasing."
            )
            try:
                seg.translated_text = retranslate_fn(seg, seg.target_duration)
                seg.synth_audio_path = resynthesize_fn(seg)
                seg.synth_duration = _wav_duration(seg.synth_audio_path)
                ratio = seg.target_duration / max(seg.synth_duration, 1e-6)
                n_rephrased += 1
            except Exception as e:
                logger.warning(f"{seg.id}: re-phrase attempt failed ({e}); falling back to clamped stretch.")

        # Tier 3: clamp and accept drift
        clamped_ratio = max(config.min_stretch_ratio, min(config.max_stretch_ratio, ratio))
        out_path = out_dir / f"{seg.id}.wav"
        _time_stretch(seg.synth_audio_path, out_path, clamped_ratio)
        seg.aligned_audio_path = out_path

        actual_duration = seg.synth_duration * clamped_ratio
        drift_ms = abs(actual_duration - seg.target_duration) * 1000
        total_drift_ms += drift_ms
        n_drifted += 1
        if drift_ms > config.segment_tolerance_ms:
            logger.warning(
                f"{seg.id}: residual drift {drift_ms:.0f}ms exceeds tolerance "
                f"({config.segment_tolerance_ms:.0f}ms) even after clamped stretch."
            )

    logger.info(
        f"Alignment summary: {n_stretched} clean-stretched, {n_rephrased} re-phrased, "
        f"{n_drifted} accepted drift (avg {total_drift_ms / max(n_drifted, 1):.0f}ms over tolerance)."
    )
    return segments


def _time_stretch(in_path: Path, out_path: Path, ratio: float) -> None:
    """Pitch-preserving time-stretch. `ratio` = target_duration / current_duration,
    i.e. ratio > 1 means SLOW DOWN (stretch longer), ratio < 1 means speed up.
    """
    try:
        _time_stretch_rubberband(in_path, out_path, ratio)
    except (ImportError, OSError) as e:
        logger.warning(f"pyrubberband unavailable ({e}); falling back to ffmpeg atempo.")
        _time_stretch_ffmpeg(in_path, out_path, ratio)


def _time_stretch_rubberband(in_path: Path, out_path: Path, ratio: float) -> None:
    import pyrubberband as pyrb
    import soundfile as sf

    y, sr = sf.read(str(in_path))
    # pyrubberband's `rate` param is a speed multiplier: to make audio LONGER
    # (ratio > 1) we need to slow it down, i.e. rate = 1/ratio.
    stretched = pyrb.time_stretch(y, sr, rate=1.0 / ratio)
    sf.write(str(out_path), stretched, sr)


def _time_stretch_ffmpeg(in_path: Path, out_path: Path, ratio: float) -> None:
    """ffmpeg's atempo filter accepts [0.5, 2.0] per instance; chain
    multiple instances for ratios outside that range."""
    import ffmpeg

    speed = 1.0 / ratio  # atempo speed-up factor
    factors = []
    remaining = speed
    while remaining < 0.5 or remaining > 2.0:
        step = 2.0 if remaining > 2.0 else 0.5
        factors.append(step)
        remaining /= step
    factors.append(remaining)

    stream = ffmpeg.input(str(in_path))
    for f in factors:
        stream = stream.filter("atempo", f)
    stream.output(str(out_path)).overwrite_output().run(quiet=True)


def _wav_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate
