"""[5] Voice Cloning + TTS Synthesis

Default backend is Coqui XTTS-v2: zero-shot multilingual cloning from a
~6s reference clip. An F5-TTS backend stub is included behind the same
interface (`synthesize_segments`) since the spec lists it as an
alternative — swap `tts_backend="f5"` in PipelineConfig once
https://github.com/SWivid/F5-TTS is installed; see the TODO below.

Reference-clip extraction: per speaker, we take the longest contiguous
diarized turn (up to ~15s) as the cloning reference, on the assumption
that a longer, single-speaker, non-overlapping clip gives XTTS the
cleanest voice signal. Turns flagged `overlap=True` are excluded from
reference-clip candidates.
"""
from __future__ import annotations

from pathlib import Path

import soundfile as sf
from loguru import logger

from src.types import Segment, SpeakerProfile

_xtts_cache = None


def build_speaker_profiles(
    audio_path: Path,
    segments: list[Segment],
    work_dir: Path,
    min_reference_seconds: float = 6.0,
    max_reference_seconds: float = 15.0,
) -> dict[str, SpeakerProfile]:
    """Pick the best reference clip per speaker and write it to disk."""
    import soundfile as sf_
    audio, sr = sf_.read(str(audio_path))

    by_speaker: dict[str, list[Segment]] = {}
    for seg in segments:
        if seg.overlap:
            continue
        by_speaker.setdefault(seg.speaker, []).append(seg)

    profiles: dict[str, SpeakerProfile] = {}
    ref_dir = Path(work_dir) / "speaker_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    for speaker, segs in by_speaker.items():
        segs.sort(key=lambda s: s.duration, reverse=True)
        best = segs[0]
        if best.duration < min_reference_seconds:
            logger.warning(
                f"Speaker {speaker}: best available turn is only "
                f"{best.duration:.1f}s (< {min_reference_seconds}s minimum). "
                f"Cloning quality may degrade — see docs/evaluation.md."
            )

        clip_end = min(best.end, best.start + max_reference_seconds)
        s_idx, e_idx = int(best.start * sr), int(clip_end * sr)
        clip = audio[s_idx:e_idx]

        ref_path = ref_dir / f"{speaker}.wav"
        sf.write(str(ref_path), clip, sr)

        profiles[speaker] = SpeakerProfile(
            speaker_id=speaker,
            reference_clip_path=ref_path,
            reference_duration=(clip_end - best.start),
        )
        logger.info(f"Reference clip for {speaker}: {ref_path} ({clip_end - best.start:.1f}s)")

    return profiles


def synthesize_segments(
    segments: list[Segment],
    speaker_profiles: dict[str, SpeakerProfile],
    target_lang: str,
    work_dir: Path,
    backend: str = "xtts",
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    device: str = "cuda",
) -> list[Segment]:
    """Synthesize `Segment.translated_text` in the cloned voice of
    `Segment.speaker`. Fills `synth_audio_path` and `synth_duration`.
    """
    out_dir = Path(work_dir) / "synth_segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "xtts":
        _synthesize_xtts(segments, speaker_profiles, target_lang, out_dir, xtts_model, device)
    elif backend == "f5":
        _synthesize_f5(segments, speaker_profiles, target_lang, out_dir, device)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")

    return segments


# ----------------------------------------------------------------------- XTTS

def _load_xtts(model_name: str, device: str):
    global _xtts_cache
    if _xtts_cache is not None:
        return _xtts_cache

    from TTS.api import TTS

    logger.info(f"Loading XTTS-v2 ('{model_name}')...")
    tts = TTS(model_name).to(device)
    _xtts_cache = tts
    return tts


def _synthesize_xtts(segments, speaker_profiles, target_lang, out_dir, model_name, device):
    tts = _load_xtts(model_name, device)

    logger.info(f"Synthesizing {len(segments)} segment(s) via XTTS-v2...")
    for seg in segments:
        if not seg.translated_text.strip():
            continue
        profile = speaker_profiles.get(seg.speaker)
        if profile is None:
            logger.warning(f"No reference profile for speaker {seg.speaker}, skipping {seg.id}")
            continue

        out_path = out_dir / f"{seg.id}.wav"
        tts.tts_to_file(
            text=seg.translated_text,
            speaker_wav=str(profile.reference_clip_path),
            language=target_lang,
            file_path=str(out_path),
        )
        seg.synth_audio_path = out_path
        seg.synth_duration = _wav_duration(out_path)


# ------------------------------------------------------------------------ F5

def _synthesize_f5(segments, speaker_profiles, target_lang, out_dir, device):
    """Alternative backend. F5-TTS isn't pip-installable (it's a git repo
    per the spec's tech table), so we import lazily and give a clear error
    if it isn't present rather than making it a hard dependency for
    everyone using the default XTTS path.
    """
    try:
        from f5_tts.api import F5TTS  # type: ignore
    except ImportError as e:
        raise ImportError(
            "F5-TTS backend selected but not installed. Install with:\n"
            "  pip install git+https://github.com/SWivid/F5-TTS.git\n"
        ) from e

    model = F5TTS(device=device)
    logger.info(f"Synthesizing {len(segments)} segment(s) via F5-TTS...")
    for seg in segments:
        if not seg.translated_text.strip():
            continue
        profile = speaker_profiles.get(seg.speaker)
        if profile is None:
            continue
        out_path = out_dir / f"{seg.id}.wav"
        model.infer(
            ref_file=str(profile.reference_clip_path),
            ref_text="",  # F5 can auto-transcribe the reference if left blank
            gen_text=seg.translated_text,
            file_wave=str(out_path),
        )
        seg.synth_audio_path = out_path
        seg.synth_duration = _wav_duration(out_path)


def _wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
