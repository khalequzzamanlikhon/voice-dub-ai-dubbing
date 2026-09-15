"""[3] STT / Transcription

Uses faster-whisper (CTranslate2 backend) for word-level timestamped
transcription. We transcribe the *whole* track once (fast, and gives
Whisper full context for better accuracy) and then assign each whisper
segment/word to the diarized speaker turn it falls inside, rather than
running whisper separately per speaker turn (which fragments context and
is much slower for long files).
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.types import Segment, Word

_model_cache: dict[str, object] = {}


def _load_model(model_size: str, device: str, compute_type: str):
    key = f"{model_size}:{device}:{compute_type}"
    if key in _model_cache:
        return _model_cache[key]

    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_cache[key] = model
    return model


def transcribe(
    audio_path: Path,
    diarized_segments: list[Segment],
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
) -> list[Segment]:
    """Transcribe `audio_path` and distribute words into `diarized_segments`
    by timestamp overlap. Segments that end up with no words (e.g. pure
    non-speech turns misfired by diarization) are dropped.

    Returns the same list of Segment objects, mutated in place, filtered
    to only those with text.
    """
    model = _load_model(model_size, device, compute_type)

    logger.info(f"Transcribing {audio_path}...")
    whisper_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,               # skip silence, reduces hallucination
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    all_words: list[Word] = []
    for seg in whisper_segments:
        for w in (seg.words or []):
            all_words.append(Word(text=w.word.strip(), start=w.start, end=w.end,
                                   probability=w.probability))

    logger.info(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    logger.info(f"Transcribed {len(all_words)} words.")

    _assign_words_to_segments(all_words, diarized_segments)

    filled = [s for s in diarized_segments if s.words]
    for s in filled:
        s.text = _words_to_text(s.words)

    dropped = len(diarized_segments) - len(filled)
    if dropped:
        logger.warning(f"Dropped {dropped} diarized turn(s) with no aligned words.")

    return filled


def _assign_words_to_segments(words: list[Word], segments: list[Segment]) -> None:
    """Greedy assignment: each word goes to the segment with the greatest
    temporal overlap with the word's [start, end) window. O(n_words *
    n_segments) — fine at these scales (minutes of audio -> low thousands
    of words / tens-to-hundreds of segments)."""
    segments = sorted(segments, key=lambda s: s.start)
    for w in words:
        best_seg, best_overlap = None, 0.0
        for seg in segments:
            if seg.end < w.start:
                continue
            if seg.start > w.end:
                break
            overlap = min(seg.end, w.end) - max(seg.start, w.start)
            if overlap > best_overlap:
                best_overlap, best_seg = overlap, seg
        if best_seg is not None:
            best_seg.words.append(w)
        # words that fall entirely outside any diarized turn (diarization
        # miss) are dropped — logged in aggregate by the caller via count diff


def _words_to_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words).strip()
