"""[Orchestrator] Wires stages [1]-[8] together per the spec architecture
diagram. Also exposes a CLI (`python -m src.pipeline ...`) for headless
runs, used by both `app.py` and `tests/test_pipeline.py`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src import align, clone_tts, diarize, extract, reassemble, transcribe, translate
from src.types import PipelineConfig, Segment

load_dotenv()


class DubbingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.segments: list[Segment] = []
        self.timings: dict[str, float] = {}

    def run(self) -> Path:
        t_total = time.time()
        cfg = self.config

        audio_path = self._timed("extract", extract.extract_audio, cfg.input_path, cfg.work_dir)
        duration = extract.probe_duration(audio_path)

        if cfg.diarize:
            raw_segments = self._timed(
                "diarize", diarize.diarize, audio_path, cfg.device, cfg.min_speakers, cfg.max_speakers
            )
        else:
            raw_segments = diarize.single_speaker_fallback(duration)

        self.segments = self._timed(
            "transcribe", transcribe.transcribe, audio_path, raw_segments,
            cfg.whisper_model, cfg.device, cfg.whisper_compute_type, cfg.source_lang,
        )

        self._timed(
            "translate", translate.translate_segments, self.segments,
            cfg.source_lang, cfg.target_lang, cfg.translation_backend,
            cfg.nllb_model, cfg.gpt_model, cfg.device,
        )

        profiles = self._timed(
            "build_speaker_profiles", clone_tts.build_speaker_profiles,
            audio_path, self.segments, cfg.work_dir, cfg.min_reference_seconds,
        )

        self._timed(
            "synthesize", clone_tts.synthesize_segments, self.segments, profiles,
            cfg.target_lang, cfg.work_dir, cfg.tts_backend, cfg.xtts_model, cfg.device,
        )

        retranslate_fn, resynthesize_fn = self._build_rephrase_hooks(profiles)
        self._timed(
            "align", align.align_segments, self.segments, cfg, retranslate_fn, resynthesize_fn,
        )

        dubbed_audio = self._timed(
            "reassemble", reassemble.reassemble, self.segments, audio_path, duration,
            cfg.work_dir, cfg.preserve_background,
        )

        final_output = dubbed_audio
        if cfg.remux_video and extract.has_video_stream(cfg.input_path):
            final_output = self._timed(
                "remux", reassemble.remux_with_video, dubbed_audio, cfg.input_path, cfg.output_path,
            )
        else:
            final_output = cfg.output_path.with_suffix(".wav")
            final_output.parent.mkdir(parents=True, exist_ok=True)
            Path(dubbed_audio).replace(final_output)

        self.timings["total"] = time.time() - t_total
        self._log_summary(duration)
        return final_output

    # ------------------------------------------------------------ helpers

    def _timed(self, name, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        self.timings[name] = time.time() - t0
        logger.info(f"[{name}] done in {self.timings[name]:.1f}s")
        return result

    def _build_rephrase_hooks(self, profiles):
        """Only wire up the re-phrase-on-overflow tier (align.py tier 2)
        when using the GPT translation backend, since NLLB has no
        instruction-following re-prompt mechanism."""
        if self.config.translation_backend != "gpt":
            return None, None

        def retranslate_fn(seg: Segment, target_seconds: float) -> str:
            import os
            from openai import OpenAI
            client = OpenAI()
            approx_wpm = 150  # rough spoken words-per-minute for pacing guidance
            target_words = max(1, int(target_seconds / 60 * approx_wpm))
            prompt = (
                f"Re-translate this line into {self.config.target_lang}, but make it "
                f"concise enough to be comfortably spoken in about {target_seconds:.1f} seconds "
                f"(~{target_words} words), while preserving meaning. Only output the translation.\n\n"
                f"Original ({self.config.source_lang}): {seg.text}\n"
                f"Previous translation: {seg.translated_text}"
            )
            resp = client.chat.completions.create(
                model=self.config.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()

        def resynthesize_fn(seg: Segment) -> Path:
            clone_tts.synthesize_segments(
                [seg], profiles, self.config.target_lang, self.config.work_dir,
                self.config.tts_backend, self.config.xtts_model, self.config.device,
            )
            return seg.synth_audio_path

        return retranslate_fn, resynthesize_fn

    def _log_summary(self, input_duration: float):
        logger.info("=" * 60)
        logger.info("Pipeline complete.")
        logger.info(f"  Input duration:  {input_duration:.1f}s")
        logger.info(f"  Total time:      {self.timings['total']:.1f}s")
        logger.info(f"  Realtime factor: {self.timings['total'] / max(input_duration, 1e-6):.2f}x")
        for stage, t in self.timings.items():
            if stage != "total":
                logger.info(f"  - {stage:<24s} {t:6.1f}s")
        logger.info("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Voice Cloning & Dubbing Pipeline")
    p.add_argument("input", type=Path, help="Input video/audio file")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/dubbed.mp4"))
    p.add_argument("--source-lang", default="en")
    p.add_argument("--target-lang", default="es")
    p.add_argument("--no-diarize", action="store_true", help="Force single-speaker mode")
    p.add_argument("--no-remux", action="store_true", help="Audio-only output, skip video remux")
    p.add_argument("--preserve-background", action="store_true")
    p.add_argument("--translation-backend", choices=["nllb", "gpt"], default="nllb")
    p.add_argument("--tts-backend", choices=["xtts", "f5"], default="xtts")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--work-dir", type=Path, default=Path("outputs/tmp"))
    return p


def main():
    args = build_arg_parser().parse_args()
    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        diarize=not args.no_diarize,
        remux_video=not args.no_remux,
        preserve_background=args.preserve_background,
        translation_backend=args.translation_backend,
        tts_backend=args.tts_backend,
        whisper_model=args.whisper_model,
        device=args.device,
        work_dir=args.work_dir,
    )
    pipeline = DubbingPipeline(config)
    output = pipeline.run()
    logger.info(f"Output written to: {output}")


if __name__ == "__main__":
    main()
