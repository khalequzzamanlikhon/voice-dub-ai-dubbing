"""Run voice_dub's DubbingPipeline on one input and save README assets: the original, the
dubbed output, and summary.json / transcript.md with per-segment source vs. translated text,
stage timings and realtime factor. Run from the voice_dub repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src import extract  # noqa: E402
from src.pipeline import DubbingPipeline  # noqa: E402
from src.types import PipelineConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-lang", default="en")
    ap.add_argument("--target-lang", default="es")
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--whisper-model", default="small")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        input_path=args.input,
        output_path=out / f"dubbed_{args.target_lang}.mp4",
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        diarize=not args.no_diarize,
        whisper_model=args.whisper_model,
        device="cuda",
        work_dir=out / "work",
    )
    pipeline = DubbingPipeline(config)
    result = pipeline.run()

    shutil.copy(args.input, out / f"original{args.input.suffix}")
    duration = extract.probe_duration(args.input)
    segments = [
        {
            "id": s.id,
            "speaker": s.speaker,
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text,
            "translated_text": s.translated_text,
            "synth_duration": round(s.synth_duration, 2),
            "target_duration": round(s.target_duration, 2),
            "aligned": s.aligned_audio_path is not None,
            "overlap": s.overlap,
        }
        for s in pipeline.segments
    ]
    summary = {
        "input": args.input.name,
        "output": Path(result).name,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "diarize": not args.no_diarize,
        "whisper_model": args.whisper_model,
        "input_duration_s": round(duration, 2),
        "timings_s": {k: round(v, 2) for k, v in pipeline.timings.items()},
        "realtime_factor": round(pipeline.timings["total"] / max(duration, 1e-6), 2),
        "speakers": sorted({s["speaker"] for s in segments}),
        "segments": segments,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {args.input.name}: {args.source_lang} -> {args.target_lang}",
        "",
        f"Input {duration:.1f}s, total {pipeline.timings['total']:.1f}s "
        f"(realtime factor {summary['realtime_factor']}x), diarization {'on' if config.diarize else 'off'}.",
        "",
        "| # | speaker | time | original | translated |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(segments):
        lines.append(f"| {i} | {s['speaker']} | {s['start']:.1f}-{s['end']:.1f}s | {s['text']} | {s['translated_text']} |")
    (out / "transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
