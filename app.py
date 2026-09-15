"""Gradio demo for the voice cloning & dubbing pipeline (spec §5, M7).

Run with: python app.py
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import gradio as gr
from loguru import logger

from src.pipeline import DubbingPipeline
from src.types import PipelineConfig

LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Chinese (Simplified)": "zh",
    "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Arabic": "ar",
    "Hindi": "hi", "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
}


def run_dubbing(
    media_file,
    source_lang_name: str,
    target_lang_name: str,
    diarize_enabled: bool,
    preserve_background: bool,
    translation_backend: str,
    device: str,
    progress=gr.Progress(track_tqdm=True),
):
    if media_file is None:
        raise gr.Error("Please upload a video or audio file.")

    work_dir = Path(tempfile.mkdtemp(prefix="voicedub_"))
    output_path = work_dir / "dubbed_output.mp4"

    config = PipelineConfig(
        input_path=Path(media_file),
        output_path=output_path,
        source_lang=LANGUAGES[source_lang_name],
        target_lang=LANGUAGES[target_lang_name],
        diarize=diarize_enabled,
        preserve_background=preserve_background,
        translation_backend=translation_backend,
        device=device,
        work_dir=work_dir / "tmp",
    )

    try:
        progress(0.05, desc="Starting pipeline...")
        pipeline = DubbingPipeline(config)
        result_path = pipeline.run()

        summary_lines = [f"**Realtime factor:** {pipeline.timings['total']:.1f}s total"]
        for stage, t in pipeline.timings.items():
            if stage != "total":
                summary_lines.append(f"- {stage}: {t:.1f}s")
        summary = "\n".join(summary_lines)

        return str(result_path), summary
    except Exception as e:
        logger.error(traceback.format_exc())
        raise gr.Error(f"Pipeline failed: {e}")


with gr.Blocks(title="AI Voice Cloning & Dubbing") as demo:
    gr.Markdown(
        """
        # 🎙️ AI Voice Cloning & Dubbing Pipeline
        Upload a video or audio clip, pick source/target languages, and get back
        a dubbed version **in the original speaker's own cloned voice**, time-aligned
        to the original.
        """
    )

    with gr.Row():
        with gr.Column():
            media_input = gr.File(label="Input video/audio", file_types=["video", "audio"])
            with gr.Row():
                source_lang = gr.Dropdown(list(LANGUAGES.keys()), value="English", label="Source language")
                target_lang = gr.Dropdown(list(LANGUAGES.keys()), value="Spanish", label="Target language")
            with gr.Accordion("Advanced options", open=False):
                diarize_toggle = gr.Checkbox(value=True, label="Multi-speaker diarization")
                bg_toggle = gr.Checkbox(value=False, label="Preserve background audio (experimental)")
                backend = gr.Radio(["nllb", "gpt"], value="nllb", label="Translation backend")
                device = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")
            run_btn = gr.Button("Run dubbing pipeline", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Dubbed output")
            timing_report = gr.Markdown(label="Timing breakdown")

    run_btn.click(
        fn=run_dubbing,
        inputs=[media_input, source_lang, target_lang, diarize_toggle, bg_toggle, backend, device],
        outputs=[output_video, timing_report],
    )

    gr.Markdown(
        """
        ---
        **Notes:** First run downloads model weights (Whisper, pyannote, XTTS-v2) —
        expect a multi-GB download and a slow first call. Requires `HF_TOKEN` set
        (see `.env.example`) for diarization. See `docs/evaluation.md` for known
        limitations and failure cases.
        """
    )

if __name__ == "__main__":
    demo.queue().launch()
