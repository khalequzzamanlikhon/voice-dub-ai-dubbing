# voice-dub: AI Voice Cloning & Dubbing Pipeline

[![CI](https://github.com/khalequzzamanlikhon/voice-dub-ai-dubbing/actions/workflows/ci.yml/badge.svg)](https://github.com/khalequzzamanlikhon/voice-dub-ai-dubbing/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Takes a video or audio file in one language and produces a dubbed version **in the
original speaker's cloned voice**, time-aligned to the original, with optional
multi-speaker handling.

```
Input (video/audio)
  → [1] Audio extraction ffmpeg
  → [2] Speaker diarization pyannote.audio 3.1
  → [3] Transcription faster-whisper (word timestamps)
  → [4] Translation NLLB-200 (local) / GPT-4o-mini (API)
  → [5] Voice cloning + TTS XTTS-v2, per-speaker reference clips
  → [6] Time alignment pyrubberband / ffmpeg atempo, re-phrase fallback
  → [7] Reassembly pydub / ffmpeg
  → [8] Re-mux with video ffmpeg -map
Output (dubbed audio/video)
```

---

## Verified run (2026-09-15)

I ran the full pipeline end-to-end on real models on a Linux GPU server, and every
stage finished successfully. Outputs are in [`demo_outputs/`](demo_outputs/). The demo
clips are in [`docs/demo/`](docs/demo/).

**Environment:** Python 3.11 (conda), `torch 2.4.1+cu121`, `ctranslate2 4.5.0` (cuDNN 9),
`TTS 0.22.0` (XTTS-v2), rubberband CLI from conda-forge, NVIDIA RTX A5000.
**Settings:** Whisper `small`, NLLB-200 600M, XTTS-v2, **diarization off** (`--no-diarize`,
so no gated pyannote model was used).
**Input:** pyannote's `tutorials/assets/sample.wav`, a 30 s two-speaker phone
conversation, wrapped as a waveform MP4.

| Stage | What ran | Result |
|---|---|---|
| Tests | `pytest -v` | **9 passed**, 1 integration test deselected |
| Dub en → es | `DubbingPipeline`, 30.0 s input | yes `dubbed_es.mp4`, total **136.2 s** (realtime factor **4.54×**) |
| Dub en → de | same | yes `dubbed_de.mp4`, total **145.8 s** (realtime factor **4.86×**) |

 **Clips:** [original (en)](docs/demo/original_en.mp4) · [dubbed (es)](docs/demo/dubbed_es.mp4) · [dubbed (de)](docs/demo/dubbed_de.mp4)
 **Transcripts:** [en → es](docs/demo/transcript_en_to_es.md) · [en → de](docs/demo/transcript_en_to_de.md)

### Stage timings (seconds)

| Stage | en → es | en → de |
|---|---|---|
| extract | 0.16 | 0.13 |
| transcribe (Whisper small) | 8.16 | 8.10 |
| translate (NLLB-600M) | 8.76 | 9.30 |
| speaker profiles | 0.02 | 0.02 |
| **synthesize (XTTS-v2)** | **117.39** | **126.67** |
| align | 0.85 | 0.62 |
| reassemble + remux | 0.69 | 0.76 |
| **total** | **136.21** | **145.78** |

Voice synthesis is 86–87 % of runtime.

### Sample (en → es, first lines)

> **en:** Hello. Hello. Oh, hello. didn't know you were there. Neither did I. … This is Diane in New Jersey. And I'm Sheila in Texas, originally from Chicago.
>
> **es:** Hola. Hola. Oh, hola. No sabía que estuvieras allí. Yo tampoco. … Esta es Diane en Nueva Jersey. Y soy Sheila en Texas, originalmente de Chicago.

### Against the spec's success criteria

| Criterion | Target | This run | Status |
|---|---|---|---|
| Processing time | ≤ 3× realtime | 4.54× (es), 4.86× (de) | no not met, synthesis-bound |
| Time alignment | ±300 ms per segment | residual drift 766 ms (es), 6,854 ms (de) | no not met on this clip |
| Multi-speaker cloning | per-speaker voices | diarization off → both speakers → one `SPEAKER_00` voice | not exercised |
| Transcription | intelligible | mostly correct; "I heard it deep" is a mishear (Whisper `small`) | acceptable |
| Translation | idiomatic | fluent, literal phrasing ("lo oí profundamente") | acceptable |
| Voice similarity | cosine similarity | not measured | — |
| Demo UI | Gradio | not exercised in this run | — |

### Why the alignment missed, and what to fix

With diarization off, faster-whisper returned **one 30 s segment** for the whole
conversation. The translated speech came out longer: 41.0 s for Spanish (stretch ratio
1.37) and 49.2 s for German (1.64). Both exceed `max_stretch_ratio = 1.35`. Re-phrasing
only exists on the GPT translation backend, so `align.py` fell back to a clamped 1.35×
stretch and accepted the drift (`Alignment summary: 0 clean-stretched, 0 re-phrased, 1 accepted drift`).
The remux keeps the 30 s video length, so **the last ~6 s of the German audio is cut off**.

Fixes, in order of impact:
1. **Enable diarization** (`HF_TOKEN` + pyannote licence) so each speaker turn is its own
   short segment with its own voice.
2. **Split long Whisper segments** on sentence or word-timestamp boundaries before
   translation, so drift can't accumulate across 30 s.
3. Use `--translation-backend gpt` so over-long lines get re-phrased instead of clamped.
4. For speed: XTTS-v2 streaming/DeepSpeed inference, or `--tts-backend f5`, and
   Whisper `large-v3` only when accuracy matters more than time.

---

## Quickstart

### 1. Environment

```bash
git clone https://github.com/khalequzzamanlikhon/voice-dub-ai-dubbing.git
cd voice-dub-ai-dubbing
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install -r requirements.txt
pip install ctranslate2==4.5.0      # needed with torch 2.4.x (cuDNN 9); requirements pin 4.3.1 (cuDNN 8)
sudo apt-get install ffmpeg rubberband-cli libsndfile1
```

Version notes from the verified run (see `demo_outputs/run_scripts/constraints.txt`):
- `torch 2.4.1`: newer torch defaults `torch.load(weights_only=True)`, which breaks XTTS-v2 loading.
- `numpy<2`, `transformers<4.50`.
- `rubberband-cli` must be on `PATH`. `pyrubberband` raises `RuntimeError` without it, which the ffmpeg fallback doesn't catch.
- If ctranslate2 can't find cuDNN/cuBLAS, add `site-packages/nvidia/{cudnn,cublas}/lib` to `LD_LIBRARY_PATH`.

### 2. Credentials and licences

```bash
cp .env.example .env
export COQUI_TOS_AGREED=1   # accept the Coqui CPML licence for XTTS-v2 (non-commercial)
```

- `HF_TOKEN`: required for diarization. Accept the licence for
  [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) first.
- `OPENAI_API_KEY`: only for `--translation-backend gpt`.

### 3. Run

```bash
python -m src.pipeline path/to/input.mp4 --source-lang en --target-lang es --output outputs/dubbed.mp4
python app.py                       # Gradio UI at http://localhost:7860
docker build -t voice-dub . && docker run --gpus all -p 7860:7860 -v $(pwd)/.env:/app/.env voice-dub
```

Reproduce the verified run (writes `summary.json` + `transcript.md` next to the output):

```bash
python demo_outputs/run_scripts/dub_demo.py input.mp4 --target-lang es --no-diarize \
  --whisper-model small --out-dir demo_outputs/my_run
```

## CLI options

```
python -m src.pipeline INPUT [options]

  -o, --output PATH output file (default: outputs/dubbed.mp4)
  --source-lang CODE ISO 639-1, default: en
  --target-lang CODE ISO 639-1, default: es
  --no-diarize force single-speaker mode (skip diarization)
  --no-remux audio-only output, skip video remux
  --preserve-background duck-mix original track under dubbed speech
  --translation-backend {nllb,gpt}
  --tts-backend {xtts,f5}
  --whisper-model MODEL faster-whisper model size (default: large-v3)
  --device {cuda,cpu}
  --work-dir PATH intermediate-file scratch dir
```

## Key design decisions

- **Segment-length mismatch** (`src/align.py`): three tiers. First, a pitch-preserving
  stretch within [0.75, 1.35]. Then, on the GPT backend only, LLM re-phrasing to a target
  duration. Last, clamped stretch with bounded drift. Audio is not stretched past those
  bounds, which is why the run above shows drift rather than chipmunk audio.
- **Background audio** (`src/reassemble.py`): full replacement by default.
  `--preserve-background` duck-mixes the original track underneath. Demucs isolation is a stretch goal.
- **Overlapping speech** (`src/diarize.py`): flagged, excluded from voice reference clips,
  and still dubbed sequentially.

See [`docs/evaluation.md`](docs/evaluation.md) for the metric definitions and failure-case template.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                  # fast unit tests, no models/GPU (9 tests)
pytest -m integration   # full pipeline on real models (GPU + HF_TOKEN)
```

## License

MIT — see [LICENSE](LICENSE). Model weights carry their own licences: Whisper (MIT),
NLLB-200 (CC-BY-NC 4.0), XTTS-v2 (Coqui CPML, non-commercial), pyannote (gated, MIT).
