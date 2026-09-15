# Run Guide — AI Voice Cloning & Dubbing Pipeline

This guide walks through getting the project running end-to-end, from a
clean machine to a dubbed output file. Follow it top to bottom on first
setup; jump to specific sections afterward.

---

## 0. Prerequisites

| Requirement | Why | Notes |
|---|---|---|
| Python 3.11 | Runtime | 3.10-3.12 likely fine; 3.11 tested |
| NVIDIA GPU + CUDA 12.1 | Whisper large-v3, XTTS-v2, NLLB all run far faster on GPU | CPU works but is slow (see §7) |
| ~10-15 GB disk | Model weight downloads (Whisper, pyannote, XTTS-v2, NLLB) | Cached after first run |
| Hugging Face account | Required for diarization (gated model) | Free |
| OpenAI API key | Only if using `--translation-backend gpt` | Optional |
| `ffmpeg`, `rubberband-cli`, `libsndfile1` | System-level audio/video tools | Installed via apt below |

---

## 1. Get the code

Unzip the project and `cd` into it:

```bash
unzip voice-dub.zip
cd voice-dub
```

## 2. System dependencies

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg rubberband-cli libsndfile1
```

Verify:
```bash
ffmpeg -version
rubberband -version
```

## 3. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs faster-whisper, pyannote.audio, transformers (for NLLB),
Coqui TTS (XTTS-v2), pyrubberband, resemblyzer, gradio, and supporting
libraries. Expect this step to take several minutes — `torch` and `TTS`
are large.

> **GPU users:** confirm PyTorch sees your GPU before proceeding:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```
> If this prints `False`, reinstall torch with the correct CUDA build from
> https://pytorch.org/get-started/locally/ before continuing.

## 4. Credentials

```bash
cp .env.example .env
```

Edit `.env`:

```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx   # optional, only for --translation-backend gpt
DEVICE=cuda
```

**Getting `HF_TOKEN`:**
1. Create a free account at https://huggingface.co
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click
   "Agree and access repository" (also accept the linked
   `pyannote/segmentation-3.0` license if prompted)
3. Go to https://huggingface.co/settings/tokens → create a token with
   **read** access
4. Paste it into `.env` as `HF_TOKEN`

Without this, `src/diarize.py` will raise a clear error at runtime — it
won't fail silently.

## 5. Sanity-check the install

Run the fast unit tests (no models, no GPU, no network — pure logic):

```bash
pytest
```

Expected: `9 passed, 1 deselected` in well under a second. If this fails,
something is wrong with the Python environment itself, before you even
touch models — fix this first.

## 6. Get a test clip

Drop a short (30s-2min to start) video or audio file somewhere accessible,
e.g. `examples/my_test.mp4`. A single clear speaker talking for 30-60
seconds is the easiest first test. See `examples/README.md` for suggested
sources (TED talks, Common Voice, self-recorded).

## 7. Run it

### Option A — CLI (fastest way to test the pipeline)

```bash
python -m src.pipeline examples/my_test.mp4 \
  --source-lang en \
  --target-lang es \
  --output outputs/dubbed.mp4
```

First run will download model weights (Whisper large-v3 ~3GB, pyannote
~50MB, XTTS-v2 ~2GB, NLLB ~2.5GB) — this can take 10-20+ minutes depending
on connection speed. Subsequent runs reuse the cache (`~/.cache/huggingface`,
`~/.local/share/tts`).

Useful flag combinations:

```bash
# Fast smoke test: skip diarization (single-speaker), audio-only output
python -m src.pipeline examples/my_test.mp4 \
  --no-diarize --no-remux --target-lang es

# CPU-only (slow, but works without a GPU)
python -m src.pipeline examples/my_test.mp4 \
  --device cpu --target-lang es

# Higher-quality translation via GPT-4o-mini (needs OPENAI_API_KEY)
python -m src.pipeline examples/my_test.mp4 \
  --translation-backend gpt --target-lang de

# Try to preserve background music/ambience (experimental duck-mix)
python -m src.pipeline examples/my_test.mp4 \
  --preserve-background --target-lang fr
```

Full flag reference is in the main `README.md`.

Watch the terminal — the pipeline logs progress per stage (`[extract]`,
`[diarize]`, `[transcribe]`, `[translate]`, `[synthesize]`, `[align]`,
`[reassemble]`, `[remux]`) and prints a timing summary at the end:

```
============================================================
Pipeline complete.
  Input duration:  45.2s
  Total time:      78.3s
  Realtime factor: 1.73x
  - extract                  0.4s
  - diarize                  6.1s
  - transcribe               8.9s
  - translate                2.1s
  - build_speaker_profiles   0.3s
  - synthesize               52.7s
  - align                    5.2s
  - reassemble                1.8s
  - remux                    0.8s
============================================================
```

### Option B — Gradio web demo

```bash
python app.py
```

Open the printed URL (default `http://localhost:7860`). Upload a file,
pick source/target languages, toggle diarization/backend options in
"Advanced options," and click **Run dubbing pipeline**. Output plays
inline with a timing breakdown below it.

### Option C — Docker (reproducible, isolated environment)

```bash
docker build -t voice-dub .

docker run --gpus all -p 7860:7860 \
  -v $(pwd)/.env:/app/.env \
  -v hf_cache:/root/.cache/huggingface \
  -v tts_cache:/root/.local/share/tts \
  -v $(pwd)/outputs:/app/outputs \
  voice-dub
```

This launches the Gradio app inside the container at `localhost:7860`.
The named volumes (`hf_cache`, `tts_cache`) persist downloaded model
weights across container restarts so you don't re-download on every run.

To run the CLI inside Docker instead of the Gradio app:
```bash
docker run --gpus all \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/examples:/app/examples \
  -v $(pwd)/outputs:/app/outputs \
  -v hf_cache:/root/.cache/huggingface \
  -v tts_cache:/root/.local/share/tts \
  voice-dub \
  python -m src.pipeline examples/my_test.mp4 --target-lang es
```

## 8. Where output goes

- Video input + `--no-remux` not set → dubbed video at `--output` path (default `outputs/dubbed.mp4`)
- Audio-only input, or `--no-remux` → dubbed audio `.wav` at the same path with `.wav` extension
- Intermediate files (extracted audio, per-speaker reference clips, per-segment synthesized/aligned clips) live under `--work-dir` (default `outputs/tmp/`) — useful for debugging a specific stage without re-running the whole pipeline

## 9. Running tests

```bash
pytest                    # fast unit tests only — default, no GPU/models needed
pytest -m integration     # full pipeline test — needs HF_TOKEN, GPU, model weights, and a fixture clip in examples/
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: HF_TOKEN not set` | `.env` missing or token not accepted for gated model | Re-check §4 — you must click "Agree and access repository" on the pyannote model page, not just create a token |
| `CUDA out of memory` | Whisper large-v3 + XTTS-v2 loaded together on a small GPU | Use `--whisper-model medium` or `--device cpu`; or run stages on separate GPUs by editing `PipelineConfig.device` per-call |
| pyrubberband errors / `OSError` | `rubberband-cli` not installed | Re-run §2; the pipeline auto-falls-back to ffmpeg `atempo` if rubberband is missing, so this is non-fatal but check logs |
| Silence in output for some segments | Diarization missed that speech, or translation returned empty string | Check `outputs/tmp/synth_segments/` — missing `.wav` files there confirm which segment dropped; see logged warnings from `transcribe.py`/`clone_tts.py` |
| `ffmpeg.Error` on remux | Input container has no video stream, or codec mismatch | Use `--no-remux` for audio-only output |
| Very slow on CPU | Expected — large-v3 Whisper + XTTS-v2 are GPU-oriented | Use `--whisper-model small` and expect >1x realtime latency; GPU strongly recommended for anything beyond a quick test |
| `ValueError: Language pair ... not in the NLLB code map` | Requested language not in `src/translate.py`'s `_NLLB_LANG_MAP` | Add the FLORES-200 code for that language to the map, or switch to `--translation-backend gpt` |

## 11. Next steps

Once you have a clip dubbing successfully end-to-end:
1. Run it on 5+ diverse clips (different languages, speaker counts, audio quality) per the spec's evaluation milestone
2. Fill in `docs/evaluation.md` with the metrics it already scaffolds (WER, speaker similarity via resemblyzer, timing drift, latency)
3. Record a demo GIF/video and link it in the main `README.md`
4. If deploying publicly, consider a Hugging Face Space (Gradio apps deploy there natively) rather than keeping the demo local-only
