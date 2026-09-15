# Examples

Drop test clips here to use with `tests/test_pipeline.py::test_full_pipeline_single_speaker_clip`
and for the README demo GIF/video.

Suggested set (per spec §10 "Datasets / Test Material"):

| File | Source | Purpose |
|---|---|---|
| `sample_en_single_speaker.wav` | Self-recorded or royalty-free TED talk clip | Single-speaker smoke test |
| `sample_multi_speaker.wav` | Self-recorded conversation (you control ground truth) | Diarization accuracy testing |
| `sample_noisy.wav` | Common Voice (Mozilla) clip | STT robustness on noisy/accented audio |

These files are intentionally **not committed** to the repo (see `.gitignore`) —
audio/video fixtures bloat git history. Fetch or record your own, or host
them via Git LFS / a release asset if you want reproducible CI fixtures.

Outputs from running the pipeline on these clips (before/after audio, plus
a short write-up per clip) belong in `docs/evaluation.md`.
