# Evaluation

This document reports pipeline quality on the test set in `examples/`, per
spec §7 (metrics to report) and §8 (known hard problems). Fill in the
`<TODO>` fields after running the pipeline on your own clips — the
structure and metric definitions below are ready to use as-is.

## 1. Test set

| Clip | Duration | Languages | Speakers | Audio quality |
|---|---|---|---|---|
| `sample_en_single_speaker.wav` | <TODO> | en → es | 1 | clean |
| `sample_multi_speaker.wav` | <TODO> | en → de | 2-3 | clean |
| `sample_noisy.wav` | <TODO> | en → fr | 1 | noisy/accented |
| ... | | | | |

Aim for 5+ clips spanning different target languages, speaker counts, and
audio conditions (spec §5 M8).

## 2. Metrics

### 2.1 Transcription — Word Error Rate
Qualitative spot-check by default (no labeled reference transcripts for
self-recorded clips); WER computed where ground-truth captions exist
(e.g. TED talk clips with official transcripts):

```
WER = (substitutions + deletions + insertions) / reference_word_count
```

| Clip | WER | Notes |
|---|---|---|
| <TODO> | <TODO> | |

### 2.2 Translation quality — qualitative
Per spec §7, BLEU is not meaningful here (single-reference, spoken-register
text, small samples) and is intentionally skipped. Instead: idiomatic
correctness spot-check, rated 1-5 by a fluent speaker of the target
language, with notes on specific mistranslations.

| Clip | Rating (1-5) | Notes |
|---|---|---|
| <TODO> | <TODO> | |

### 2.3 Voice similarity — speaker embedding cosine similarity
Computed with `resemblyzer` embeddings between each speaker's reference
clip and their synthesized (pre-stretch) output:

```python
from resemblyzer import VoiceEncoder, preprocess_wav
encoder = VoiceEncoder()
ref_embed = encoder.embed_utterance(preprocess_wav("reference.wav"))
synth_embed = encoder.embed_utterance(preprocess_wav("synth.wav"))
similarity = ref_embed @ synth_embed  # cosine similarity, embeddings are unit-norm
```

| Clip | Speaker | Cosine similarity | Notes |
|---|---|---|---|
| <TODO> | <TODO> | <TODO> | |

Rule of thumb from XTTS-v2 community benchmarks: >0.80 reads as
"recognizably the same speaker" to most listeners; 0.65-0.80 is
borderline; <0.65 usually sounds like a different person.

### 2.4 Timing drift
Average absolute difference between original segment duration and final
(post-alignment) dubbed segment duration, in ms — logged automatically by
`align.py` and summarized at pipeline end.

| Clip | Avg drift (ms) | % segments over 300ms tolerance | Notes |
|---|---|---|---|
| <TODO> | <TODO> | <TODO> | |

### 2.5 Latency
Per-stage and total wall-clock time, reported by `pipeline.py`'s summary
log (`pipeline.timings`), normalized to seconds of processing per minute
of input audio.

| Clip | Input duration | Total time | Realtime factor | GPU |
|---|---|---|---|---|
| <TODO> | <TODO> | <TODO> | <TODO> | <TODO> |

## 3. Known limitations & failure cases (spec §8)

Explicitly addressed in this implementation:

### 3.1 Language pairs with divergent sentence length (e.g., en → de)
**Symptom:** German translations routinely run 15-30% longer than English
source audio for the same content, which without mitigation would force
heavy time-stretching and audibly unnatural pacing.
**Mitigation implemented:** `align.py`'s three-tier strategy — stretch
within ±35%/-25% bounds, then (GPT backend only) re-phrase for a target
syllable count, then accept bounded drift as a last resort. NLLB-backend
runs have no re-phrase tier and will show more drift on this language
pair — <TODO: quantify from test set>.

### 3.2 Cloning quality degrading on noisy/short reference audio
**Symptom:** <TODO: describe what you observed — e.g. artifacts, flat
prosody, or the model rendering background noise as source-speaker
"vocal texture">.
**Mitigation implemented:** `clone_tts.build_speaker_profiles` selects the
*longest* non-overlapping turn per speaker (up to 15s) and logs a warning
when the best available reference is under the 6s XTTS-v2 minimum. No
denoising is applied — a documented gap, not a fix; see §4.

### 3.3 Diarization errors cascading into wrong-voice assignment
**Symptom:** A single misattributed diarization turn causes that line to
be cloned in the wrong speaker's voice, which is more noticeable/jarring
than a transcription error.
**Mitigation implemented:** overlap flagging (`diarize._flag_overlaps`)
excludes ambiguous turns from reference-clip selection so at least the
*reference* voice isn't contaminated by a diarization mistake; the
downstream mis-assigned utterance itself is not auto-corrected. <TODO:
report observed diarization error rate on `sample_multi_speaker.wav`
where you control ground truth.>

### 3.4 Emotional/prosodic transfer
**Symptom:** Cloned voice is timbrally correct but delivery is flatter
than the source (less pitch variation, weaker emphasis) — a known
XTTS-v2 zero-shot limitation, more visible on high-emotion source clips.
**Not mitigated in this repo** — noted as a stretch-goal-adjacent problem;
fine-tuning XTTS on a target voice (spec §9) would help but is out of
scope for zero-shot cloning.

## 4. What I'd fix with more time

- Denoising (e.g. a lightweight RNNoise/DeepFilterNet pass) on reference
  clips before cloning, to decouple "noisy source" from "noisy clone."
- Demucs-based source separation for real background-music preservation,
  replacing the current duck-mix approximation (`reassemble.py`,
  `preserve_background=True`).
- A lightweight diarization-confidence check (e.g. re-scoring turns against
  speaker embeddings post-hoc) to catch and flag likely misattributions
  before they propagate to voice cloning.
- Wav2Lip-based lip-sync for the video remux path (spec §9 stretch goal).
- Batch/streaming variants of the STT and translation stages to cut
  latency on long inputs — current pipeline is fully sequential.
