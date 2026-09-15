# Portfolio review — voice-dub

## Verdict

Yes, this is a good portfolio project — with one caveat: as found, it had
never actually been run end-to-end, and one module didn't even parse on the
Python version the docs claim to support. That gap between "looks complete"
and "actually works" is the single biggest risk for a portfolio piece,
because a recruiter or interviewer who clones it and tries `pytest` or the
CLI is exactly the audience you most need it to work for.

The architecture and writing are genuinely strong:
- The 8-stage pipeline (`src/extract.py` → `.../reassemble.py`) is cleanly
  decomposed, one concern per file, with a shared `Segment` contract in
  [src/types.py](src/types.py) so `pipeline.py` reads as an orchestrator, not
  a monolith.
- Docstrings explain *why*, not just *what* — the three-tier time-alignment
  strategy in [src/align.py](src/align.py) and the overlap-handling call in
  [src/diarize.py](src/diarize.py) both read like real design-review notes
  ("professional dubs re-write lines rather than distort audio"), which is
  exactly the signal that separates this from a tutorial copy.
- Tests are split into fast unit tests (pure logic, no models) and a marked
  `@pytest.mark.integration` test — the right shape for CI, and it already
  passes cleanly (`9 passed, 1 deselected`).
- [docs/evaluation.md](docs/evaluation.md) scaffolds real metrics (WER,
  speaker-embedding cosine similarity, timing drift, realtime factor)
  instead of hand-waving quality — this is the part most portfolio projects
  skip entirely.
- [RUN_GUIDE.md](RUN_GUIDE.md) is unusually thorough, including a
  troubleshooting table — this alone makes the project easier to actually
  evaluate than most.

## What I fixed directly (see the diff / new files)

1. **Critical: `src/translate.py` failed to parse on Python 3.11.** The GPT
   backend's prompt used an f-string with a backslash inside the `{...}`
   expression part — that's only legal from Python 3.12 (PEP 701). Since the
   README/RUN_GUIDE both say "Python 3.11 tested," this module would have
   raised `SyntaxError` on import for anyone following the stated setup.
   Verified: it now parses cleanly on 3.11 (confirmed with a local 3.11
   interpreter) and still passes on 3.12.
2. **No LICENSE despite the README saying "pick one."** Added MIT, matching
   the README's own recommendation.
3. **Broken image link.** `README.md` referenced `docs/demo.gif`, which
   doesn't exist — that renders as a broken image on GitHub. Replaced with a
   plain-text placeholder until a real demo is recorded.
4. **No CI despite a test suite built for it.** Added
   [.github/workflows/ci.yml](.github/workflows/ci.yml): installs
   `requirements-dev.txt` (lightweight — no torch/TTS/pyannote), runs
   `py_compile` across every module as a cheap syntax gate (this alone would
   have caught the bug above in an automated check instead of a manual
   review), then runs `pytest`.
5. **Added `requirements-dev.txt`** so CI (and anyone else) can run the fast
   test suite without a multi-GB ML install.

## What's still worth doing (not done here — needs a GPU + real clips)

These require actually running the pipeline, which this environment can't
do (no GPU, no HF-gated model access, no sample media) — they're the honest
next steps, not blockers to publishing:

1. **Run it on at least one real clip and fill in `docs/evaluation.md`.**
   Every metrics table is currently `<TODO>`. Right now the README's
   "Success criteria checklist" checks off things (≤3x realtime, ±300ms
   tolerance, >0.80 speaker similarity) that have never been measured on
   this codebase. Filling in even one row per table with real numbers turns
   this from "designed to hit these targets" into "measured to hit these
   targets" — a meaningfully stronger claim for an interview.
2. **Record the demo GIF/video** referenced in the README, once you have a
   run to show.
3. **Add 1-2 short sample clips** (or document exactly where you sourced
   them) so `pytest -m integration` and the Gradio demo are runnable by
   someone else without them going and finding their own test media.
4. **Be upfront that the default config never exercises align.py's tier 2.**
   The re-phrase tier only activates with `--translation-backend gpt`
   (NLLB has no re-prompting mechanism), but `nllb` is the default backend.
   This is already noted honestly in `docs/evaluation.md` §3.1 — consider
   surfacing it in the main README bullet too, since as written the
   README's "Key design decisions" section reads like all three tiers are
   always in play.
5. **Commit history.** The project had no git history at all when reviewed
   — everything as one initial state. A portfolio project benefits from a
   handful of real commits (even if you make them now, one per logical
   piece) since some reviewers do look at commit granularity as a signal of
   process, not just the end state. See `GITHUB_SETUP.md` for exact commands
   to get this initialized and pushed.

## Smaller polish items (optional)

- `requirements.txt` pins exact versions for everything, including `torch`
  without a CUDA-specific index URL — a fresh `pip install` will grab
  whatever default `torch` wheel PyPI resolves to, which may not match the
  CUDA 12.1 assumption baked into the Dockerfile. Consider either pointing
  at the PyTorch CUDA index in the README's pip command, or noting that
  CPU-only installs are expected to need a manual torch reinstall (the
  RUN_GUIDE already gestures at this with the `torch.cuda.is_available()`
  check — worth being a bit more explicit).
- No linter/formatter config (ruff/black). Not essential, but a `pyproject.toml`
  with ruff would be a five-minute addition if you want the repo to look
  actively maintained rather than just functional.
- `examples/README.md` is good about explaining why fixtures aren't
  committed, but as a result the integration test and Gradio demo are
  effectively untestable by anyone who clones the repo cold. Worth deciding
  if a tiny (a few seconds, royalty-free) clip is small enough to commit
  directly, sidestepping Git LFS entirely.

## Bottom line

Strong systems-design thinking and unusually good documentation for a solo
portfolio project — the kind of thing that reads well in an interview
*if* the person reading it can clone it and have `pytest` pass on the first
try, which it now can. The main remaining gap is proof of execution: run it
once, capture real numbers and a demo, and this becomes a project you can
talk through end-to-end rather than one you have to caveat.
