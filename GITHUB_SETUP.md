# Pushing this project to GitHub

This project isn't a git repository yet. This is a copy-pasteable guide to
get it into git locally and up on GitHub as `khalequzzamanlikhon/voice-dub`
(swap the repo name in the commands below if you want a different one).

## 0. Before you start

Double-check nothing sensitive is about to be committed:

```bash
cat .env 2>/dev/null   # should not exist yet, or should be empty/untracked
```

`.gitignore` already excludes `.env`, `outputs/`, model caches, and media
files — leave it as-is so you never accidentally commit an API key or a
multi-GB model checkpoint.

## 1. Initialize git and make your first commits

A few small, logical commits reads better than one giant "initial commit"
to anyone browsing your history:

```bash
git init
git add src/ tests/ app.py requirements.txt requirements-dev.txt pytest.ini \
        Dockerfile .env.example .gitignore
git commit -m "Core dubbing pipeline: extract, diarize, transcribe, translate, clone, align, reassemble"

git add README.md RUN_GUIDE.md docs/ examples/README.md LICENSE PORTFOLIO_REVIEW.md
git commit -m "Add docs, evaluation scaffold, and license"

git add .github/
git commit -m "Add CI: syntax check + fast unit tests on push/PR"
```

(If you'd rather have one commit, that's fine too — `git add -A && git commit -m "Initial commit"` works just as well. The split above is only a suggestion.)

## 2. Create the GitHub repo

Pick one:

**Option A — GitHub CLI** (fastest, if you have `gh` installed and logged in):
```bash
gh auth login          # skip if already logged in
gh repo create voice-dub --public --source=. --remote=origin --push
```
That single `gh repo create` command creates the repo, adds the remote, and
pushes in one step — skip straight to step 4 if you use it.

**Option B — GitHub website:**
1. Go to https://github.com/new
2. Repository name: `voice-dub`
3. Leave "Initialize this repository with a README" **unchecked** (you
   already have one — checking it creates a conflicting commit)
4. Create repository

## 3. Connect your local repo to GitHub (skip if you used `gh repo create`)

GitHub will show you these commands after creating the repo — they'll look
like:

```bash
git remote add origin https://github.com/khalequzzamanlikhon/voice-dub.git
git branch -M main
git push -u origin main
```

## 4. Verify

```bash
git remote -v      # confirm origin points at your GitHub repo
git log --oneline  # confirm your commits are there
```

Then open `https://github.com/khalequzzamanlikhon/voice-dub` in a browser —
you should see the README rendered, the CI badge (it'll show "no status" or
run automatically once Actions kicks in on the push), and all files present.

## 5. After the first push

- Check the **Actions** tab on GitHub — the CI workflow
  ([.github/workflows/ci.yml](.github/workflows/ci.yml)) should run
  automatically and pass (it only needs `requirements-dev.txt`, not the full
  GPU stack, so it finishes in seconds).
- Consider adding a short repo description and topics (e.g. `python`,
  `speech-to-speech`, `voice-cloning`, `dubbing`, `whisper`, `xtts`) on the
  GitHub repo page — this is what shows up in search and on your profile.
- If you later record the demo GIF, drop it at `docs/demo.gif` and restore
  the `![demo](docs/demo.gif)` line in `README.md` (currently a text
  placeholder — see `PORTFOLIO_REVIEW.md`).
- Pin the repo on your GitHub profile (Profile → Customize your pins) so
  it's visible to anyone viewing your profile, not just people who find it
  via search.
