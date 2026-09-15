# CUDA 12.1 base to match torch>=2.1 GPU wheels used in requirements.txt.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps:
#  - ffmpeg: audio/video extraction & remux
#  - rubberband-cli / libsndfile1: pyrubberband time-stretch backend
#  - git: pip installs from git (e.g. optional F5-TTS backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    ffmpeg rubberband-cli libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Model weights (Whisper, pyannote, XTTS-v2) are pulled on first run and
# cached in these volumes rather than baked into the image, to keep the
# image itself small and avoid embedding gated-model credentials at build time.
VOLUME ["/root/.cache/huggingface", "/root/.local/share/tts", "/app/outputs"]

EXPOSE 7860

CMD ["python", "app.py"]
