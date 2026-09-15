#!/usr/bin/env bash
# Sets up and runs all four projects on the GPU server, saving README demo assets.
#
# Usage (inside tmux):
#   bash run_all.sh                      # every stage, in order
#   bash run_all.sh setup                # only the conda envs + pip installs
#   bash run_all.sh samples depth video  # any subset of: setup samples depth video voice phone report
#
# Every sub-step logs to demo_outputs/logs/<step>.log and appends OK/FAIL to
# demo_outputs/STATUS.tsv. A failing step never aborts the rest of the run.
#
# Env knobs:
#   CUDA_VISIBLE_DEVICES   default 1 (GPU 0 is shared with other jobs)
#   COQUI_TOS_AGREED=1     required for voice_dub (XTTS-v2 is under the Coqui CPML licence)
#   PHONE_OLLAMA_MODEL     default llama3.1:latest
set -uo pipefail

ROOT=/home/likhon/.slk/.prjt
RUN="$ROOT/run_all"
ENVS="$ROOT/envs"
SAMPLES="$ROOT/samples"
OUT="$ROOT/demo_outputs"
LOGS="$OUT/logs"
STATUS="$OUT/STATUS.tsv"
CONDA="$HOME/miniconda3/bin/conda"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the project envs
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8  # tmux sessions here default to an ASCII locale
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
PHONE_OLLAMA_MODEL="${PHONE_OLLAMA_MODEL:-llama3.1:latest}"
VOICE_WHISPER_MODEL="${VOICE_WHISPER_MODEL:-small}"
TORCH_PINS="torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"

mkdir -p "$ENVS" "$SAMPLES" "$OUT" "$LOGS"
[ -f "$STATUS" ] || printf 'time\tstage\tresult\tseconds\tlog\n' > "$STATUS"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

# run_stage NAME CMD... -> output to logs/NAME.log, result to STATUS.tsv, never aborts the run.
run_stage() {
  local name=$1; shift
  local logf="$LOGS/$name.log" t0=$SECONDS rc res
  log "START $name  (log: $logf)"
  "$@" >"$logf" 2>&1; rc=$?
  res=OK; [ $rc -eq 0 ] || res="FAIL(rc=$rc)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$res" "$((SECONDS - t0))" "$logf" >> "$STATUS"
  log "END   $name -> $res in $((SECONDS - t0))s"
  [ $rc -eq 0 ] || tail -n 20 "$logf"
  return 0
}

py() { local env=$1; shift; "$ENVS/$env/bin/python" "$@"; }

wait_http() { # URL SECONDS
  local i; for i in $(seq 1 "$2"); do curl -sf -o /dev/null "$1" && return 0; sleep 1; done
  echo "timed out waiting for $1"; return 1
}

# ------------------------------------------------------------------ setup

write_constraints() {
  cat > "$RUN/constraints.txt" <<'EOF'
# torch 2.4.1: last line before torch.load(weights_only=True) became default (breaks XTTS-v2 loading)
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
numpy<2
transformers<4.50
EOF
}

create_env() { # NAME [extra conda-forge packages...]
  local name=$1; shift
  if [ -x "$ENVS/$name/bin/python" ]; then echo "env $name already exists"; return 0; fi
  "$CONDA" create -y -p "$ENVS/$name" --override-channels -c conda-forge python=3.11 pip "$@"
}

setup_depth() ( set -e
  create_env depth_craft
  cd "$ROOT/depth_craft"
  py depth_craft -m pip install -c "$RUN/constraints.txt" $TORCH_PINS
  py depth_craft -m pip install -c "$RUN/constraints.txt" -e ".[dev,depth]"
  py depth_craft -c "import torch, transformers, open3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| transformers', transformers.__version__, '| open3d', open3d.__version__)"
)

setup_video() ( set -e
  create_env video-anomaly
  cd "$ROOT/video-anomaly"
  py video-anomaly -m pip install -c "$RUN/constraints.txt" $TORCH_PINS
  py video-anomaly -m pip install -c "$RUN/constraints.txt" -e ".[dev]" psutil
  py video-anomaly -c "import torch, ultralytics, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), '| ultralytics', ultralytics.__version__, '| cv2', cv2.__version__)"
)

setup_voice() ( set -e
  # rubberband-cli isn't installed system-wide (no sudo); pyrubberband raises RuntimeError without it,
  # which align.py's ImportError/OSError fallback doesn't catch -> take the CLI from conda-forge.
  create_env voice_dub rubberband
  cd "$ROOT/voice_dub"
  py voice_dub -m pip install -c "$RUN/constraints.txt" $TORCH_PINS
  py voice_dub -m pip install -c "$RUN/constraints.txt" -r requirements.txt
  # requirements pin ctranslate2 4.3.1 (cuDNN 8); torch 2.4.1 ships cuDNN 9 -> 4.5.0 is the first cuDNN 9 build.
  py voice_dub -m pip install -c "$RUN/constraints.txt" "ctranslate2==4.5.0" pytest
  "$ENVS/voice_dub/bin/rubberband" --version || echo "WARN: rubberband CLI missing from env"
  voice_env
  py voice_dub -c "import torch, ctranslate2, TTS, pyannote.audio, faster_whisper; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| ct2 cuda devices', ctranslate2.get_cuda_device_count(), '| TTS', TTS.__version__)"
)

setup_phone() ( set -e
  create_env phone-support-agent
  cd "$ROOT/phone-support-agent"
  py phone-support-agent -m pip install -c "$RUN/constraints.txt" -r requirements.txt
)

# ------------------------------------------------------------------ samples

fetch() { # URL DEST
  if [ -s "$2" ]; then echo "have $2"; return 0; fi
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" && ls -l "$2"
}

stage_samples() ( set -e
  mkdir -p "$SAMPLES/images"
  fetch https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi "$SAMPLES/vtest.avi"
  fetch https://raw.githubusercontent.com/pyannote/pyannote-audio/develop/tutorials/assets/sample.wav "$SAMPLES/two_speakers.wav"
  for i in 01 02 03 04 05 06; do
    fetch "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/assets/examples/demo$i.jpg" "$SAMPLES/images/demo$i.jpg"
  done
  for f in "$SAMPLES/vtest.avi" "$SAMPLES/two_speakers.wav"; do
    ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,sample_rate,channels -of compact "$f"
  done
)

# ------------------------------------------------------------------ depth_craft

depth_tests() ( cd "$ROOT/depth_craft" && py depth_craft -m pytest tests/ -v )

depth_mvp_repo() ( set -e  # the repo's own mvp_demo.py, unmodified
  cd "$ROOT/depth_craft"
  img=$(ls "$SAMPLES"/images/*.jpg | head -1)
  py depth_craft "$RUN/depth_craft/patched_run.py" mvp_demo.py --image "$img" --headless --output-dir "$OUT/depth_craft/mvp_demo_repo"
  grep -q "backend: transformers" "$LOGS/depth_mvp_repo.log" || { echo "mvp_demo used the fallback estimator"; exit 3; }
)

depth_demo() ( cd "$ROOT/depth_craft" && py depth_craft "$RUN/depth_craft/depth_demo.py" --images "$SAMPLES/images" --out "$OUT/depth_craft" )

depth_bench() ( set -e
  cd "$ROOT/depth_craft"
  img=$(ls "$SAMPLES"/images/*.jpg | head -1)
  py depth_craft "$RUN/depth_craft/patched_run.py" scripts/benchmark_inference.py --image "$img" --n-runs 30 | tee "$OUT/depth_craft/benchmark_inference.txt"
)

depth_api() ( set -e
  cd "$ROOT/depth_craft"
  d="$OUT/depth_craft/api"; port=8765; mkdir -p "$d"
  img=$(ls "$SAMPLES"/images/*.jpg | head -1)
  py depth_craft "$RUN/depth_craft/patched_run.py" -m uvicorn depthcraft.api.main:app --host 127.0.0.1 --port $port > "$d/uvicorn.log" 2>&1 &
  pid=$!; trap 'kill $pid 2>/dev/null' EXIT
  wait_http "http://127.0.0.1:$port/api/v1/health" 180
  curl -sf "http://127.0.0.1:$port/api/v1/health" | tee "$d/health_before.json"; echo
  curl -sf -X POST "http://127.0.0.1:$port/api/v1/depth?with_uncertainty=true" -F "file=@$img" | tee "$d/depth.json"; echo
  job=$(py depth_craft -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' < "$d/depth.json")
  curl -sf -o "$d/depth_vis.png" "http://127.0.0.1:$port/outputs/$job/depth_vis.png"
  curl -sf -o "$d/uncertainty_vis.png" "http://127.0.0.1:$port/outputs/$job/uncertainty_vis.png"
  curl -sf -X POST "http://127.0.0.1:$port/api/v1/reconstruct" -F "file=@$img" | tee "$d/reconstruct.json"; echo
  curl -sf "http://127.0.0.1:$port/api/v1/health" | tee "$d/health_after.json"; echo
  grep -q '"backend":"transformers"' "$d/depth.json" || { echo "API used the fallback estimator"; exit 3; }
)

# ------------------------------------------------------------------ video-anomaly

video_tests() ( cd "$ROOT/video-anomaly" && py video-anomaly -m pytest -v )

video_render() ( set -e
  cd "$ROOT/video-anomaly"
  py video-anomaly "$RUN/video-anomaly/render_demo.py" --source "$SAMPLES/vtest.avi" --out "$OUT/video-anomaly/vtest_demo_profile" --profile demo
  py video-anomaly "$RUN/video-anomaly/render_demo.py" --source "$SAMPLES/vtest.avi" --out "$OUT/video-anomaly/vtest_default_profile" --profile default --gif-seconds 0
)

video_bench() ( set -e
  cd "$ROOT/video-anomaly"
  for w in yolov8n.pt; do  # nano only: smallest weights (~6 MB)
    py video-anomaly scripts/benchmark_fps.py --source "$SAMPLES/vtest.avi" --weights $w --device 0 --frames 300
  done | tee "$OUT/video-anomaly/benchmark_fps.txt"
)

video_live_config() {
  local cfgd="$OUT/video-anomaly/live_config" live="$OUT/video-anomaly/live"
  mkdir -p "$cfgd" "$live"
  cp "$ROOT/video-anomaly/config/models.yaml" "$ROOT/video-anomaly/config/alerts.yaml" "$cfgd/"
  cat > "$cfgd/cameras.yaml" <<EOF
# Headless server: no webcam, so both sources are files.
sources:
  - {name: vtest, type: file, source: $SAMPLES/vtest.avi, roi: null}
  - {name: synthetic, type: file, source: $ROOT/video-anomaly/data/raw/sample_clips/sample1.mp4, roi: null}
expected_flow_deg: {vtest: 0, synthetic: 0}
reconnect: {max_retries: 3, retry_delay_sec: 1.0}
EOF
  sed -i "s#db_path: .*#db_path: $live/alerts.db#; s#clips_dir: .*#clips_dir: $live/clips#; s#thumbnails_dir: .*#thumbnails_dir: $live/thumbnails#" "$cfgd/alerts.yaml"
  echo "$cfgd"
}

video_live() ( set -e  # the repo's own CLI, headless, 90 s
  cd "$ROOT/video-anomaly"
  cfgd=$(video_live_config)
  timeout -s INT 90 "$ENVS/video-anomaly/bin/python" -m sentinel.main --config "$cfgd" --headless --metrics-port 9187 &
  pid=$!
  sleep 60
  curl -s http://127.0.0.1:9187/metrics | grep -vE '^#' | grep -E 'inference_fps|alert_count|stream_lag|frames_processed' | tee "$OUT/video-anomaly/live/metrics_snapshot.txt" || true
  wait $pid || [ $? -eq 124 ]
)

video_stress() ( set -e
  cd "$ROOT/video-anomaly"
  cfgd=$(video_live_config)
  py video-anomaly scripts/stress_test.py --duration 3m --config "$cfgd" --sample-interval-sec 20 | tee "$OUT/video-anomaly/stress_test_3m.txt"
)

# ------------------------------------------------------------------ voice_dub

voice_ld() {
  local sp; sp=$("$ENVS/voice_dub/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  echo "$sp/nvidia/cudnn/lib:$sp/nvidia/cublas/lib"
}
voice_env() {
  export PATH="$ENVS/voice_dub/bin:$PATH"
  export LD_LIBRARY_PATH="$(voice_ld)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

voice_tests() ( voice_env; cd "$ROOT/voice_dub" && py voice_dub -m pytest -v )

voice_inputs() ( set -e  # audio samples -> waveform videos, so the dub is remuxed into an MP4
  d="$OUT/voice_dub/inputs"; mkdir -p "$d"
  for f in "$SAMPLES"/two_speakers.wav "$SAMPLES"/my_voice.*; do
    [ -f "$f" ] || continue
    n=$(basename "${f%.*}")
    ffmpeg -y -loglevel error -i "$f" \
      -filter_complex "[0:a]showwaves=s=1280x360:mode=cline:colors=0x3b82f6,format=yuv420p[v]" \
      -map "[v]" -map 0:a -c:v libx264 -c:a aac -shortest "$d/$n.mp4"
    ls -l "$d/$n.mp4"
  done
)

voice_run() (
  voice_env
  cd "$ROOT/voice_dub" || exit 1
  if [ "${COQUI_TOS_AGREED:-}" != "1" ]; then
    echo "COQUI_TOS_AGREED is not 1: XTTS-v2 needs its CPML licence accepted before it can download."; exit 4
  fi
  # Kept small on purpose: diarization off (no gated pyannote model), Whisper "small" instead of
  # large-v3. NLLB-600M and XTTS-v2 are already the smallest options the pipeline supports.
  fail=0
  for inp in "$OUT"/voice_dub/inputs/*.mp4; do
    n=$(basename "$inp" .mp4)
    for lang in es de; do
      py voice_dub "$RUN/voice_dub/dub_demo.py" "$inp" --target-lang $lang --no-diarize \
        --whisper-model "$VOICE_WHISPER_MODEL" --out-dir "$OUT/voice_dub/${n}_en_to_${lang}" || fail=1
    done
  done
  exit $fail
)

# ------------------------------------------------------------------ phone-support-agent

phone_tests() ( cd "$ROOT/phone-support-agent" && py phone-support-agent -m pytest -v )

phone_eval() ( set -e
  cd "$ROOT/phone-support-agent"
  d="$OUT/phone-support-agent"; mkdir -p "$d"
  py phone-support-agent -m eval.run_eval
  cp eval/results.md "$d/eval_results.md"
  git checkout -- eval/results.md  # keep the repo tree clean; the fresh copy lives in demo_outputs
)

phone_cli_sim() ( set -e  # the repo's own interactive simulator, fed a scripted caller on stdin
  cd "$ROOT/phone-support-agent"
  d="$OUT/phone-support-agent"; mkdir -p "$d"
  printf '%s\n' "I'd like to book a teeth cleaning" "monday" "9:00" "Priya Nair" "yes please" "quit" \
    | LLM_PROVIDER=fake timeout 120 "$ENVS/phone-support-agent/bin/python" -m src.simulate | tee "$d/simulate_cli_fake.txt"
)

phone_calls_fake() ( cd "$ROOT/phone-support-agent" && py phone-support-agent "$RUN/phone-support-agent/call_demo.py" --provider fake --out "$OUT/phone-support-agent/calls_fake" )

phone_calls_ollama() ( set -e
  cd "$ROOT/phone-support-agent"
  curl -sf http://127.0.0.1:11434/api/tags | grep -q "\"${PHONE_OLLAMA_MODEL}\"" || { echo "Ollama model $PHONE_OLLAMA_MODEL not found"; exit 5; }
  # Ollama's OpenAI-compatible endpoint; the key is a placeholder string Ollama ignores.
  LLM_PROVIDER=openai OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama OPENAI_MODEL="$PHONE_OLLAMA_MODEL" \
    py phone-support-agent "$RUN/phone-support-agent/call_demo.py" --provider openai --label "ollama:$PHONE_OLLAMA_MODEL" --out "$OUT/phone-support-agent/calls_ollama"
)

phone_server() ( set -e
  cd "$ROOT/phone-support-agent"
  d="$OUT/phone-support-agent/server"; port=8766; mkdir -p "$d"
  DATABASE_PATH="$d/calls.db" py phone-support-agent -m uvicorn src.server:app --host 127.0.0.1 --port $port > "$d/uvicorn.log" 2>&1 &
  pid=$!; trap 'kill $pid 2>/dev/null' EXIT
  wait_http "http://127.0.0.1:$port/health" 60
  curl -sf "http://127.0.0.1:$port/health" | tee "$d/health.json"; echo
  curl -sf -X POST "http://127.0.0.1:$port/voice" --data-urlencode "From=+15555550100" | tee "$d/voice_twiml.xml"; echo
  curl -s -o "$d/dashboard.html" -w 'dashboard HTTP %{http_code}\n' "http://127.0.0.1:$port/dashboard"
)

# ------------------------------------------------------------------ main

stage() {
  case "$1" in
    setup)   write_constraints
             run_stage setup_depth setup_depth; run_stage setup_video setup_video
             run_stage setup_voice setup_voice; run_stage setup_phone setup_phone ;;
    samples) run_stage samples stage_samples ;;
    depth)   run_stage depth_tests depth_tests; run_stage depth_mvp_repo depth_mvp_repo
             run_stage depth_demo depth_demo; run_stage depth_bench depth_bench; run_stage depth_api depth_api ;;
    video)   run_stage video_tests video_tests; run_stage video_render video_render
             run_stage video_bench video_bench; run_stage video_live video_live; run_stage video_stress video_stress ;;
    voice)   run_stage voice_tests voice_tests; run_stage voice_inputs voice_inputs; run_stage voice_run voice_run ;;
    phone)   run_stage phone_tests phone_tests; run_stage phone_eval phone_eval; run_stage phone_cli_sim phone_cli_sim
             run_stage phone_calls_fake phone_calls_fake; run_stage phone_calls_ollama phone_calls_ollama
             run_stage phone_server phone_server ;;
    report)  python3 "$RUN/make_summary.py" "$OUT" && echo "wrote $OUT/SUMMARY.md" ;;
    *)       echo "unknown stage: $1"; return 1 ;;
  esac
}

log "run_all: stages=${*:-all}  GPU=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
for s in ${@:-setup samples depth video voice phone report}; do stage "$s"; done
log "run_all finished. Status:"
column -t -s $'\t' "$STATUS" | tail -n 40
