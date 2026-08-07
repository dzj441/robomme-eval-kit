#!/usr/bin/env bash
# Run the same RoboMME episode through independent legacy servers and retain
# client-side traces for first-divergence analysis.
set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
TASK=${TASK:-BinFill}
EPISODE_ID=${EPISODE_ID:-5}
REPLICAS=${REPLICAS:-16}
NUM_GPUS=${NUM_GPUS:-8}
BASE_PORT=${BASE_PORT:-8800}
SEED=${SEED:-7}
MEM_FRACTION=${MEM_FRACTION:-0.95}
OUT=${OUT:-$ROOT/eval_out/diagnostics/legacy_trace_${TASK}_ep${EPISODE_ID}}
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$ROOT/eval_out/.jax_compilation_cache_550}
JAX_CACHE_MODE=${JAX_CACHE_MODE:-shared}
SERVER_XLA_FLAGS=${SERVER_XLA_FLAGS-${XLA_FLAGS:-}}

CKPT=${CKPT:-$ROOT/ckpt/perceptual-framesamp-modul/home/daiyp/MME-VLA-Suite/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999}
SERVER_PY=${SERVER_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme-vla/bin/python}
EVAL_PY=${EVAL_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme/bin/python}
SERVER_SITE_PACKAGES=${SERVER_SITE_PACKAGES:-$($SERVER_PY -c 'import site; print(site.getsitepackages()[0])')}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_ckpt/openpi_assets}

NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')}
NVIDIA_DRIVER_ROOT=${NVIDIA_DRIVER_ROOT:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/$NVIDIA_DRIVER_VERSION}
NVIDIA_RENDER_LIBS=$NVIDIA_DRIVER_ROOT/runtime-libs
NVIDIA_VK_ICD=$NVIDIA_DRIVER_ROOT/nvidia_icd.local.json
NVIDIA_EGL_VENDOR=$NVIDIA_DRIVER_ROOT/10_nvidia.local.json
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/robomme-legacy-trace-${BASE_PORT}}

if [[ -e "$OUT" ]]; then
  echo "refusing to reuse diagnostic output: $OUT" >&2
  exit 1
fi
mkdir -p \
  "$OUT/logs" \
  "$OUT/traces" \
  "$OUT/policy_traces" \
  "$XDG_RUNTIME_DIR"
if [[ "$JAX_CACHE_MODE" == "shared" ]]; then
  mkdir -p "$JAX_CACHE_DIR"
elif [[ "$JAX_CACHE_MODE" == "per_replica" ]]; then
  mkdir -p "$OUT/jax_cache"
else
  echo "JAX_CACHE_MODE must be shared or per_replica" >&2
  exit 1
fi
chmod 700 "$XDG_RUNTIME_DIR"

SERVER_LD_PATH=$(find "$SERVER_SITE_PACKAGES/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
SERVER_PYTHONPATH="$POLICY_REPO/src:$POLICY_REPO/packages/openpi-client/src"
EVAL_PYTHONPATH="$POLICY_REPO/packages/openpi-client/src:$POLICY_REPO/examples/robomme:$POLICY_REPO/src"

SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for replica in $(seq 0 $((REPLICAS - 1))); do
  gpu=$((replica % NUM_GPUS))
  port=$((BASE_PORT + replica))
  replica_cache_dir="$JAX_CACHE_DIR"
  if [[ "$JAX_CACHE_MODE" == "per_replica" ]]; then
    replica_cache_dir="$OUT/jax_cache/replica${replica}"
    mkdir -p "$replica_cache_dir"
  fi
  (
    cd "$POLICY_REPO"
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      LD_LIBRARY_PATH="$SERVER_LD_PATH" \
      OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
      PYTHONPATH="$SERVER_PYTHONPATH" \
      ROBOMME_POLICY_TRACE_DIR="$OUT/policy_traces" \
      ROBOMME_POLICY_TRACE_LABEL="replica${replica}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
      JAX_COMPILATION_CACHE_DIR="$replica_cache_dir" \
      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
      XLA_FLAGS="$SERVER_XLA_FLAGS" \
      "$SERVER_PY" scripts/serve_policy.py \
        --seed="$SEED" \
        --port="$port" \
        policy:checkpoint \
        --policy.config=mme_vla_suite \
        --policy.dir="$CKPT"
  ) > "$OUT/logs/server${replica}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

for replica in $(seq 0 $((REPLICAS - 1))); do
  port=$((BASE_PORT + replica))
  ready=0
  for _ in $(seq 1 180); do
    if "$EVAL_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$port),2); s.close()" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "replica $replica server failed to start" >&2
    tail -n 50 "$OUT/logs/server${replica}.log" >&2 || true
    exit 1
  fi
done

start_epoch=$(date +%s)
for replica in $(seq 0 $((REPLICAS - 1))); do
  gpu=$((replica % NUM_GPUS))
  port=$((BASE_PORT + replica))
  save_dir="$OUT/replica${replica}"
  (
    cd "$POLICY_REPO/examples/robomme"
    env \
      LD_LIBRARY_PATH="$NVIDIA_RENDER_LIBS" \
      VK_ICD_FILENAMES="$NVIDIA_VK_ICD" \
      __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR" \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      EGL_PLATFORM=surfaceless \
      XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$EVAL_PYTHONPATH" \
      "$EVAL_PY" eval.py \
        --args.host=127.0.0.1 \
        --args.port="$port" \
        --args.serving-mode=legacy \
        --args.model-seed="$SEED" \
        --args.policy-name=framesamp-modul \
        --args.model-ckpt-id=79999 \
        --args.video-mode=off \
        --args.num-episodes=50 \
        --args.only-tasks="$TASK" \
        --args.episode-ids="$EPISODE_ID" \
        --args.trace-dir="$OUT/traces" \
        --args.trace-label="replica${replica}" \
        --args.worker-id="$replica" \
        --args.save-dir="$save_dir"
  ) > "$OUT/logs/eval${replica}.log" 2>&1 &
  EVAL_PIDS+=("$!")
done

fail=0
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
EVAL_PIDS=()

end_epoch=$(date +%s)
"$EVAL_PY" - "$OUT" "$TASK" "$EPISODE_ID" "$REPLICAS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task = sys.argv[2]
episode_id = str(int(sys.argv[3]))
replicas = int(sys.argv[4])
outcomes = []
for replica in range(replicas):
    paths = list((root / f"replica{replica}").glob("**/progress.json"))
    value = None
    if paths:
        value = json.loads(paths[0].read_text()).get(task, {}).get(episode_id)
    outcomes.append(value)
payload = {
    "task": task,
    "episode_id": int(episode_id),
    "outcomes": outcomes,
    "successes": sum(value is True for value in outcomes),
    "failures": sum(value is False for value in outcomes),
}
(root / "SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

printf 'wall_seconds=%s\n' "$((end_epoch - start_epoch))" > "$OUT/TIMING.txt"
exit "$fail"
