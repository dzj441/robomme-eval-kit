#!/usr/bin/env bash
# One stateless batched policy server per GPU, shared by configurable evaluators.
set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset ONLY_TASKS

ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
KIT_ROOT=${KIT_ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/robomme-eval-kit-stateless-batched}

CKPT=${CKPT:-$ROOT/ckpt/perceptual-framesamp-modul/home/daiyp/MME-VLA-Suite/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999}
CKPT_ID=${CKPT_ID:-79999}
POLICY_NAME=${POLICY_NAME:-framesamp-modul}
SEED=${SEED:-7}
OUT=${OUT:-$ROOT/eval_out/8gpu_16shards_seed${SEED}_550_stateless_batched}
EPISODES=${EPISODES:-50}

NUM_GPUS=${NUM_GPUS:-8}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-2}
NUM_SHARDS=$((NUM_GPUS * CLIENTS_PER_GPU))
BASE_PORT=${BASE_PORT:-8600}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-2}
MAX_WAIT_MS=${MAX_WAIT_MS:-100}
MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-256}
MEM_FRACTION=${MEM_FRACTION:-0.95}
SHARDING=${SHARDING:-global_round_robin_v1}
WEIGHT_PROFILE=${WEIGHT_PROFILE:-}
VIDEO_MODE=${VIDEO_MODE:-save}
EVAL_NUM_THREADS=${EVAL_NUM_THREADS:-1}
ENCODE_MAX_BATCH_SIZE=${ENCODE_MAX_BATCH_SIZE:-1}
ENCODE_MAX_WAIT_MS=${ENCODE_MAX_WAIT_MS:-0}
ENCODE_MAX_TOTAL_FRAMES=${ENCODE_MAX_TOTAL_FRAMES:-128}
CPU_AFFINITY=${CPU_AFFINITY:-none}
LAZY_HISTORY_ENCODE=${LAZY_HISTORY_ENCODE:-false}
HISTORY_TRANSPORT_DTYPE=${HISTORY_TRANSPORT_DTYPE:-float32}
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$ROOT/eval_out/.jax_compilation_cache_550}

SERVER_PY=${SERVER_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme-vla/bin/python}
EVAL_PY=${EVAL_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme/bin/python}
SERVER_SITE_PACKAGES=/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme-vla/lib/python3.11/site-packages
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_ckpt/openpi_assets}

NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)}
NVIDIA_DRIVER_ROOT=${NVIDIA_DRIVER_ROOT:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/$NVIDIA_DRIVER_VERSION}
NVIDIA_RENDER_LIBS=$NVIDIA_DRIVER_ROOT/runtime-libs
NVIDIA_VK_ICD=$NVIDIA_DRIVER_ROOT/nvidia_icd.local.json
NVIDIA_EGL_VENDOR=$NVIDIA_DRIVER_ROOT/10_nvidia.local.json
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/robomme-stateless-${BASE_PORT}}

for required_path in \
  "$POLICY_REPO" \
  "$KIT_ROOT" \
  "$CKPT" \
  "$NVIDIA_RENDER_LIBS" \
  "$NVIDIA_VK_ICD" \
  "$NVIDIA_EGL_VENDOR"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing required path: $required_path" >&2
    exit 1
  fi
done

visible_gpus=$(nvidia-smi -L | wc -l)
if (( visible_gpus < NUM_GPUS )); then
  echo "need at least $NUM_GPUS visible GPUs, found $visible_gpus" >&2
  exit 1
fi

mkdir -p "$OUT/logs" "$XDG_RUNTIME_DIR" "$JAX_CACHE_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

ALL_TASKS="BinFill,StopCube,PickXtimes,SwingXtimes,VideoUnmask,ButtonUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"
EXPECTED_EPISODES=$((16 * EPISODES))
TIMING_FILE=${TIMING_FILE:-$OUT/TIMING.txt}
INITIAL_COMPLETED=$(find "$OUT" -type f -name '*.mp4' | wc -l)
START_EPOCH=$(date +%s)
START_TIME=$(date -Is)

SERVER_PIDS=()
EVAL_PIDS=()

completed_count() {
  if [[ -f "$OUT/ownership.json" ]]; then
    "$EVAL_PY" "$KIT_ROOT/orchestration/count_owned_progress.py" "$OUT"
  else
    find "$OUT" -type f -name '*.mp4' | wc -l
  fi
}

gpu_cpu_list() {
  local gpu=$1 first
  if (( gpu < 4 )); then
    first=$((gpu * 8))
  else
    first=$((32 + (gpu - 4) * 8))
  fi
  echo "$first-$((first + 7)),$((first + 64))-$((first + 71))"
}

gpu_numa_node() {
  local gpu=$1
  if (( gpu < 4 )); then echo 0; else echo 1; fi
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${EVAL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

record_timing() {
  local status=$1
  local end_epoch end_time wall_seconds total_completed new_completed episodes_per_hour seconds_per_episode
  end_epoch=$(date +%s)
  end_time=$(date -Is)
  wall_seconds=$((end_epoch - START_EPOCH))
  total_completed=$(completed_count)
  new_completed=$((total_completed - INITIAL_COMPLETED))
  if (( new_completed > 0 && wall_seconds > 0 )); then
    episodes_per_hour=$(awk -v n="$new_completed" -v s="$wall_seconds" 'BEGIN {printf "%.2f", n * 3600 / s}')
    seconds_per_episode=$(awk -v n="$new_completed" -v s="$wall_seconds" 'BEGIN {printf "%.3f", s / n}')
  else
    episodes_per_hour=NA
    seconds_per_episode=NA
  fi
  {
    echo "end_time=$end_time"
    echo "wall_seconds=$wall_seconds"
    echo "total_completed_episodes=$total_completed"
    echo "new_completed_episodes=$new_completed"
    echo "episodes_per_hour=$episodes_per_hour"
    echo "seconds_per_episode=$seconds_per_episode"
    echo "status=$status"
  } >> "$TIMING_FILE"
}

{
  echo "initial_status=running"
  echo "start_time=$START_TIME"
  echo "seed=$SEED"
  echo "expected_episodes=$EXPECTED_EPISODES"
  echo "initial_completed_episodes=$INITIAL_COMPLETED"
  echo "num_gpus=$NUM_GPUS"
  echo "clients_per_gpu=$CLIENTS_PER_GPU"
  echo "num_shards=$NUM_SHARDS"
  echo "max_batch_size=$MAX_BATCH_SIZE"
  echo "max_wait_ms=$MAX_WAIT_MS"
  echo "sharding=$SHARDING"
  echo "weight_profile=$WEIGHT_PROFILE"
  echo "video_mode=$VIDEO_MODE"
  echo "eval_num_threads=$EVAL_NUM_THREADS"
  echo "encode_max_batch_size=$ENCODE_MAX_BATCH_SIZE"
  echo "encode_max_wait_ms=$ENCODE_MAX_WAIT_MS"
  echo "encode_max_total_frames=$ENCODE_MAX_TOTAL_FRAMES"
  echo "cpu_affinity=$CPU_AFFINITY"
  echo "lazy_history_encode=$LAZY_HISTORY_ENCODE"
  echo "history_transport_dtype=$HISTORY_TRANSPORT_DTYPE"
  echo "jax_cache_dir=$JAX_CACHE_DIR"
  echo "policy_repo=$POLICY_REPO"
  echo "kit_root=$KIT_ROOT"
} > "$TIMING_FILE"

if [[ "$SHARDING" =~ ^(balanced_lpt_v1|dynamic_queue_v1|balanced_steal_v1)$ && ! -f "$WEIGHT_PROFILE" ]]; then
  echo "$SHARDING requires WEIGHT_PROFILE, got: $WEIGHT_PROFILE" >&2
  exit 1
fi
if [[ "$HISTORY_TRANSPORT_DTYPE" != "float16" && "$HISTORY_TRANSPORT_DTYPE" != "float32" ]]; then
  echo "HISTORY_TRANSPORT_DTYPE must be float16 or float32" >&2
  exit 1
fi
if [[ "$CPU_AFFINITY" != "none" && "$CPU_AFFINITY" != "per_gpu_v1" ]]; then
  echo "unsupported CPU_AFFINITY: $CPU_AFFINITY" >&2
  exit 1
fi
if [[ "$LAZY_HISTORY_ENCODE" != "true" && "$LAZY_HISTORY_ENCODE" != "false" ]]; then
  echo "LAZY_HISTORY_ENCODE must be true or false" >&2
  exit 1
fi

echo "OUT=$OUT seed=$SEED shards=$NUM_SHARDS episodes=$EPISODES sharding=$SHARDING serving=stateless_batched" \
  | tee "$OUT/run_config.txt"
SHARD_ARGS=(
  --out "$OUT"
  --num-shards "$NUM_SHARDS"
  --episodes "$EPISODES"
  --policy-name "$POLICY_NAME"
  --ckpt-id "$CKPT_ID"
  --seed "$SEED"
  --sharding "$SHARDING"
)
if [[ -n "$WEIGHT_PROFILE" ]]; then
  SHARD_ARGS+=(--weight-profile "$WEIGHT_PROFILE")
fi
"$EVAL_PY" "$KIT_ROOT/orchestration/seed_episode_shards.py" \
  "${SHARD_ARGS[@]}" \
  | tee -a "$OUT/run_config.txt"

SERVER_LD_PATH=$(find "$SERVER_SITE_PACKAGES/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
SERVER_PYTHONPATH="$POLICY_REPO/src:$POLICY_REPO/packages/openpi-client/src"
EVAL_PYTHONPATH="$POLICY_REPO/packages/openpi-client/src:$POLICY_REPO/examples/robomme:$POLICY_REPO/src"

echo "starting $NUM_GPUS stateless batched policy servers"
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + gpu))
  (
    cd "$POLICY_REPO"
    affinity_prefix=()
    if [[ "$CPU_AFFINITY" == "per_gpu_v1" ]]; then
      affinity_prefix=(
        numactl
        --physcpubind="$(gpu_cpu_list "$gpu")"
        --membind="$(gpu_numa_node "$gpu")"
      )
    fi
    "${affinity_prefix[@]}" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      LD_LIBRARY_PATH="$SERVER_LD_PATH" \
      OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
      PYTHONPATH="$SERVER_PYTHONPATH" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
      JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" \
      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
      "$SERVER_PY" scripts/serve_policy.py \
        --seed="$SEED" \
        --port="$port" \
        --serving-mode=stateless-batched \
        --max-batch-size="$MAX_BATCH_SIZE" \
        --max-wait-ms="$MAX_WAIT_MS" \
        --max-queue-size="$MAX_QUEUE_SIZE" \
        --encode-max-batch-size="$ENCODE_MAX_BATCH_SIZE" \
        --encode-max-wait-ms="$ENCODE_MAX_WAIT_MS" \
        --encode-max-total-frames="$ENCODE_MAX_TOTAL_FRAMES" \
        policy:checkpoint \
        --policy.config=mme_vla_suite \
        --policy.dir="$CKPT"
  ) > "$OUT/logs/server_gpu${gpu}.log" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "  gpu$gpu server pid=$! port=$port"
done

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + gpu))
  ready=0
  for _ in $(seq 1 180); do
    if "$EVAL_PY" -c "import socket; s=socket.create_connection(('127.0.0.1',$port),2); s.close()" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "server gpu$gpu on port $port failed to become ready" >&2
    tail -n 80 "$OUT/logs/server_gpu${gpu}.log" >&2 || true
    record_timing server_start_failed
    exit 1
  fi
done

echo "starting $NUM_SHARDS evaluators"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard % NUM_GPUS))
  port=$((BASE_PORT + gpu))
  shard_dir="$OUT/shard${shard}"
  (
    cd "$POLICY_REPO/examples/robomme"
    affinity_prefix=()
    if [[ "$CPU_AFFINITY" == "per_gpu_v1" ]]; then
      affinity_prefix=(
        numactl
        --physcpubind="$(gpu_cpu_list "$gpu")"
        --membind="$(gpu_numa_node "$gpu")"
      )
    fi
    queue_args=()
    if [[ "$SHARDING" =~ ^(dynamic_queue_v1|balanced_steal_v1)$ ]]; then
      queue_args=(
        --args.work-queue-dir="$OUT/work_queue"
        --args.worker-id="$shard"
      )
    fi
    lazy_encode_args=()
    if [[ "$LAZY_HISTORY_ENCODE" == "true" ]]; then
      lazy_encode_args=(--args.lazy-history-encode)
    fi
    "${affinity_prefix[@]}" env \
      LD_LIBRARY_PATH="$NVIDIA_RENDER_LIBS" \
      VK_ICD_FILENAMES="$NVIDIA_VK_ICD" \
      __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR" \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      EGL_PLATFORM=surfaceless \
      XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS="$EVAL_NUM_THREADS" \
      MKL_NUM_THREADS="$EVAL_NUM_THREADS" \
      OPENBLAS_NUM_THREADS="$EVAL_NUM_THREADS" \
      NUMEXPR_NUM_THREADS="$EVAL_NUM_THREADS" \
      PYTHONPATH="$EVAL_PYTHONPATH" \
      "$EVAL_PY" eval.py \
        --args.host=127.0.0.1 \
        --args.port="$port" \
        --args.serving-mode=stateless-batched \
        --args.model-seed="$SEED" \
        --args.policy-name="$POLICY_NAME" \
        --args.model-ckpt-id="$CKPT_ID" \
        --args.video-mode="$VIDEO_MODE" \
        --args.history-transport-dtype="$HISTORY_TRANSPORT_DTYPE" \
        "${lazy_encode_args[@]}" \
        --args.num-episodes="$EPISODES" \
        --args.only-tasks="$ALL_TASKS" \
        "${queue_args[@]}" \
        --args.save-dir="$shard_dir"
  ) > "$OUT/logs/eval_shard${shard}.log" 2>&1 &
  EVAL_PIDS+=("$!")
  echo "  shard$shard pid=$! -> gpu$gpu port$port"
done

FAIL=0
for index in "${!EVAL_PIDS[@]}"; do
  if ! wait "${EVAL_PIDS[$index]}"; then
    echo "evaluator shard$index failed" >&2
    FAIL=1
  fi
done
EVAL_PIDS=()

cleanup
trap - EXIT INT TERM

"$EVAL_PY" "$KIT_ROOT/orchestration/merge_robomme.py" "$OUT" \
  | tee "$OUT/FINAL_REPORT.txt"

total_completed=$(completed_count)
error_count=$("$EVAL_PY" -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "aggregate.json"
payload = json.loads(p.read_text()) if p.exists() else {}
print(sum(payload.get("errors", {}).values()))
' "$OUT")
if (( FAIL == 0 && total_completed >= EXPECTED_EPISODES && error_count == 0 )); then
  record_timing complete
  exit 0
fi

record_timing incomplete
exit 1
