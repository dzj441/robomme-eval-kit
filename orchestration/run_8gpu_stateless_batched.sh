#!/usr/bin/env bash
# One stateless batched policy server per GPU, shared by configurable evaluators.
set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset ONLY_TASKS

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
KIT_ROOT=${KIT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

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
LEGACY_EXACT_ENCODE=${LEGACY_EXACT_ENCODE:-false}
DETERMINISTIC_PREWARM=${DETERMINISTIC_PREWARM:-true}
HISTORY_TRANSPORT_DTYPE=${HISTORY_TRANSPORT_DTYPE:-float32}
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$ROOT/eval_out/.jax_compilation_cache_550}
SERVER_XLA_FLAGS=${SERVER_XLA_FLAGS-${XLA_FLAGS:-}}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-1200}

SERVER_PY=${SERVER_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme-vla/bin/python}
EVAL_PY=${EVAL_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme/bin/python}
SERVER_SITE_PACKAGES=${SERVER_SITE_PACKAGES:-$("$SERVER_PY" -c 'import site; print(site.getsitepackages()[0])')}
OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_ckpt/openpi_assets}

NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')}
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
EXPECTED_MODEL_FINGERPRINT=$("$EVAL_PY" - "$CKPT" "$SEED" <<'PY'
import hashlib
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1]).resolve()
seed = int(sys.argv[2])
history_path = checkpoint_dir.parent / "history_config.txt"
history_config = history_path.read_text() if history_path.exists() else ""
payload = "\n".join(
    [
        "robomme-stateless-protocol-v2",
        "mme_vla_suite",
        str(checkpoint_dir),
        history_config,
        str(seed),
    ]
)
print(hashlib.sha256(payload.encode()).hexdigest())
PY
)

SERVER_PIDS=()
SERVER_PGIDS=()
EVAL_PIDS=()
EVAL_PGIDS=()
CLEANUP_DONE=0

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

process_group_has_live_members() {
  local pgid=$1
  ps -eo pgid=,stat= | awk -v target="$pgid" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

terminate_process_groups() {
  local label=$1
  shift
  local -a pgids=("$@")
  local pgid deadline any_alive
  if (( ${#pgids[@]} == 0 )); then
    return 0
  fi

  for pgid in "${pgids[@]}"; do
    if process_group_has_live_members "$pgid"; then
      echo "stopping $label process group $pgid"
      kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pgid in "${pgids[@]}"; do
      if process_group_has_live_members "$pgid"; then
        any_alive=1
        break
      fi
    done
    if (( any_alive == 0 )); then
      return 0
    fi
    sleep 0.2
  done

  for pgid in "${pgids[@]}"; do
    if process_group_has_live_members "$pgid"; then
      echo "force-stopping $label process group $pgid" >&2
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done
}

cleanup_processes() {
  local pid
  if (( CLEANUP_DONE == 1 )); then
    return
  fi
  CLEANUP_DONE=1

  # Stop clients before servers so no request is in flight during shutdown.
  terminate_process_groups evaluator "${EVAL_PGIDS[@]}"
  terminate_process_groups server "${SERVER_PGIDS[@]}"
  for pid in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_processes
  exit "$status"
}

on_signal() {
  local status=$1
  trap - EXIT INT TERM
  cleanup_processes
  exit "$status"
}
trap on_exit EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

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
  echo "ckpt=$CKPT"
  echo "ckpt_id=$CKPT_ID"
  echo "base_port=$BASE_PORT"
  echo "expected_model_fingerprint=$EXPECTED_MODEL_FINGERPRINT"
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
  echo "legacy_exact_encode=$LEGACY_EXACT_ENCODE"
  echo "deterministic_prewarm=$DETERMINISTIC_PREWARM"
  echo "history_transport_dtype=$HISTORY_TRANSPORT_DTYPE"
  echo "jax_cache_dir=$JAX_CACHE_DIR"
  echo "server_xla_flags=$SERVER_XLA_FLAGS"
  echo "server_ready_timeout_seconds=$SERVER_READY_TIMEOUT_SECONDS"
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
if [[ "$LEGACY_EXACT_ENCODE" != "true" && "$LEGACY_EXACT_ENCODE" != "false" ]]; then
  echo "LEGACY_EXACT_ENCODE must be true or false" >&2
  exit 1
fi
if [[ "$DETERMINISTIC_PREWARM" != "true" && "$DETERMINISTIC_PREWARM" != "false" ]]; then
  echo "DETERMINISTIC_PREWARM must be true or false" >&2
  exit 1
fi
if [[ ! "$SERVER_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SERVER_READY_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
if [[ "$LEGACY_EXACT_ENCODE" == "true" && "$LAZY_HISTORY_ENCODE" == "true" ]]; then
  echo "LEGACY_EXACT_ENCODE is incompatible with LAZY_HISTORY_ENCODE" >&2
  exit 1
fi
if [[ "$LEGACY_EXACT_ENCODE" == "true" && "$ENCODE_MAX_BATCH_SIZE" != "1" ]]; then
  echo "LEGACY_EXACT_ENCODE requires ENCODE_MAX_BATCH_SIZE=1" >&2
  exit 1
fi

# Bind all ports at once before spending time loading checkpoints. SO_REUSEADDR
# matches the websocket server while still rejecting a live listener.
if ! "$EVAL_PY" - "$BASE_PORT" "$NUM_GPUS" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
num_ports = int(sys.argv[2])
sockets = []
try:
    for port in range(base_port, base_port + num_ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sockets.append(sock)
except OSError as exc:
    print(f"required server port {port} is unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    for sock in sockets:
        sock.close()
PY
then
  record_timing ports_busy
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

if [[ "$DETERMINISTIC_PREWARM" == "true" ]]; then
  prewarm_signature=$(
    {
      printf '%s\n' \
        "$NVIDIA_DRIVER_VERSION" \
        "$CKPT" \
        "$SEED" \
        "$MAX_BATCH_SIZE" \
        "$LEGACY_EXACT_ENCODE" \
        "$SERVER_XLA_FLAGS"
      (
        cd "$POLICY_REPO"
        find scripts/serve_policy.py src/mme_vla_suite src/openpi \
          -type f -name '*.py' -print0 \
          | LC_ALL=C sort -z \
          | xargs -0 sha256sum
      )
      "$SERVER_PY" -c \
        'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)'
    } | sha256sum | awk '{print $1}'
  )
  prewarm_marker="$JAX_CACHE_DIR/.robomme_prewarm_${prewarm_signature}"
  echo "prewarm_signature=$prewarm_signature" >> "$TIMING_FILE"
  if [[ -f "$prewarm_marker" ]]; then
    echo "reusing deterministic JAX prewarm marker"
    echo "prewarm_cache_hit=true" >> "$TIMING_FILE"
  else
    echo "prewarming JAX cache in one isolated GPU process"
    echo "prewarm_cache_hit=false" >> "$TIMING_FILE"
    prewarm_affinity=()
    if [[ "$CPU_AFFINITY" == "per_gpu_v1" ]]; then
      prewarm_affinity=(
        numactl
        --physcpubind="$(gpu_cpu_list 0)"
        --membind="$(gpu_numa_node 0)"
      )
    fi
    prewarm_exact_args=()
    if [[ "$LEGACY_EXACT_ENCODE" == "true" ]]; then
      prewarm_exact_args=(--legacy-exact-encode)
    fi
    (
      cd "$POLICY_REPO"
      "${prewarm_affinity[@]}" env \
        CUDA_VISIBLE_DEVICES=0 \
        LD_LIBRARY_PATH="$SERVER_LD_PATH" \
        OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
        PYTHONPATH="$SERVER_PYTHONPATH" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
        JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" \
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
        XLA_FLAGS="$SERVER_XLA_FLAGS" \
        "$SERVER_PY" scripts/serve_policy.py \
          --seed="$SEED" \
          --serving-mode=stateless-batched \
          --prewarm-only \
          --max-batch-size="$MAX_BATCH_SIZE" \
          --encode-max-total-frames="$ENCODE_MAX_TOTAL_FRAMES" \
          "${prewarm_exact_args[@]}" \
          policy:checkpoint \
          --policy.config=mme_vla_suite \
          --policy.dir="$CKPT"
    ) > "$OUT/logs/prewarm.log" 2>&1
    : > "$prewarm_marker"
    echo "prewarm complete"
  fi
fi

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
    exact_encode_args=()
    if [[ "$LEGACY_EXACT_ENCODE" == "true" ]]; then
      exact_encode_args=(--legacy-exact-encode)
    fi
    exec setsid "${affinity_prefix[@]}" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      LD_LIBRARY_PATH="$SERVER_LD_PATH" \
      OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
      PYTHONPATH="$SERVER_PYTHONPATH" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
      JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" \
      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
      XLA_FLAGS="$SERVER_XLA_FLAGS" \
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
        "${exact_encode_args[@]}" \
        policy:checkpoint \
        --policy.config=mme_vla_suite \
        --policy.dir="$CKPT"
  ) > "$OUT/logs/server_gpu${gpu}.log" 2>&1 &
  SERVER_PIDS+=("$!")
  SERVER_PGIDS+=("$!")
  echo "  gpu$gpu server pid=$! port=$port"
done

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + gpu))
  pid=${SERVER_PIDS[$gpu]}
  metadata_file="$OUT/logs/server_gpu${gpu}.metadata.json"
  metadata_tmp="${metadata_file}.tmp"
  ready=0
  ready_deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
  while (( SECONDS < ready_deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    if PYTHONPATH="$POLICY_REPO/packages/openpi-client/src" \
      "$EVAL_PY" "$KIT_ROOT/orchestration/verify_stateless_server.py" \
        --port "$port" \
        --fingerprint "$EXPECTED_MODEL_FINGERPRINT" \
        --timeout 2 \
        > "$metadata_tmp" 2>/dev/null; then
      mv "$metadata_tmp" "$metadata_file"
      ready=1
      break
    fi
    sleep 2
  done
  rm -f "$metadata_tmp"
  if (( ready == 0 )); then
    if kill -0 "$pid" 2>/dev/null; then
      echo "server gpu$gpu pid=$pid on port $port failed identity/readiness verification" >&2
    else
      echo "server gpu$gpu pid=$pid exited before becoming ready on port $port" >&2
    fi
    tail -n 80 "$OUT/logs/server_gpu${gpu}.log" >&2 || true
    record_timing server_start_failed
    exit 1
  fi
  actual_pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]')
  if [[ "$actual_pgid" != "$pid" ]]; then
    echo "server gpu$gpu is not in its tracked process group: pid=$pid pgid=$actual_pgid" >&2
    record_timing server_process_group_failed
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
    exec setsid "${affinity_prefix[@]}" env \
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
  EVAL_PGIDS+=("$!")
  echo "  shard$shard pid=$! -> gpu$gpu port$port"
done

FAIL=0
for index in "${!EVAL_PIDS[@]}"; do
  if ! wait "${EVAL_PIDS[$index]}"; then
    echo "evaluator shard$index failed" >&2
    FAIL=1
  fi
done

# A completed episode count isn't sufficient: verify that the exact servers
# launched for this run stayed alive and never failed to bind.
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  pid=${SERVER_PIDS[$gpu]}
  port=$((BASE_PORT + gpu))
  server_log="$OUT/logs/server_gpu${gpu}.log"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "server gpu$gpu pid=$pid exited during evaluation" >&2
    FAIL=1
  elif ! PYTHONPATH="$POLICY_REPO/packages/openpi-client/src" \
    "$EVAL_PY" "$KIT_ROOT/orchestration/verify_stateless_server.py" \
      --port "$port" \
      --fingerprint "$EXPECTED_MODEL_FINGERPRINT" \
      --timeout 2 \
      >/dev/null; then
    echo "server gpu$gpu failed final identity verification" >&2
    FAIL=1
  fi
  if grep -Eq 'Traceback|address already in use|OSError: \[Errno 98\]' "$server_log"; then
    echo "server gpu$gpu log contains a fatal startup/runtime error" >&2
    FAIL=1
  fi
done

cleanup_processes
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
