#!/usr/bin/env bash
# Evaluate the compact-cache 8xH100 training run on a driver-550 GPU node:
#   1. seeds 7/17/27 for checkpoints 60k/70k/79999;
#   2. seed 7 for checkpoints 10k..50k, reusing the three seed-7 runs above;
#   3. generate the complete seed-7 SR-vs-step curve.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
TRAIN_RUN=${TRAIN_RUN:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME_policy/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul_8h100_b64_fastio_seed42}
GRID_OUT=${GRID_OUT:-$ROOT/eval_out/h100_b64_fastio_seed42_3ckpt_3seed_fastest}
CURVE_OUT=${CURVE_OUT:-$ROOT/eval_out/h100_b64_fastio_seed42_seed7_curve_fastest}
POLICY_NAME=${POLICY_NAME:-h100-b64-fastio-seed42}
readonly EXPECTED_DRIVER=550.163.01
readonly NVIDIA_DRIVER_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/550.163.01
BASE_PORT=${BASE_PORT:-8600}
EVAL_PY=${EVAL_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme/bin/python}
NUM_GPUS=${NUM_GPUS:-8}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-4}
CPU_AFFINITY=${CPU_AFFINITY:-per_gpu_v1}

LAST_THREE="79999 70000 60000"
GRID_SEEDS="7 17 27"
EARLY_CHECKPOINTS="10000 20000 30000 40000 50000"
ALL_CHECKPOINTS="10000 20000 30000 40000 50000 60000 70000 79999"

host_tag=$(hostname -s | tr -cs '[:alnum:]_.-' '_')
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$ROOT/eval_out/.jax_compilation_cache_550_fastio_${host_tag}}

log() {
  echo "[$(date -Is)] $*"
}

if [[ ! -x "$EVAL_PY" ]]; then
  echo "missing evaluator Python: $EVAL_PY" >&2
  exit 1
fi
export PATH="$(dirname "$EVAL_PY"):$PATH"

for required_path in "$POLICY_REPO" "$TRAIN_RUN"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing required path: $required_path" >&2
    exit 1
  fi
done

if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ || ! "$CLIENTS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS and CLIENTS_PER_GPU must be positive integers" >&2
  exit 1
fi

visible_gpus=$(nvidia-smi -L | wc -l)
if (( visible_gpus < NUM_GPUS )); then
  echo "this evaluation requires $NUM_GPUS visible GPUs, found $visible_gpus" >&2
  exit 1
fi

mapfile -t driver_versions < <(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
)
if (( ${#driver_versions[@]} != 1 )) || [[ "${driver_versions[0]}" != "$EXPECTED_DRIVER" ]]; then
  echo "driver mismatch: expected all GPUs to use $EXPECTED_DRIVER, got: ${driver_versions[*]}" >&2
  exit 1
fi

runtime_root=$NVIDIA_DRIVER_ROOT
if [[ ! -d "$runtime_root/runtime-libs" ]]; then
  echo "missing driver runtime assets: $runtime_root" >&2
  exit 1
fi

if pgrep -f '[/]run_training4.py' >/dev/null; then
  echo "run_training4.py GPU holder is still running; stop it before evaluation" >&2
  exit 1
fi

for checkpoint in $ALL_CHECKPOINTS; do
  if [[ ! -f "$TRAIN_RUN/$checkpoint/_CHECKPOINT_METADATA" ]]; then
    echo "checkpoint is missing or incomplete: $TRAIN_RUN/$checkpoint" >&2
    exit 1
  fi
done

mkdir -p "$GRID_OUT" "$CURVE_OUT" "$JAX_CACHE_DIR"
cat > "$ROOT/eval_out/h100_b64_fastio_seed42_full_eval_550.MANIFEST.txt" <<EOF
created=$(date -Is)
host=$(hostname)
driver=$EXPECTED_DRIVER
train_run=$TRAIN_RUN
grid_out=$GRID_OUT
curve_out=$CURVE_OUT
last_three=$LAST_THREE
grid_seeds=$GRID_SEEDS
early_checkpoints=$EARLY_CHECKPOINTS
curve_seed=7
num_gpus=$NUM_GPUS
clients_per_gpu=$CLIENTS_PER_GPU
cpu_affinity=$CPU_AFFINITY
jax_cache_dir=$JAX_CACHE_DIR
EOF

export ROOT POLICY_REPO POLICY_NAME EVAL_PY NUM_GPUS CLIENTS_PER_GPU CPU_AFFINITY
export NVIDIA_DRIVER_VERSION=$EXPECTED_DRIVER NVIDIA_DRIVER_ROOT
export JAX_CACHE_DIR

log "PHASE 1/3: 3 checkpoints x 3 seeds"
TRAIN_RUN="$TRAIN_RUN" \
OUT_ROOT="$GRID_OUT" \
CHECKPOINTS="$LAST_THREE" \
SEEDS="$GRID_SEEDS" \
POLICY_NAME="$POLICY_NAME" \
BASE_PORT="$BASE_PORT" \
  bash "$SCRIPT_DIR/run_3seeds_3ckpts_fastest.sh"

log "PHASE 2/3: link completed seed-7 results into the curve directory"
for checkpoint in $LAST_THREE; do
  source_run=$GRID_OUT/ckpt$checkpoint/seed7
  target_run=$CURVE_OUT/ckpt$checkpoint/seed7
  if [[ ! -f "$source_run/aggregate.json" || ! -f "$source_run/TIMING.txt" ]]; then
    echo "completed source run is missing: $source_run" >&2
    exit 1
  fi
  if ! grep -q '^status=complete$' "$source_run/TIMING.txt"; then
    echo "source run is not complete: $source_run" >&2
    exit 1
  fi

  if [[ -L "$target_run" ]] && \
    [[ "$(readlink -f "$target_run")" == "$(readlink -f "$source_run")" ]]; then
    continue
  fi
  if [[ -e "$target_run" || -L "$target_run" ]]; then
    archived=${target_run}.replaced.$(date -u +%Y%m%dT%H%M%SZ)
    mv "$target_run" "$archived"
    log "ARCHIVE existing curve entry: $target_run -> $archived"
  fi
  mkdir -p "$(dirname "$target_run")"
  ln -s "$source_run" "$target_run"
done

log "PHASE 3/3: seed 7 for checkpoints 10k..50k"
TRAIN_RUN="$TRAIN_RUN" \
OUT_ROOT="$CURVE_OUT" \
CHECKPOINTS="$EARLY_CHECKPOINTS" \
SEEDS="7" \
POLICY_NAME="$POLICY_NAME" \
BASE_PORT="$BASE_PORT" \
  bash "$SCRIPT_DIR/run_3seeds_3ckpts_fastest.sh"

log "Generating the complete seed-7 SR-vs-step report"
# shellcheck disable=SC2086
"$EVAL_PY" "$SCRIPT_DIR/summarize_checkpoint_curve.py" "$CURVE_OUT" \
  --checkpoints $ALL_CHECKPOINTS --seed 7 \
  | tee "$CURVE_OUT/CURVE.stdout.txt"

log "COMPLETE"
log "3x3 summary: $GRID_OUT/SUMMARY.md"
log "seed-7 curve: $CURVE_OUT/CURVE.md"
log "seed-7 plot: $CURVE_OUT/sr_vs_step.png"
