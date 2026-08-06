#!/usr/bin/env bash
# Fastest validated 8-GPU RoboMME configuration (800 episodes, no videos).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
SEED=${SEED:-7}

export ROOT SEED
export OUT=${OUT:-$ROOT/eval_out/8gpu_fastest_seed${SEED}_stateless}
export NUM_GPUS=${NUM_GPUS:-8}
export CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-4}
export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-1}
export MAX_WAIT_MS=${MAX_WAIT_MS:-0}
export SHARDING=${SHARDING:-balanced_steal_v1}
export WEIGHT_PROFILE=${WEIGHT_PROFILE:-$SCRIPT_DIR/profiles/seed7_lazy_numa_warm_episode_seconds.json}
export VIDEO_MODE=${VIDEO_MODE:-off}
export EVAL_NUM_THREADS=${EVAL_NUM_THREADS:-1}
export LAZY_HISTORY_ENCODE=${LAZY_HISTORY_ENCODE:-true}
export HISTORY_TRANSPORT_DTYPE=${HISTORY_TRANSPORT_DTYPE:-float32}
export ENCODE_MAX_BATCH_SIZE=${ENCODE_MAX_BATCH_SIZE:-1}
export ENCODE_MAX_WAIT_MS=${ENCODE_MAX_WAIT_MS:-0}
export CPU_AFFINITY=${CPU_AFFINITY:-per_gpu_v1}

exec bash "$SCRIPT_DIR/run_8gpu_stateless_batched.sh"
