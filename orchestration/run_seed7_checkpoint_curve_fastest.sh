#!/usr/bin/env bash
# Evaluate the 10k..80k checkpoint trajectory at seed 7 with the validated
# fastest stateless preset. Completed seed-7 runs from a preceding grid are
# reused instead of spending another full evaluation on the same tuple.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
TRAIN_RUN=${TRAIN_RUN:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME_policy/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul_8h100_b64_seed42}
OUT_ROOT=${OUT_ROOT:-$ROOT/eval_out/h100_b64_seed42_seed7_curve_fastest}
REUSE_ROOT=${REUSE_ROOT:-$ROOT/eval_out/h100_b64_seed42_3ckpt_3seed_fastest}
CHECKPOINTS=${CHECKPOINTS:-"10000 20000 30000 40000 50000 60000 70000 79999"}
SEED=${SEED:-7}
POLICY_NAME=${POLICY_NAME:-h100-b64-seed42}
BASE_PORT=${BASE_PORT:-8600}
EVAL_PY=${EVAL_PY:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/miniconda3/envs/robomme/bin/python}
NUM_GPUS=${NUM_GPUS:-8}
CLIENTS_PER_GPU=${CLIENTS_PER_GPU:-4}
CPU_AFFINITY=${CPU_AFFINITY:-per_gpu_v1}
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$ROOT/eval_out/.jax_compilation_cache_550}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-1200}

export ROOT POLICY_REPO POLICY_NAME NUM_GPUS CLIENTS_PER_GPU CPU_AFFINITY
export JAX_CACHE_DIR SERVER_READY_TIMEOUT_SECONDS
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ || ! "$CLIENTS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS and CLIENTS_PER_GPU must be positive integers" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT"
MASTER_LOG=$OUT_ROOT/master.log

log() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

is_complete() {
  local run=$1
  [[ -f "$run/aggregate.json" && -f "$run/TIMING.txt" ]] || return 1
  grep -q '^status=complete$' "$run/TIMING.txt" || return 1
  "$EVAL_PY" - "$run/aggregate.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
assert len(payload.get("per_task", {})) == 16
assert sum(payload.get("errors", {}).values()) == 0
PY
}

summarize() {
  # shellcheck disable=SC2086
  "$EVAL_PY" "$SCRIPT_DIR/summarize_checkpoint_curve.py" "$OUT_ROOT" \
    --checkpoints $CHECKPOINTS --seed "$SEED"
}

for checkpoint in $CHECKPOINTS; do
  checkpoint_dir=$TRAIN_RUN/$checkpoint
  if [[ ! -f "$checkpoint_dir/_CHECKPOINT_METADATA" ]]; then
    log "ERROR checkpoint is missing or incomplete: $checkpoint_dir"
    exit 1
  fi
done

cat > "$OUT_ROOT/MANIFEST.txt" <<EOF
created=$(date -Is)
root=$ROOT
policy_repo=$POLICY_REPO
train_run=$TRAIN_RUN
checkpoints=$CHECKPOINTS
seed=$SEED
reuse_root=$REUSE_ROOT
preset=run_8gpu_fastest.sh
num_gpus=$NUM_GPUS
servers=$NUM_GPUS
clients_per_gpu=$CLIENTS_PER_GPU
env_workers=$((NUM_GPUS * CLIENTS_PER_GPU))
cpu_affinity=$CPU_AFFINITY
jax_cache_dir=$JAX_CACHE_DIR
max_batch_size=1
max_wait_ms=0
video_mode=off
sharding=balanced_steal_v1
EOF

child_pid=
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# Materialize every already-complete point before launching missing runs. This
# keeps the partial curve useful even if a long sweep is interrupted.
for checkpoint in $CHECKPOINTS; do
  run=$OUT_ROOT/ckpt$checkpoint/seed$SEED
  if is_complete "$run"; then
    continue
  fi

  reusable=$REUSE_ROOT/ckpt$checkpoint/seed$SEED
  if is_complete "$reusable"; then
    if [[ -e "$run" || -L "$run" ]]; then
      archived=${run}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)
      mv "$run" "$archived"
      log "ARCHIVE incomplete run: $run -> $archived"
    fi
    mkdir -p "$(dirname "$run")"
    ln -s "$reusable" "$run"
    log "REUSE ckpt=$checkpoint seed=$SEED from $reusable"
  fi
done

summarize
log "BEGIN seed-$SEED checkpoint curve; checkpoints=[$CHECKPOINTS]"
for checkpoint in $CHECKPOINTS; do
  run=$OUT_ROOT/ckpt$checkpoint/seed$SEED
  if is_complete "$run"; then
    log "SKIP ckpt=$checkpoint seed=$SEED (already complete)"
    continue
  fi

  if [[ -e "$run" || -L "$run" ]]; then
    archived=${run}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)
    mv "$run" "$archived"
    log "ARCHIVE incomplete run: $run -> $archived"
  fi
  mkdir -p "$(dirname "$run")"
  log "START ckpt=$checkpoint seed=$SEED out=$run"
  started=$(date +%s)
  env \
    CKPT="$TRAIN_RUN/$checkpoint" \
    CKPT_ID="$checkpoint" \
    SEED="$SEED" \
    OUT="$run" \
    BASE_PORT="$BASE_PORT" \
    NUM_GPUS="$NUM_GPUS" \
    CLIENTS_PER_GPU="$CLIENTS_PER_GPU" \
    CPU_AFFINITY="$CPU_AFFINITY" \
    JAX_CACHE_DIR="$JAX_CACHE_DIR" \
    SERVER_READY_TIMEOUT_SECONDS="$SERVER_READY_TIMEOUT_SECONDS" \
    bash "$SCRIPT_DIR/run_8gpu_fastest.sh" \
    >> "$MASTER_LOG" 2>&1 &
  child_pid=$!
  rc=0
  wait "$child_pid" || rc=$?
  child_pid=
  elapsed=$(( $(date +%s) - started ))
  if (( rc != 0 )) || ! is_complete "$run"; then
    log "FAILED ckpt=$checkpoint seed=$SEED rc=$rc elapsed=${elapsed}s; stopping curve"
    summarize || true
    exit 1
  fi
  log "DONE ckpt=$checkpoint seed=$SEED elapsed=${elapsed}s"
  summarize
done

trap - INT TERM EXIT
summarize
log "COMPLETE seed-$SEED curve; plot=$OUT_ROOT/sr_vs_step.png"
