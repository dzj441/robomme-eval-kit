#!/usr/bin/env bash
# Evaluate three checkpoints at three model seeds with the fastest validated
# 8-GPU stateless configuration. Runs are intentionally sequential because
# each one occupies all eight GPUs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-stateless-batched}
TRAIN_RUN=${TRAIN_RUN:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME_policy/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul_8h100_b64_seed42}
OUT_ROOT=${OUT_ROOT:-$ROOT/eval_out/h100_b64_seed42_3ckpt_3seed_fastest}
CHECKPOINTS=${CHECKPOINTS:-"79999 70000 60000"}
SEEDS=${SEEDS:-"7 17 27"}
POLICY_NAME=${POLICY_NAME:-h100-b64-seed42}
BASE_PORT=${BASE_PORT:-8600}

export ROOT POLICY_REPO POLICY_NAME
mkdir -p "$OUT_ROOT"
MASTER_LOG=$OUT_ROOT/master.log

log() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

ports_are_free() {
  python - "$BASE_PORT" 8 <<'PY'
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
except OSError:
    raise SystemExit(1)
finally:
    for sock in sockets:
        sock.close()
PY
}

wait_for_ports_free() {
  local deadline=$((SECONDS + 45))
  while (( SECONDS < deadline )); do
    if ports_are_free; then
      return 0
    fi
    sleep 1
  done
  return 1
}

is_complete() {
  local run=$1
  local expected_checkpoint=$2
  local expected_seed=$3
  [[ -f "$run/aggregate.json" && -f "$run/TIMING.txt" ]] || return 1
  grep -q '^status=complete$' "$run/TIMING.txt" || return 1
  for gpu in $(seq 0 7); do
    [[ -f "$run/logs/server_gpu${gpu}.log" ]] || return 1
    [[ -f "$run/logs/server_gpu${gpu}.metadata.json" ]] || return 1
    grep -q 'server listening on' "$run/logs/server_gpu${gpu}.log" || return 1
    if grep -Eq 'Traceback|address already in use|OSError: \[Errno 98\]' \
      "$run/logs/server_gpu${gpu}.log"; then
      return 1
    fi
  done
  python - "$run" "$expected_checkpoint" "$expected_seed" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected_checkpoint = sys.argv[2]
expected_seed = sys.argv[3]
payload = json.loads((run / "aggregate.json").read_text())
assert len(payload.get("per_task", {})) == 16
assert sum(payload.get("errors", {}).values()) == 0

timing = {}
for line in (run / "TIMING.txt").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        timing[key] = value
assert timing.get("ckpt_id") == expected_checkpoint
assert timing.get("seed") == expected_seed
fingerprint = timing["expected_model_fingerprint"]
for gpu in range(8):
    metadata = json.loads(
        (run / "logs" / f"server_gpu{gpu}.metadata.json").read_text()
    )
    assert metadata["protocol_version"] == 2
    assert metadata["serving_mode"] == "stateless-batched"
    assert metadata["model_fingerprint"] == fingerprint
PY
}

summarize() {
  # shellcheck disable=SC2086
  python "$SCRIPT_DIR/summarize_eval_grid.py" "$OUT_ROOT" \
    --checkpoints $CHECKPOINTS --seeds $SEEDS \
    > "$OUT_ROOT/SUMMARY.stdout.txt"
  cat "$OUT_ROOT/SUMMARY.stdout.txt"
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
seeds=$SEEDS
preset=run_8gpu_fastest.sh
num_gpus=8
servers=8
env_workers=32
max_batch_size=1
max_wait_ms=0
video_mode=off
sharding=balanced_steal_v1
lifecycle_guard=process_groups_ports_and_fingerprint_v1
EOF

child_pid=
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    deadline=$((SECONDS + 40))
    while kill -0 "$child_pid" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 0.5
    done
    if kill -0 "$child_pid" 2>/dev/null; then
      kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if ! ports_are_free; then
  log "ERROR required ports ${BASE_PORT}-$((BASE_PORT + 7)) are already occupied"
  exit 1
fi

log "BEGIN 3 checkpoints x 3 seeds; checkpoints=[$CHECKPOINTS], seeds=[$SEEDS]"
for checkpoint in $CHECKPOINTS; do
  for seed in $SEEDS; do
    run=$OUT_ROOT/ckpt$checkpoint/seed$seed
    if is_complete "$run" "$checkpoint" "$seed"; then
      log "SKIP ckpt=$checkpoint seed=$seed (already complete)"
      continue
    fi
    if [[ -e "$run" ]]; then
      archived=${run}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)
      mv "$run" "$archived"
      log "ARCHIVE incomplete run: $run -> $archived"
    fi
    mkdir -p "$(dirname "$run")"
    if ! ports_are_free; then
      log "FAILED ports ${BASE_PORT}-$((BASE_PORT + 7)) are occupied before ckpt=$checkpoint seed=$seed"
      exit 1
    fi
    log "START ckpt=$checkpoint seed=$seed out=$run"
    started=$(date +%s)
    env \
      CKPT="$TRAIN_RUN/$checkpoint" \
      CKPT_ID="$checkpoint" \
      SEED="$seed" \
      OUT="$run" \
      BASE_PORT="$BASE_PORT" \
      bash "$SCRIPT_DIR/run_8gpu_fastest.sh" \
      >> "$MASTER_LOG" 2>&1 &
    child_pid=$!
    rc=0
    wait "$child_pid" || rc=$?
    child_pid=
    elapsed=$(( $(date +%s) - started ))
    ports_rc=0
    wait_for_ports_free || ports_rc=$?
    if (( rc != 0 || ports_rc != 0 )) || ! is_complete "$run" "$checkpoint" "$seed"; then
      log "FAILED ckpt=$checkpoint seed=$seed rc=$rc ports_rc=$ports_rc elapsed=${elapsed}s; stopping grid"
      summarize || true
      exit 1
    fi
    log "DONE ckpt=$checkpoint seed=$seed elapsed=${elapsed}s"
    summarize
  done
done

trap - INT TERM EXIT
summarize
log "COMPLETE all 9 runs; summary=$OUT_ROOT/SUMMARY.md"
