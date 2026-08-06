#!/usr/bin/env bash
# Smoke-test each memory intervention on one GPU before committing 96 minutes to it.
#
# budget128 gets VideoPlaceOrder deliberately: it is the task whose 1107-frame
# history already asks for ~5.7 GB in a single allocation, so if quadrupling the
# memory sequence blows up anywhere it blows up there. The rest get VideoUnmask,
# which is short and memory-critical.
set -uo pipefail
ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/smoke_interventions.log
: > "$LOG"

run_one() {
  local mode=$1 gpu=$2 port=$3 task=$4
  OUT=$ROOT/eval_out/smoke_$mode
  rm -rf "$OUT"
  ROBOMME_MEM_MODE=$mode SERVE_SCRIPT=$ROOT/serve_policy_mem.py \
    ONLY_TASKS="$task" EPISODES=3 \
    OUT="$OUT" SEED=7 NUM_SHARDS=1 NUM_GPUS=1 BASE_PORT=$port \
    MEM_FRACTION=0.95 LP_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=$gpu \
    bash "$ROOT/run_eval_episode_sharded.sh" > "$ROOT/smoke_$mode.log" 2>&1
  local eps
  eps=$(find "$OUT" -name "*.mp4" 2>/dev/null | wc -l)
  local oom
  oom=$(grep -c RESOURCE_EXHAUSTED "$OUT"/logs/server0.log 2>/dev/null || echo 0)
  local hit
  hit=$(grep -c "mem_intervention. mode=$mode" "$OUT"/logs/server0.log 2>/dev/null || echo 0)
  echo "[$mode] task=$task episodes=$eps patch_engaged=$hit oom=$oom" >> "$LOG"
}

run_one off       0 9100 VideoUnmask      &
run_one timeshuf  1 9200 VideoUnmask      &
run_one primacy   2 9300 VideoUnmask      &
run_one budget128 3 9400 VideoPlaceOrder  &
wait
echo "=== smoke done ===" >> "$LOG"
cat "$LOG"
