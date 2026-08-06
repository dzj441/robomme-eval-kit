#!/usr/bin/env bash
# Four test-time memory interventions on the released FrameSamp+Modul checkpoint,
# each a full 16-task / 800-episode run at seed 7 so it lines up directly with
# eval_out/ultra3_seed7.
#
# Order is deliberate. `off` comes first because it fixes the scale for
# everything after it: if memory is worth only a couple of points under these
# weights, no result about *which* frames it holds means much. `timeshuf` then
# asks whether the ordering is used at all, and only then do the two
# "spend the budget differently" conditions run.
#
# 16 shards / 4 per GPU, no MEM_FRACTION cap -- VideoPlaceOrder's add_buffer
# peaks near 20 GB and OOMed both the 20- and 24-shard attempts.
set -uo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/run_interventions.log
MODES=${MODES:-"off timeshuf budget128 primacy"}
SEED=7

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== interventions: $MODES ==="
idx=0
for mode in $MODES; do
  idx=$((idx + 1))
  OUT=$ROOT/eval_out/interv_$mode
  if [[ -f "$OUT/.done" ]]; then
    log "SKIP $mode (already complete)"
    continue
  fi
  log "BEGIN $mode"
  t0=$(date +%s)
  ROBOMME_MEM_MODE=$mode SERVE_SCRIPT=$ROOT/serve_policy_mem.py \
    OUT="$OUT" SEED=$SEED NUM_SHARDS=16 MEM_FRACTION=0.95 \
    BASE_PORT=$((8900 + idx * 40)) LP_NUM_THREADS=4 \
    bash "$ROOT/run_eval_episode_sharded.sh" >> "$ROOT/interv_$mode.log" 2>&1
  dt=$(( $(date +%s) - t0 ))

  # A shard whose server OOMs dies quietly and eval.py abandons its remaining
  # tasks, so trust the episode count rather than the exit code. Also confirm the
  # patch actually engaged -- a silently-unpatched run would look like a clean
  # reproduction of the baseline and be badly misleading.
  eps=$(find "$OUT" -name "*.mp4" 2>/dev/null | wc -l)
  ooms=$(grep -l RESOURCE_EXHAUSTED "$OUT"/logs/server*.log 2>/dev/null | wc -l)
  patched=$(grep -l "mem_intervention" "$OUT"/logs/server*.log 2>/dev/null | wc -l)
  if (( eps >= 800 && ooms == 0 && patched == 16 )); then
    date -Is > "$OUT/.done"
    log "DONE $mode in $((dt / 60))m ($eps eps, patch on all $patched servers)"
  else
    log "INCOMPLETE $mode: $eps/800 eps, $ooms OOM, patched=$patched/16 — continuing to next mode"
  fi
done

log "=== aggregating ==="
/home/qid/dzj/miniconda3/envs/robomme/bin/python "$ROOT/merge_interventions.py" \
  2>&1 | tee "$ROOT/eval_out/INTERVENTIONS.txt"
log "=== finished ==="
