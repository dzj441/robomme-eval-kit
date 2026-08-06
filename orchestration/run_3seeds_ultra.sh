#!/usr/bin/env bash
# Task 3: three-seed RoboMME baseline on the ultra-fast configuration.
#
#   20 shards, episode-level (ep % 20 == shard), lavapipe software Vulkan.
#
# Shard count is capped by GPU memory, not CPU: memory is per-session so every
# shard needs its own policy server, and shards land on GPUs round-robin, so the
# binding constraint is servers-per-GPU. Measured: a server needs ~15 GB to
# survive VideoPlaceButton's 700-frame add_buffer (11.8 GB was enough to load the
# model but OOMed on episode 2), and 5 x 15.2 GB = 76 GB fits an 80 GB card.
# Six per card would need 90 GB — that is exactly why the 24-shard attempt lost
# 22 of its shards mid-run.
#
# Rendering stays on lavapipe: the NVIDIA graphics path faulted this host 412
# times and crashed it three times, while software Vulkan has run clean since.
set -uo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/run_3seeds_ultra.log
# 16 shards / 4 per GPU is the hard ceiling: VideoPlaceOrder's 1107-frame
# add_buffer asks for 5.7 GB in one allocation on top of an 11.8 GB resident
# model, so a server needs ~20 GB. Five per card leaves only 16 GB each and OOMs
# on exactly that task -- which is what killed the 20- and 24-shard attempts.
# No MEM_FRACTION cap: the pool must be free to grow to that peak.
NUM_SHARDS=${NUM_SHARDS:-16}
MEM_FRACTION=${MEM_FRACTION:-0.95}

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== 3-seed ultra run: ${NUM_SHARDS} shards, mem_fraction=${MEM_FRACTION} ==="
for s in 7 17 27; do
  OUT=$ROOT/eval_out/ultra3_seed$s
  if [[ -f "$OUT/.done" ]]; then
    log "SKIP seed $s (already complete)"
    continue
  fi
  log "BEGIN seed $s"
  t0=$(date +%s)
  OUT="$OUT" SEED=$s NUM_SHARDS=$NUM_SHARDS MEM_FRACTION=$MEM_FRACTION \
    BASE_PORT=$((8700 + s * 30)) LP_NUM_THREADS=4 \
    bash "$ROOT/run_eval_episode_sharded.sh" >> "$ROOT/ultra3_seed$s.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  # A shard whose server OOMs dies silently and eval.py abandons the rest of its
  # tasks, so count episodes rather than trusting the exit code.
  eps=$(find "$OUT" -name "*.mp4" 2>/dev/null | wc -l)
  ooms=$(grep -l RESOURCE_EXHAUSTED "$OUT"/logs/server*.log 2>/dev/null | wc -l)
  if (( eps >= 800 && ooms == 0 )); then
    date -Is > "$OUT/.done"
    log "DONE seed $s in $((dt / 60))m ($eps eps)"
  else
    log "INCOMPLETE seed $s: $eps/800 eps, $ooms servers OOMed, rc=$rc — stopping"
    log "  (raise MEM_FRACTION or lower NUM_SHARDS before retrying)"
    break
  fi
done

log "=== aggregating ==="
/home/qid/dzj/miniconda3/envs/robomme/bin/python "$ROOT/merge_seeds.py" "$ROOT/eval_out" \
  2>&1 | tee "$ROOT/eval_out/BASELINE_3SEED.txt"
