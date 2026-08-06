#!/usr/bin/env bash
# Episode-level sharded RoboMME evaluation.
#
# Every shard takes all 16 tasks but only the episodes with
# `episode_id % NUM_SHARDS == shard`, so the task mix — and therefore the cost —
# is identical across shards. Task-level sharding could never beat the slowest
# single task (VideoPlaceOrder alone runs ~28 min for its 50 episodes, and it
# was still running while 12 of 16 shards sat idle).
#
# Rendering goes through Mesa lavapipe: this host faults in the NVIDIA driver's
# graphics path (412 GPFs, three crashes), while software Vulkan has now run
# clean. The policy servers still use CUDA compute, which was never implicated.
set -uo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

POLICY_REPO=/home/qid/dzj/RoboMME_policy
ROOT=/datadrive1/dzj/RoboMME
CKPT=${CKPT:-$ROOT/ckpt/perceptual-framesamp-modul/home/daiyp/MME-VLA-Suite/runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999}
OUT=${OUT:-$ROOT/eval_out/ultra_seed7}
SEED=${SEED:-7}
CKPT_ID=${CKPT_ID:-79999}
NUM_SHARDS=${NUM_SHARDS:-16}
NUM_GPUS=${NUM_GPUS:-4}
BASE_PORT=${BASE_PORT:-8500}
LP_NUM_THREADS=${LP_NUM_THREADS:-4}
# Each shard needs its own server (memory is per-session), so server count == shard
# count and GPU memory caps parallelism. Measured: the default pool takes 16.9 GB,
# capping us at 18; 0.14 holds the same model in 11.8 GB and lifts that to 27, which
# moves the ceiling onto CPU (96 cores / LP_NUM_THREADS).
MEM_FRACTION=${MEM_FRACTION:-0.14}
POLICY_NAME=framesamp-modul
# Memory interventions swap in a wrapper that patches MemoryBuffer before
# serve_policy.py runs; unset, this is the stock server and stock code path.
SERVE_SCRIPT=${SERVE_SCRIPT:-$POLICY_REPO/scripts/serve_policy.py}
MEM_MODE=${ROBOMME_MEM_MODE:-baseline}
EPISODES=${EPISODES:-50}

SERVER_PY=/home/qid/dzj/miniconda3/envs/robomme-vla/bin/python
EVAL_PY=/home/qid/dzj/miniconda3/envs/robomme/bin/python
SP=/home/qid/dzj/miniconda3/envs/robomme-vla/lib/python3.11/site-packages
# system ldconfig shadows the pip nvJitLink with CUDA 12.1's -> jax silently on CPU
LDP="$(find $SP/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')"

ALL_TASKS=${ONLY_TASKS:-"BinFill,StopCube,PickXtimes,SwingXtimes,VideoUnmask,ButtonUnmask,VideoUnmaskSwap,ButtonUnmaskSwap,PickHighlight,VideoRepick,VideoPlaceButton,VideoPlaceOrder,MoveCube,InsertPeg,PatternLock,RouteStick"}

mkdir -p "$OUT/logs"
echo "OUT=$OUT seed=$SEED shards=$NUM_SHARDS lp_threads=$LP_NUM_THREADS mem_mode=$MEM_MODE episodes=$EPISODES" | tee "$OUT/run_config.txt"

echo "seeding per-shard progress.json..." | tee -a "$OUT/run_config.txt"
"$EVAL_PY" "$ROOT/seed_episode_shards.py" --out "$OUT" --num-shards "$NUM_SHARDS" --episodes "$EPISODES" \
  --policy-name "$POLICY_NAME" --ckpt-id "$CKPT_ID" --seed "$SEED" | tee -a "$OUT/run_config.txt"

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=$((i % NUM_GPUS))
  PORT=$((BASE_PORT + i))
  SDIR="$OUT/shard$i"

  (
    cd "$POLICY_REPO"     # get_history_config() resolves its yaml relative to CWD
    CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH="$LDP" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
      ROBOMME_MEM_MODE="$MEM_MODE" \
      "$SERVER_PY" "$SERVE_SCRIPT" \
      --seed=$SEED --port=$PORT policy:checkpoint \
      --policy.dir="$CKPT" --policy.config=mme_vla_suite \
      > "$OUT/logs/server$i.log" 2>&1 &
    SRV=$!
    trap 'kill $SRV 2>/dev/null' EXIT

    for t in $(seq 1 180); do
      sleep 5
      "$EVAL_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
try: s.connect(('127.0.0.1',$PORT)); sys.exit(0)
except Exception: sys.exit(1)
finally: s.close()" 2>/dev/null && break
      kill -0 $SRV 2>/dev/null || { echo "server$i died"; tail -20 "$OUT/logs/server$i.log"; exit 1; }
    done
    echo "[shard$i] server ready"

    cd "$POLICY_REPO/examples/robomme"
    env ROBOMME_USE_LAVAPIPE=1 LP_NUM_THREADS="$LP_NUM_THREADS" \
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$POLICY_REPO/examples/robomme" \
      "$EVAL_PY" "$ROOT/eval_lavapipe.py" \
        --args.host=127.0.0.1 --args.port=$PORT --args.model_seed=$SEED \
        --args.policy_name="$POLICY_NAME" --args.model_ckpt_id=$CKPT_ID \
        --args.only_tasks="$ALL_TASKS" --args.save_dir="$SDIR" \
        > "$OUT/logs/eval$i.log" 2>&1
    echo "[shard$i] eval rc=$?"
    kill $SRV 2>/dev/null
  ) &
  PIDS+=($!)
done

echo "launched ${#PIDS[@]} shards"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
echo "=== done (fail=$FAIL) ==="
"$EVAL_PY" "$ROOT/merge_robomme.py" "$OUT" | tee "$OUT/FINAL_REPORT.txt"
