#!/usr/bin/env bash
# The control the intervention batch was missing.
#
# All four interventions ran through serve_policy_mem.py while the baseline ran
# the stock serve_policy.py, and all four landed at 15-20 AVG against the
# baseline's 46.5. A side effect of the wrapper itself -- the sys.path insert,
# the runpy re-entry -- would produce exactly that pattern, so it has to be ruled
# out before any of those numbers mean anything.
#
# ROBOMME_MEM_MODE=baseline makes mem_intervention.apply() return before it
# patches anything, so this is the stock code path reached through the wrapper.
# Restricted to the four Counting tasks, which fell the furthest (69.5 -> ~20)
# and so give the sharpest read on whether the wrapper is implicated.
set -uo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/run_interventions.log
OUT=$ROOT/eval_out/interv_wrapctl

log() { echo "[$(date -Is)] [wrapctl] $*" | tee -a "$LOG"; }

n_fin() { local c; c=$(grep -c "=== finished ===" "$LOG" 2>/dev/null | head -1); echo "${c:-0}"; }
want=$(( $(n_fin) + 1 ))
log "waiting for the running queue to finish (marker #$want)"
while (( $(n_fin) < want )); do sleep 60; done

log "BEGIN wrapper control (baseline mode, Counting suite)"
t0=$(date +%s)
ROBOMME_MEM_MODE=baseline SERVE_SCRIPT=$ROOT/serve_policy_mem.py \
  ONLY_TASKS="BinFill,StopCube,PickXtimes,SwingXtimes" \
  OUT="$OUT" SEED=7 NUM_SHARDS=16 MEM_FRACTION=0.95 \
  BASE_PORT=9500 LP_NUM_THREADS=4 \
  bash "$ROOT/run_eval_episode_sharded.sh" >> "$ROOT/interv_wrapctl.log" 2>&1
log "DONE wrapper control in $(( ($(date +%s) - t0) / 60 ))m ($(find "$OUT" -name '*.mp4' | wc -l) eps)"

/home/qid/dzj/miniconda3/envs/robomme/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
run = pathlib.Path("/datadrive1/dzj/RoboMME/eval_out/interv_wrapctl")
base = {"BinFill": 46.0, "StopCube": 44.0, "PickXtimes": 94.0, "SwingXtimes": 94.0}
off  = {"BinFill": 24.0, "StopCube": 8.0, "PickXtimes": 28.0, "SwingXtimes": 28.0}
shards = sorted(run.glob("shard*/**/progress.json"),
                key=lambda p: int(str(p).split("shard")[1].split("/")[0]))
n = len(shards)
owned = {}
for prog in shards:
    sh = int(str(prog).split("shard")[1].split("/")[0])
    for task, eps in json.loads(prog.read_text()).items():
        for ep, v in eps.items():
            if int(ep) % n == sh:
                owned.setdefault(task, {})[ep] = v
print(f"\n{'task':<14}{'stock base':>11}{'via wrapper':>13}{'off':>8}")
for t in base:
    e = owned.get(t, {})
    if not e:
        continue
    v = 100.0 * sum(1 for x in e.values() if x is True) / len(e)
    print(f"{t:<14}{base[t]:11.1f}{v:13.1f}{off[t]:8.1f}")
print("\nIf the wrapper column tracks the stock baseline, the wrapper is clean and\n"
      "the intervention drops are real. If it tracks `off`, they are an artifact.")
PY
