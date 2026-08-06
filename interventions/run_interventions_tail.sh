#!/usr/bin/env bash
# Second half of the intervention queue, chained behind the first.
#
# `primacy` was pre-marked done so the running queue skips it: with the coverage
# analysis showing uniform sampling misses no subgoal segment at all, spending
# the budget differently is the weaker hypothesis, and `budget8` -- which
# completes the 0/8/32/128 dose-response curve -- is worth the slot more. The
# last condition in a queue is the one that gets cut if time runs out, so the
# order here is the order of what we would least mind losing.
#
# Chaining waits on the first queue's own end-of-run log line rather than on a
# process match: several shells in this session carry "run_interventions.sh" in
# their command line, so pgrep would wait forever on a wrapper that never exits.
set -uo pipefail
ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/run_interventions.log

log() { echo "[$(date -Is)] [tail] $*" | tee -a "$LOG"; }

# grep -c prints 0 *and* exits 1 on no match, so a `|| echo 0` fallback would
# yield "0\n0" and break the arithmetic.
n_fin() { local c; c=$(grep -c "=== finished ===" "$LOG" 2>/dev/null | head -1); echo "${c:-0}"; }

want=$(( $(n_fin) + 1 ))
log "waiting for finished-marker #$want"
while (( $(n_fin) < want )); do
  sleep 60
done
log "first queue finished; starting budget8 then primacy"

rm -f "$ROOT/eval_out/interv_primacy/.done"
MODES="budget8 primacy" bash "$ROOT/run_interventions.sh"
