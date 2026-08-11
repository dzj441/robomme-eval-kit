#!/usr/bin/env bash
# Evaluate the two failure-driven memory experiments sequentially on seed 7.
#
# Defaults to deterministic parity.  Use PRESET=fastest for the lower-latency
# validated configuration.  A completed run is skipped so the script is safe to
# resume after an interruption.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}
POLICY_REPO=${POLICY_REPO:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/.worktrees/RoboMME_policy-memory-experiments}
TRAIN_ROOT=${TRAIN_ROOT:-/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_training/training_runs/ckpts/mme_vla_suite}
OUT_ROOT=${OUT_ROOT:-$ROOT/eval_out/memory_explorations_seed7}
SEED=${SEED:-7}
PRESET=${PRESET:-deterministic}
STEP=${STEP:-79999}

EXPERIMENTS=${EXPERIMENTS:-"perceptual-anchor-recent-modul_8h100_b64_fastio_seed42 perceptual-temporal-state-modul_8h100_b64_fastio_seed42"}

case "$PRESET" in
  deterministic)
    RUNNER=$SCRIPT_DIR/run_8gpu_deterministic_parity.sh
    ;;
  fastest)
    RUNNER=$SCRIPT_DIR/run_8gpu_fastest.sh
    ;;
  *)
    echo "PRESET must be 'deterministic' or 'fastest', got: $PRESET" >&2
    exit 2
    ;;
esac

if [[ ! -d "$POLICY_REPO/.git" && ! -f "$POLICY_REPO/.git" ]]; then
  echo "Policy worktree is missing: $POLICY_REPO" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
SUMMARY=$OUT_ROOT/summary.tsv
if [[ ! -f "$SUMMARY" ]]; then
  printf 'experiment\tstep\tseed\tpreset\tstatus\telapsed_s\toutput\n' > "$SUMMARY"
fi

is_complete() {
  local output=$1
  [[ -f "$output/aggregate.json" ]] &&
    [[ -f "$output/FINAL_REPORT.txt" ]] &&
    [[ -f "$output/TIMING.txt" ]] &&
    grep -qx 'status=complete' "$output/TIMING.txt"
}

for experiment in $EXPERIMENTS; do
  checkpoint=$TRAIN_ROOT/$experiment/$STEP
  output=$OUT_ROOT/$experiment/step${STEP}_${PRESET}_seed${SEED}

  if [[ ! -f "$checkpoint/_CHECKPOINT_METADATA" ]]; then
    echo "Checkpoint is missing or incomplete: $checkpoint" >&2
    exit 3
  fi
  if [[ ! -f "$TRAIN_ROOT/$experiment/history_config.txt" ]]; then
    echo "Checkpoint history_config.txt is missing: $TRAIN_ROOT/$experiment/history_config.txt" >&2
    exit 3
  fi

  if is_complete "$output"; then
    echo "SKIP complete: $experiment step=$STEP seed=$SEED"
    continue
  fi

  started=$(date +%s)
  echo "START $experiment step=$STEP seed=$SEED preset=$PRESET"
  set +e
  ROOT="$ROOT" \
  POLICY_REPO="$POLICY_REPO" \
  CKPT="$checkpoint" \
  CKPT_ID="$STEP" \
  OUT="$output" \
  SEED="$SEED" \
  NUM_GPUS=8 \
    bash "$RUNNER"
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - started ))

  if (( rc == 0 )) && is_complete "$output"; then
    status=complete
  else
    status=failed
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment" "$STEP" "$SEED" "$PRESET" "$status" "$elapsed" "$output" >> "$SUMMARY"

  if [[ "$status" != complete ]]; then
    echo "FAILED $experiment rc=$rc elapsed=${elapsed}s output=$output" >&2
    exit "${rc:-1}"
  fi
  echo "DONE $experiment elapsed=${elapsed}s output=$output"
done

echo "All requested memory experiments are complete. Summary: $SUMMARY"
