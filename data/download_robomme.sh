#!/usr/bin/env bash
# Download the full RoboMME release: 16 task datasets (~56 GB of .tar.xz) plus
# the 10 MME-VLA checkpoint variants that actually carry weights (~119 GB).
#
# No proxy: huggingface.co is directly reachable here at 11-19 MB/s.
# Idempotent: hf skips files it already has, and a .done marker per repo lets
# systemd's Restart=on-failure supervise without redoing finished work.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

HF=/home/qid/dzj/miniconda3/envs/robomem/bin/hf
ROOT=/datadrive1/dzj/RoboMME
LOG=$ROOT/download.log
MIN_FREE_GB=400

# repo : repo_type : local subdir
REPOS=(
  "Yinpei/robomme_data_h5:dataset:dataset"
  "Yinpei/mme_vla_suite:model:ckpt"
)

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p "$ROOT"
log "=== robomme download start (pid $$) ==="

failed=0
for entry in "${REPOS[@]}"; do
  IFS=: read -r repo rtype sub <<< "$entry"
  marker="$ROOT/.$sub.done"
  if [[ -f "$marker" ]]; then
    log "SKIP $repo (already complete)"
    continue
  fi

  free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    log "ABORT: only ${free_gb}G free, need >= ${MIN_FREE_GB}G"
    exit 1
  fi

  log "BEGIN $repo -> $ROOT/$sub (${free_gb}G free)"
  ok=0
  for attempt in 1 2 3 4 5; do
    if "$HF" download "$repo" --repo-type "$rtype" --local-dir "$ROOT/$sub" >>"$LOG" 2>&1; then
      ok=1; break
    fi
    log "RETRY $repo (attempt $attempt failed)"
    sleep 60
  done

  if (( ok )); then
    date -Is > "$marker"
    log "DONE $repo -> $(du -sh "$ROOT/$sub" | cut -f1)"
  else
    log "FAILED $repo after 5 attempts"
    failed=1
  fi
done

(( failed )) && { log "=== exiting non-zero; systemd will retry ==="; exit 1; }
log "=== all complete: $(du -sh "$ROOT" | cut -f1) ==="
