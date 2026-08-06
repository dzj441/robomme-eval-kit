#!/usr/bin/env bash
# Download the preprocessed RoboMME training data (pickles + cached SigLIP2
# features). Training needs both: the paper freezes the vision backbone and
# reads precomputed visual tokens, which is what makes 80k steps tractable.
#
# No proxy: huggingface.co is directly reachable here and faster than the proxy.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

HF=/home/qid/dzj/miniconda3/envs/robomem/bin/hf
DST=/datadrive1/dzj/RoboMME/preprocessed
LOG=/datadrive1/dzj/RoboMME/download_preprocessed.log
REPO=Yinpei/robomme_preprocessed_data

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

mkdir -p "$DST"
log "=== start (pid $$) -> $DST ==="

free_gb=$(df -BG --output=avail "$DST" | tail -1 | tr -dc '0-9')
if (( free_gb < 300 )); then
  log "ABORT: only ${free_gb}G free, want >= 300G"
  exit 1
fi
log "${free_gb}G free"

ok=0
for attempt in 1 2 3 4 5; do
  if "$HF" download "$REPO" --repo-type dataset --local-dir "$DST" >>"$LOG" 2>&1; then
    ok=1; break
  fi
  log "RETRY (attempt $attempt failed)"
  sleep 60
done

if (( ok )); then
  log "DONE -> $(du -sh "$DST" | cut -f1)"
  log "note: data.zip / features.zip still need unzipping before training"
else
  log "FAILED after 5 attempts"
  exit 1
fi
