#!/usr/bin/env bash
# Unpack the preprocessed RoboMME training data in place.
#
# The release ships each split as numbered zip parts (data/part_N.zip,
# features/part_N.zip) rather than one archive, so they are independent and can
# be extracted in parallel. Four at a time keeps the disk busy without thrashing
# it; the payload is ~360 GB in and roughly the same again out.
set -uo pipefail

ROOT=/datadrive1/dzj/RoboMME/preprocessed
LOG=/datadrive1/dzj/RoboMME/unzip_preprocessed.log

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "=== start (pid $$) ==="
for split in data features memer qwenvl; do
  d="$ROOT/$split"
  [[ -d $d ]] || continue
  mapfile -t zips < <(find "$d" -maxdepth 1 -name '*.zip' | sort)
  (( ${#zips[@]} )) || { log "$split: no zips"; continue; }
  log "$split: ${#zips[@]} archives -> $d/extracted"
  mkdir -p "$d/extracted"
  printf '%s\n' "${zips[@]}" \
    | xargs -P 4 -I{} sh -c 'unzip -n -q "$1" -d "$2/extracted" || echo "FAILED $1"' _ {} "$d" \
    >>"$LOG" 2>&1
  log "$split: done, $(du -sh "$d/extracted" | cut -f1)"
done
log "=== finished ==="
