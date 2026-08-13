#!/usr/bin/env bash
# Four-GPU stateless evaluation entry for nodes whose kernel driver is exactly
# 550.163.01.  The rendering ABI is intentionally fixed in this branch.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

readonly EXPECTED_DRIVER=550.163.01
readonly FIXED_DRIVER_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/dzj/robomme_runtime/nvidia/550.163.01

mapfile -t driver_versions < <(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
)
if (( ${#driver_versions[@]} != 1 )) || [[ "${driver_versions[0]}" != "$EXPECTED_DRIVER" ]]; then
  echo "driver mismatch: this launcher requires $EXPECTED_DRIVER; found: ${driver_versions[*]:-none}" >&2
  exit 1
fi

visible_gpus=$(nvidia-smi -L | wc -l)
if (( visible_gpus < 4 )); then
  echo "this launcher requires at least four visible GPUs; found $visible_gpus" >&2
  exit 1
fi

for required_path in \
  "$FIXED_DRIVER_ROOT/runtime-libs" \
  "$FIXED_DRIVER_ROOT/nvidia_icd.local.json" \
  "$FIXED_DRIVER_ROOT/10_nvidia.local.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing fixed NVIDIA rendering runtime asset: $required_path" >&2
    exit 1
  fi
done

# These values deliberately cannot be changed through the caller environment.
export NVIDIA_DRIVER_VERSION=$EXPECTED_DRIVER
export NVIDIA_DRIVER_ROOT=$FIXED_DRIVER_ROOT
export NUM_GPUS=4
export CLIENTS_PER_GPU=4
export CPU_AFFINITY=none

# Cache contents are specific to this driver/GPU execution environment.
export JAX_CACHE_DIR=${JAX_CACHE_DIR:-${ROOT:-/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME}/eval_out/.jax_compilation_cache_550_4gpu}

exec bash "$SCRIPT_DIR/run_8gpu_fastest.sh" "$@"
