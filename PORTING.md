# RoboMME eval kit — porting notes (target: 8×4090 box)

Everything needed to reproduce our RoboMME evaluation and the test-time memory
interventions on a fresh machine. Evaluation needs **no datasets** — tasks are
procedural (cubes/buttons/pegs), demos are generated in-env per seed, and the
policy's memory is encoded live from simulator frames. Verified: our full
3-seed baseline ran before any training data was extracted on the source box.

## What to fetch

| thing | where | size |
|---|---|---|
| benchmark code | `github.com/dzj441/robomme_benchmark` (private mirror of upstream `03a8faf`) | — |
| policy code | `github.com/dzj441/robomme_policy_learning` (upstream `ecf086c` + one dtype fix `67f594f`) | — |
| this kit | `github.com/dzj441/robomme-eval-kit` | — |
| checkpoint `perceptual-framesamp-modul/79999` | HF `Yinpei/mme_vla_suite` (subdir; see `data/download_robomme.sh` for the exact pattern) | 23 GB |
| openpi assets (tokenizer) | auto-downloads on first serve | 4 MB |

Training data (NOT needed for eval): `Yinpei/robomme_data_h5` (530 GB raw),
`Yinpei/robomme_preprocessed_data` (358 GB zips → 641 GB); scripts in `data/`.

## Environments

Two conda envs — sim and server stacks conflict and must stay separate:

- `robomme` (python 3.11): simulator + eval client. `env/robomme.lock.txt`.
  Key pins: `mani_skill 3.0.0b21`, `sapien 3.0.3`.
- `robomme-vla` (python 3.11): jax policy server. `env/robomme-vla.lock.txt`.
  Install the two editable packages from the policy repo
  (`pip install -e . && pip install -e packages/openpi-client`), then the lock.

Known trap from the source box: system ldconfig can shadow pip's CUDA libs
(`libnvJitLink`) → jax silently falls back to CPU with only a warning. The
runner already prepends every `site-packages/nvidia/*/lib` to LD_LIBRARY_PATH.

## Running the baseline

```
OUT=.../eval_out/seedN SEED=N NUM_SHARDS=<see below> MEM_FRACTION=0.95 \
  bash orchestration/run_eval_episode_sharded.sh
python orchestration/merge_robomme.py $OUT        # per-task vs paper Table 3
python orchestration/merge_seeds.py .../eval_out  # cross-seed aggregate
```

Sharding is episode-level: every shard runs all 16 tasks, owns episodes with
`ep % NUM_SHARDS == shard`, one policy server per shard (memory is
per-session; servers land on GPUs round-robin).

Reference numbers to reproduce before trusting anything else: seed 7 AVG
**46.5**, 3-seed **45.5 ± 0.9** (paper: 44.51). Cross-hardware bf16 drift is
expected to stay within seed noise; if seed 7 lands outside ~±2, stop and
investigate.

## 4090-specific must-dos

1. **Chunk the add_buffer encode** before first run. VideoPlaceOrder ships a
   ~1100-frame demo in one `add_buffer`; the single SigLIP allocation peaks at
   5.7 GB on top of ~12 GB resident — fine on an 80 GB A100, an OOM on 24 GB.
   Bound encodes to ≤128 frames per slice (server-side loop in
   `MemoryBuffer.add_buffer`, or client-side chunking in
   `examples/robomme/eval.py:get_action_chunk`).
2. **Drop lavapipe.** `ROBOMME_USE_LAVAPIPE` + `orchestration/lavapipe_patch.py`
   exist only because the source box's NVIDIA graphics path caused kernel
   panics (412 GPFs, 3 crashes). On a healthy consumer card use native Vulkan —
   but soak-test it first: a few GPU-rendered envs for an hour, watch dmesg.
3. Per-GPU server count: measure, don't assume. On A100-80G it was 4/GPU
   (~20 GB peak each). On 24 GB expect 1/GPU until (1) is in, then still 1 —
   the win on this box comes from more GPUs (8) + faster rendering, and later
   from batched serving (docs/PARALLEL_EVAL_PLAYBOOK.md describes the target
   architecture; our discussion settled on the stateless-server variant:
   client keeps per-frame embeddings, `encode`/`infer` endpoints, requests
   self-contained → any request to any server).

## Interventions (optional)

`interventions/mem_intervention.py` monkey-patches `MemoryBuffer` before the
stock server starts (`serve_policy_mem.py` wrapper; engage with
`ROBOMME_MEM_MODE=off|repeatcur|timeshuf|budget8|budget128|frames128`).
Run via `MODES=... bash interventions/run_interventions.sh`; results table via
`interventions/merge_interventions.py`. Completion criteria include a check
that all servers logged the patch — a silently-unpatched run reproduces the
baseline and misleads.

`results/` carries the source-box numbers these should reproduce:
`FINDINGS_interventions.md` is the narrative; `interventions.json` /
`baseline_3seed.json` the raw rates.

## Analysis scripts

`analysis/` is offline tooling (probes, coverage, contact sheets). The probes
read the *training* feature cache (`robomme_preprocessed_data`), so they need
that download — they are not part of eval.
