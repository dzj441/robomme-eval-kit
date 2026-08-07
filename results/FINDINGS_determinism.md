# Seed-7 legacy determinism findings

Scope: checkpoint `perceptual-framesamp-modul/79999`, 16 RoboMME tasks × 50 episodes, model seed 7, NVIDIA driver 550.163.01.

## Episode-level results

| Run | Compile regime | Success | Wall time |
|---|---|---:|---:|
| legacy 0 | uncached concurrent first compile | 371/800 | 1405 s |
| legacy 1 | uncached concurrent first compile | 370/800 | 1423 s |
| legacy 2 | uncached concurrent first compile | 370/800 | 1411 s |
| deterministic exact 0 | independent empty cache, autotune level 0 | 370/800 | 912 s |
| deterministic exact 1 | second independent empty cache, autotune level 0 | 370/800 | 894 s |
| pinned-cache legacy | shared autotuned persistent cache | 363/800 | 987 s |
| pinned-cache stateless, 16 env | same shared cache | 363/800 | 974 s |
| pinned-cache stateless, 32 env | same shared cache | 363/800 | 915 s |

The three uncached legacy runs have 55, 65, and 64 pairwise episode flips; 92 distinct episodes change outcome at least once. The two autotune-disabled runs have zero flips over all 800 episodes. The pinned-cache legacy and both stateless-exact layouts also have zero flips, demonstrating wrapper parity when the executable is held fixed.

## First-divergence trace

For `BinFill episode 5`, 16 independent uncached legacy servers produced 12 successes and 4 failures. Two opt-in trace layers retained:

- episode seed, pretrajectory images and states, every history update, infer request/response, and every environment action/observation;
- vision `image_emb_4x4`, transformed observation, static memory tensors, prompt tokens and masks, sample RNG key, raw model action, and output-transformed action.

The reset/pretrajectory data, raw images, state, vision embeddings, complete model input, and RNG key were bitwise identical. The first raw `_sample_actions` result differed in 425 of 640 values with maximum absolute difference `0.0048828125`; after output transforms, the maximum 8-D action difference was about `0.00293`.

Using a separate empty cache per server reproduced the differences. Filling one shared JAX persistent cache from a single isolated process first made four independent replicas, including two servers per GPU, identical for all 1103 rollout events. However, two single-server processes compiled sequentially on the same GPU into separate empty caches still differed at the first model output: 394/640 raw action values changed, with maximum absolute difference `0.001953125`. Fresh GPU autotuning itself is therefore sufficient; concurrency merely lets a legacy run mix more executable variants.

With `--xla_gpu_autotune_level=0`, four independent empty-cache single-episode runs were identical for all 1311 events. Two used only the autotune flag and two also enabled deterministic ops; all four matched, so deterministic ops is unnecessary. The two full 800-episode runs above also matched on 800/800 outcomes. Simulator reset, memory handling, RNG sequencing, and steady-state GPU sharing are not the source.

## Control used by the launchers

`orchestration/run_8gpu_stateless_batched.sh` runs one `serve_policy.py --prewarm-only` process before starting parallel servers. The cache marker includes the driver, checkpoint, model seed, batch configuration, JAX version, serving-code hashes, and server XLA flags. `orchestration/run_8gpu_deterministic_parity.sh` additionally enables B=1 legacy RNG graph parity, unbucketed exact history encode, and `--xla_gpu_autotune_level=0`.

The source-machine diagnostic artifacts are under:

```text
RoboMME/eval_out/diagnostics/legacy_trace_BinFill_ep5_r16_v1
RoboMME/eval_out/diagnostics/legacy_trace_BinFill_ep5_r4_2pergpu_freshcompile_v6
RoboMME/eval_out/diagnostics/legacy_trace_BinFill_ep5_r4_2pergpu_policytrace_v5
RoboMME/eval_out/diagnostics/legacy_trace_BinFill_ep5_single_fresh_sequential_{a,b}
RoboMME/eval_out/diagnostics/legacy_trace_BinFill_ep5_single_fresh_autotune0_{a,b}
RoboMME/eval_out/8gpu_32workers_seed7_deterministic_autotune0_exact_{full,repeat2}
```

Episode maps from any copied runs can be compared with:

```bash
python analysis/compare_episode_outcomes.py RUN_A RUN_B [RUN_C ...]
```

JSONL rollout traces can be compared with the companion policy repository's `examples/robomme/compare_episode_traces.py`.
