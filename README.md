# RoboMME Parallel Evaluation Kit

This branch contains the orchestration used to evaluate MME-VLA with one stateless policy server per GPU and multiple RoboMME environment workers per server.

This is the driver-specific `codex/stateless-batched-eval-570` branch. On a four-GPU node with kernel driver `570.124.06`, use `orchestration/run_4gpu_fastest_570.sh`; it fixes the matching userspace rendering stack, four policy servers, and sixteen environment workers. The separate `codex/stateless-batched-eval-550` branch provides `run_4gpu_fastest_550.sh`. The launchers intentionally do not auto-select a driver ABI.

Two validated presets are provided. The deterministic parity preset was run twice from independent empty compilation caches: it completed all 800 episodes in 912 and 894 seconds, produced 370/800 successes (46.25%) both times, and had zero episode-level flips. The higher-throughput preset completes in 716 seconds and produced 366/800 successes (45.75%), but its bucketed history encoding is not numerically identical to legacy.

Trace-based diagnosis found that the older uncached 16-server legacy runs (370-371/800) compiled numerically different XLA executables during first-use GPU autotuning. This also occurs in sequential fresh-cache compilations, so a shared cache only pins one arbitrary executable. The deterministic preset disables XLA GPU autotuning and performs one isolated prewarm; four single-episode cold-cache replicas and two complete 800-episode cold-cache runs then matched exactly. See the Chinese README for the evidence and configurations.

- [中文：并行化设计、运行方法、最优配置与完整测试结果](README_zh.md)
- [Companion policy implementation](https://github.com/dzj441/robomme_policy_learning/tree/codex/stateless-batched-eval)

Deterministic entry point:

```bash
ROOT=/path/to/RoboMME \
POLICY_REPO=/path/to/robomme_policy_learning \
KIT_ROOT="$PWD" \
CKPT=/path/to/checkpoint \
SERVER_PY=/path/to/vla-env/bin/python \
EVAL_PY=/path/to/robomme-env/bin/python \
OPENPI_DATA_HOME=/path/to/openpi-assets \
OUT=/path/to/eval_out/deterministic_parity \
bash orchestration/run_8gpu_deterministic_parity.sh
```

Use `orchestration/run_8gpu_fastest.sh` when maximum throughput is more important than strict legacy numerical parity.

The launcher also expects a working headless RoboMME/NVIDIA runtime. Its checked-in NVIDIA paths describe the source machine and can be overridden with `NVIDIA_DRIVER_ROOT`. See the Chinese README for configuration details and caveats.
