# RoboMME Parallel Evaluation Kit

This branch contains the orchestration used to evaluate MME-VLA with one stateless policy server per GPU and multiple RoboMME environment workers per server.

The fastest validated 8-GPU preset completes the full 800-episode benchmark in 680 seconds with a warm JAX compilation cache, versus an average of 1413 seconds for the measured 16-server legacy preset. The optimized run produced 362/800 successes (45.25%) with zero evaluation errors. This is an end-to-end preset comparison: the optimized preset disables video output while the measured legacy preset saves videos.

- [中文：并行化设计、运行方法、最优配置与完整测试结果](README_zh.md)
- [Companion policy implementation](https://github.com/dzj441/robomme_policy_learning/tree/codex/stateless-batched-eval)

Quick entry point:

```bash
ROOT=/path/to/RoboMME \
POLICY_REPO=/path/to/robomme_policy_learning \
KIT_ROOT="$PWD" \
CKPT=/path/to/checkpoint \
SERVER_PY=/path/to/vla-env/bin/python \
EVAL_PY=/path/to/robomme-env/bin/python \
OPENPI_DATA_HOME=/path/to/openpi-assets \
OUT=/path/to/eval_out/fastest \
bash orchestration/run_8gpu_fastest.sh
```

The launcher also expects a working headless RoboMME/NVIDIA runtime. Its checked-in NVIDIA paths describe the source machine and can be overridden with `NVIDIA_DRIVER_ROOT`. See the Chinese README for configuration details and caveats.
