# 新集群接入清单（Temporal/State）

## 代码分支

```bash
git clone --branch feature/shared-h100-command-bridge \
  https://github.com/dzj441/robomme_benchmark.git RoboMME
git clone --branch experiment/temporal-state-memory-eval \
  https://github.com/dzj441/robomme_policy_learning.git RoboMME_policy_eval
git clone --branch experiment/temporal-state-memory-eval \
  https://github.com/dzj441/robomme-eval-kit.git robomme-eval-kit
```

若新集群也承担训练，另 clone policy 的
`experiment/temporal-state-memory` 分支。不要用评测分支启动训练，也不要用训练
分支启动 stateless server；二者共享模型实现，但分别继承训练吞吐和并行评测基础设施。

## 环境与外部资产

- simulator/client 使用 `robomme` 环境；JAX server 使用 `robomme-vla` 环境；
- 版本参考 `env/robomme.lock.txt` 与 `env/robomme-vla.lock.txt`；
- checkpoint、openpi assets、训练数据与 frame cache 保持在数据盘，通过路径或软链接接入；
- `ghp`、ModelScope token、W&B key、数据和 checkpoint 不进入 Git；
- 启动前分别检查 `jax.devices()`、`nvidia-smi`、Vulkan headless rendering；
- 若宿主 CUDA 覆盖 pip CUDA 库，优先把当前 Python 环境下
  `site-packages/nvidia/*/lib` 放入 `LD_LIBRARY_PATH`。

## ModelScope 数据、cache 与 checkpoint

训练分支包含连续 frame-history cache、共享内存 batch IPC、80k 训练入口和
Temporal/State 新参数树，已经明显超出原仓库的 launcher 修改。旧的 343 GiB
迁移包只有原始 `data/features`、π0.5 base 和 norm stats；新机器还必须恢复
`accelerated_training/`。

公开 ModelScope 仓库：
`dzjjzd/robomme-minimal-train-bundle`。新增部分约 47 GiB，包括：

- 48 GiB `frame_sampling_history_v1`（50 GB 主 memmap 以 12 个 transport
  parts 上传，安装时无损重建）；
- 4 MiB `big_vision/paligemma_tokenizer.model`，保证离线启动；
- cache 目录内独立的 `SHA256SUMS`。

约 638 GiB 的解压数据无需上传，新机器从旧包本地解压。独立 SigLIP encoder、
两个已训练的 79999 checkpoint、失败的约 932 GiB current-image SigLIP
cache、Anchor/Recent、训练日志和中间 checkpoint 都不属于从 π0.5 base
重新训练的输入，因此不在补充包里；评测已有模型时需另行指定 checkpoint。
新集群使用 policy 训练分支自带的一键脚本：

```bash
cd /path/to/RoboMME_policy_train

# 脚本内部会显式 unset http_proxy/https_proxy/all_proxy（含大写变量）。
INSTALL_ROOT=/data/robomme_training \
POLICY_REPO="$PWD" \
bash scripts/setup_new_cluster_from_modelscope.sh
```

`modelscope==1.39.0` 已记录在 eval-kit 的 `env/robomme.lock.txt`。若 simulator
环境没有加入 `PATH`，给脚本传
`MODELSCOPE_BIN=/path/to/robomme/bin/modelscope` 即可，不需要把 ModelScope
安装进 JAX 训练环境。

已有旧 bundle/解压数据时，用
`MODE=accelerated-only EXTRACT_DATA=0` 只补新增目录。详细变量、校验、重建和
软链接规则见 policy 仓库
`docs/temporal_state_memory.zh-CN.md`；不要手工把 50 GB transport parts 目录
直接传给训练器。

## 最小评测启动

```bash
ROOT=/path/to/RoboMME \
POLICY_REPO=/path/to/RoboMME_policy_eval \
TRAIN_ROOT=/data/training_runs/ckpts/mme_vla_suite \
OUT_ROOT=/data/eval_out/temporal_state_seed7 \
NVIDIA_DRIVER_ROOT=/path/to/headless-nvidia-runtime/$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) \
PRESET=fastest NUM_GPUS=8 CLIENTS_PER_GPU=4 SEED=7 STEP=79999 \
bash orchestration/run_memory_explorations_seed7.sh
```

runner 默认只评测
`perceptual-temporal-state-modul_8h100_b64_fastio_seed42`。GPU 数和每卡 env 数
可以缩放；每张 GPU 仍只启动一个 stateless server。

## 验收

1. server metadata 中 fingerprint 与 client 期望一致；
2. `NUM_GPUS` 个 server 和 `NUM_GPUS × CLIENTS_PER_GPU` 个 env 全部启动；
3. `work_queue/done` 持续增长，GPU 利用率稳定；
4. 800/800 完成、`errors=0`、`TIMING.txt` 为 `status=complete`；
5. 进程退出后端口、server 和 env 均被清理。

共享文件系统的跨节点命令桥见 RoboMME 仓库
`tools/shared_command_bridge/README.zh-CN.md`。
