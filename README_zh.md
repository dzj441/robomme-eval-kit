# RoboMME 8-GPU 并行评测工具

本分支提供 MME-VLA 无状态推理的多 GPU 编排、episode 级负载均衡、结果合并和性能记录。配套 policy 实现在 [dzj441/robomme_policy_learning 的 `codex/stateless-batched-eval` 分支](https://github.com/dzj441/robomme_policy_learning/tree/codex/stateless-batched-eval)。

最终推荐配置在完整 800 episodes 上，warm JAX cache 用时 **680 秒（11 分 20 秒）**，成功率 **362/800 = 45.25%**，评测错误为 0。与实测 legacy preset 的三次平均 1413 秒相比，这两个端到端配置的 wall time 减少约 51.9%，吞吐约为 2.08×。

## 最终推荐配置

`orchestration/run_8gpu_fastest.sh` 固化了当前机器上最快且 SR 更稳定的配置：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `NUM_GPUS` | 8 | 每张 GPU 启动一个 policy server |
| `CLIENTS_PER_GPU` | 4 | 每张 GPU 对应 4 个 RoboMME env worker，共 32 个 |
| `MAX_BATCH_SIZE` | 1 | infer 不等待合批 |
| `MAX_WAIT_MS` | 0 | 消除 queue deadline 延迟 |
| `ENCODE_MAX_BATCH_SIZE` | 1 | history encode 单请求执行 |
| `ENCODE_MAX_WAIT_MS` | 0 | encode 不等待合批 |
| `SHARDING` | `balanced_steal_v1` | profile 初始均衡，完成自己队列后可偷取尾部任务 |
| `VIDEO_MODE` | `off` | 关闭视频编码和写盘 |
| `EVAL_NUM_THREADS` | 1 | 限制每个 env 的 BLAS/OpenMP 线程，避免 32 workers 过度争抢 CPU |
| `CPU_AFFINITY` | `per_gpu_v1` | env 与 server 固定到对应 GPU 所在 NUMA 节点 |
| `LAZY_HISTORY_ENCODE` | `false` | 使用 eager 历史编码，当前推荐 |
| `HISTORY_TRANSPORT_DTYPE` | `float32` | client/server 之间的 embedding 传输 dtype |

无状态 server 支持真正的动态 batching，但在这个 workload 上，环境步进、episode 长度和 history encode 使 infer 请求错峰。实测等待合批没有抵消 queue delay，`B=1 / wait=0` 最快。这里的收益主要来自：一个模型实例安全共享给多个 env、增加环境并行度、关闭视频 I/O、CPU/NUMA 隔离，以及消除长尾。

## 调度方案

### 一张卡一个 server，四个 env

legacy 方案为每个 env 启动独立 stateful server，8 张卡上共 16 servers / 16 envs。新 policy 把 episode 历史放到 client 后，同一张卡上的四个 env 可以安全共享一个 server，所以最终为 8 servers / 32 envs。

每个 worker 通过 `shard % NUM_GPUS` 连接对应 GPU 的 WebSocket 端口。server 内部将 encode 和 infer 放在同一条 GPU execution lane，避免多个线程同时驱动同一 JAX 模型。

### `balanced_steal_v1`

800 个 `(task, episode)` job 的耗时差异很大，单纯 round-robin 会在尾部等待少数长 episode。当前调度分两步：

1. 根据 `orchestration/profiles/seed7_lazy_numa_warm_episode_seconds.json` 的历史耗时，用 LPT（longest processing time first）计算每个 worker 的初始 owner，使估计总负载接近。
2. worker 先按 task-major 顺序处理自己的 job，保留 simulator/task 初始化局部性；自己的队列清空后，从其他 worker 的 pending 目录原子 claim job，消除尾部 straggler。

这个 profile 文件名保留了它最初采样时的 `lazy` 标记，但也用于最终 eager 配置，并得到 680 秒的实测结果。换 seed、任务集、机器或 checkpoint 后，建议重新生成 profile。

### JAX persistent cache

launcher 设置：

```text
JAX_COMPILATION_CACHE_DIR
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
```

首次出现的新 shape 会编译，所以 cold/new-compile run 为 972 秒；相同代码和 shape 的 warm-cache repeat 为 680 秒。比较吞吐时必须标明 cache 状态。

## 快速运行

建议将两个分支并排 clone：

```bash
git clone --branch codex/stateless-batched-eval \
  https://github.com/dzj441/robomme_policy_learning.git
git clone --branch codex/stateless-batched-eval \
  https://github.com/dzj441/robomme-eval-kit.git
```

policy server 和 RoboMME simulator 使用两个独立 Python 环境。完整命令：

```bash
cd /path/to/robomme-eval-kit

ROOT=/path/to/RoboMME \
POLICY_REPO=/path/to/robomme_policy_learning \
KIT_ROOT="$PWD" \
CKPT=/path/to/perceptual-framesamp-modul/79999 \
SERVER_PY=/path/to/robomme-vla-env/bin/python \
EVAL_PY=/path/to/robomme-env/bin/python \
SERVER_SITE_PACKAGES=/path/to/robomme-vla-env/lib/python3.11/site-packages \
OPENPI_DATA_HOME=/path/to/openpi-assets \
NVIDIA_DRIVER_VERSION=550.163.01 \
NVIDIA_DRIVER_ROOT=/path/to/headless-nvidia-runtime/550.163.01 \
SEED=7 \
OUT=/path/to/eval_out/8gpu_fastest_seed7 \
bash orchestration/run_8gpu_fastest.sh
```

`SERVER_SITE_PACKAGES` 可省略，launcher 默认通过 `SERVER_PY` 的 `site.getsitepackages()` 推导。以下变量必须与本机安装对应：

- `ROOT`：RoboMME 数据、checkpoint 默认路径和 JAX cache 的根目录；
- `POLICY_REPO`：配套 policy 分支；
- `KIT_ROOT`：本仓库路径；
- `CKPT`：checkpoint 目录；
- `SERVER_PY`：安装 policy/JAX 的 Python；
- `EVAL_PY`：安装 RoboMME simulator 的 Python；
- `OPENPI_DATA_HOME`：OpenPI assets/norm stats；
- `NVIDIA_DRIVER_ROOT`：当前 launcher 使用的无头渲染 runtime，内部应包含 `runtime-libs`、`nvidia_icd.local.json` 和 `10_nvidia.local.json`。

仓库中的绝对默认路径是源机器配置，只用于复现实验；其他机器请显式覆盖。`per_gpu_v1` 的 CPU 列表按源机器的 2 NUMA nodes / 128 logical CPUs 编写，拓扑不同请修改 `gpu_cpu_list` 和 `gpu_numa_node`，或先设置 `CPU_AFFINITY=none`。

## 输出与恢复

每次运行的 `OUT` 包含：

```text
run_config.txt       实际启动参数与 worker 初始化信息
TIMING.txt           wall time、吞吐、完成数和最终状态
FINAL_REPORT.txt     16 tasks 的 SR 汇总
aggregate.json       机器可读的结果与 errors
ownership.json       初始 episode owner 和估计负载
work_queue/          pending / running / done 原子队列
logs/                每个 server 和 evaluator 的日志
shard*/              worker 进度与 episode 结果
```

不要在非空的同一个 `OUT` 上直接启动一轮新实验；动态队列检测到已有 job 时会拒绝启动。中断后如需保留结果，应先复制目录，再根据具体恢复需求处理 queue 状态。

## 完整 800-episode 实测

统一条件：seed 7、checkpoint 79999、`perceptual-framesamp-modul`、16 tasks × 50 episodes。SR 使用精确 success 数计算；报告文件显示到一位小数。

| 方案 | 配置摘要 | wall time | success | SR | errors |
|---|---|---:|---:|---:|---:|
| legacy run 0 | 16 servers / 16 envs，global RR，保存视频 | 1405 s | 371/800 | 46.375% | 0 |
| legacy run 1 | 同上 | 1423 s | 370/800 | 46.250% | 0 |
| legacy run 2 | 同上 | 1411 s | 370/800 | 46.250% | 0 |
| stateless eager cold | 8 servers / 32 envs，balanced-steal，B1，视频关闭 | 972 s | 362/800 | 45.250% | 0 |
| stateless eager warm | 同上，persistent cache 已预热 | **680 s** | 362/800 | 45.250% | 0 |
| stateless lazy warm | eager 改为 lazy | 681 s | 357/800 | 44.625% | 0 |

注意：legacy preset 保存视频，而最快 preset 关闭视频，因此 2.08× 是两个实际端到端运行配置的比较，不应解读为仅由无状态 policy 带来的纯模型加速。

两次 stateless eager 的 800 个 episode outcome 完全一致（0 flips）。三次 legacy 的聚合 SR 均在 46.25–46.375%，但两两存在 55、65、64 个 episode success/failure 翻转，三轮共有 92 个 episode 至少翻转一次。这说明 aggregate 很稳定，但当前 simulator/process 调度路径并不保证 episode 级严格确定性。

eager 的 45.25% 比三次 legacy 均值 46.292% 低约 1.04 个百分点，仍处于本轮探索接受的合理范围。当前结果不证明某个单次 legacy 数值“偏高”或 stateless 数值“偏低”；要做严格科学比较，需要控制视频、进程布局和编译状态，并增加多个 seed/repeat。

lazy 与 eager 之间有 23 个 episode 翻转：eager-only success 14 个，lazy-only success 9 个，净差 5 个 success（0.625 个百分点）。两者 warm wall time 几乎相同，因此最终默认改为 eager。

## 为什么不是更大的 batch

已经探索过 16/24/32/40/64 workers、不同 infer batch/wait、encode 合批、静态均衡、全局动态队列、balanced-steal、lazy/eager、float16/float32 transport、NUMA 和视频开关。最终发现：

- 32 env workers 足以隐藏大部分 simulator 等待；继续增加会加剧 CPU 和渲染争抢；
- infer 请求到达不够同步，`B>1` 的等待经常成为关键路径；
- encode 也与环境进度错峰，强行等待合批没有产生净收益；
- balanced initial ownership 保留 task locality，尾部 stealing 又能处理偶发长 episode；
- warm cache 下 eager 与 lazy 同速，而 eager SR 更高、更可重复。

所以“支持真正 batching”是架构能力，当前最快配置使用 B1 是经过端到端搜索后的结果，并不矛盾。

## 脚本索引

```text
orchestration/run_8gpu_fastest.sh
    最终推荐 preset

orchestration/run_8gpu_stateless_batched.sh
    可配置的 8-GPU launcher

orchestration/seed_episode_shards.py
    round-robin、LPT、动态队列和 balanced-steal 初始化

orchestration/count_owned_progress.py
    动态 owner/queue 场景的完成度统计

orchestration/merge_robomme.py
    合并各 worker 输出并生成 aggregate/report

orchestration/build_episode_timing_profile.py
orchestration/build_episode_weight_profile.py
    从已有 run 生成 episode cost profile
```

## 已知限制

- launcher 目前是一张 GPU 一个 checkpoint/server，没有跨 GPU router。
- `per_gpu_v1` 是源机器定制拓扑，不可直接假设适用于其他主机。
- 最快 preset 关闭视频；需要视频时设置 `VIDEO_MODE=save` 并重新测量容量和耗时。
- timing profile 与 workload 相关；改 seed、episode 数、checkpoint 或任务列表后应重新校准。
- 本仓库不包含 checkpoint、OpenPI assets、RoboMME 数据或 NVIDIA runtime。
