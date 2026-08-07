# RoboMME 8-GPU 并行评测工具

本分支提供 MME-VLA 无状态推理的多 GPU 编排、episode 级负载均衡、结果合并和性能记录。配套 policy 实现在 [dzj441/robomme_policy_learning 的 `codex/stateless-batched-eval` 分支](https://github.com/dzj441/robomme_policy_learning/tree/codex/stateless-batched-eval)。

当前提供两个经过完整 800 episodes 验证的入口：

- **严格可复现 / legacy 数值路径**：`run_8gpu_deterministic_parity.sh`，用两个独立空 cache 完整运行分别为 **912 秒**和 **894 秒**，均为 **370/800 = 46.25%**，800 个 episode 逐项一致、0 flips。
- **最大吞吐**：`run_8gpu_fastest.sh`，用时 **716 秒**，成功率 **366/800 = 45.75%**；其 history encode 使用固定 bucket，不承诺与 legacy 逐位一致。

两者评测错误均为 0。旧的 uncached legacy 三轮也是 370–371/800，但 episode map 两两有 55–65 个 flip。trace 证明首个分叉来自 fresh XLA GPU autotune 生成的不同 executable；即使在同一 GPU 上串行编译两个独立空 cache 也会发生，不只是多进程竞争。

## 四种 8-GPU 评测方案

以下结果统一使用 seed 7、checkpoint 79999、16 tasks × 50 episodes，共 800 episodes。

| 方案 | Server / Env | 关键数值路径 | 时间 | Success / SR | 用途 |
|---|---:|---|---:|---:|---|
| 原始 legacy | 16 / 16 | stateful、默认 XLA autotune、各 server 独立首次编译 | 平均 1413 s | 370–371 / 46.25–46.375% | 原始基线 |
| 历史最快 680 秒 | 8 / 32 | stateless、B=1 显式 noise、bucket encode、warm cache | 680 s | 362 / 45.25% | 历史最快，parity 修复前 |
| 当前最快约 710 秒 | 8 / 32 | stateless、legacy B=1 RNG graph、bucket encode、autotune + 隔离预热 | 716 s | 366 / 45.75% | 当前推荐吞吐配置 |
| deterministic exact | 8 / 32 | legacy B=1 RNG graph、exact encode、autotune=0、隔离预热 | 894 / 912 s | 两次均为 370 / 46.25% | 正式、跨空 cache 可复现 |

### 1. 原始 legacy：16 servers / 16 envs

原始方案在每张 GPU 上启动两个 stateful policy server，共 16 个 server；每个 server 只服务一个 env worker。episode history、step index 和 RNG 都保存在对应 server 内。

```text
GPU 数量                  8
每卡 server               2
总 server                 16
每卡 env                  2
总 env                    16
sharding                  global_round_robin_v1
history                   保存在 server
model batch               B=1
video                     保存
XLA autotune              开启
统一编译预热              无
```

三个完整运行分别为：

```text
1405 s    371/800    46.375%
1423 s    370/800    46.250%
1411 s    370/800    46.250%
```

聚合 SR 看起来稳定，但三轮 episode outcome 两两有 55、65、64 个 flip，共 92 个 episode 至少翻转过一次。

### 2. 历史最快：680 秒

这是 parity 修复前的 stateless eager 版本：

```text
GPU 数量                  8
每卡 server               1
总 server                 8
每卡 env                  4
总 env                    32
MAX_BATCH_SIZE            1
MAX_WAIT_MS               0
history encode            eager
LAZY_HISTORY_ENCODE       false
history encode shape      bucket padding
encode batch / wait       1 / 0
sharding                  balanced_steal_v1
video                     off
CPU affinity              per_gpu_v1
transport dtype           float32
JAX cache                 warm
```

该版本即使 B=1，也先显式生成 per-request noise，再调用 batched `_sample_actions`。数学上与 legacy 等价，但改变了 JAX graph 和 BF16 数值路径。

```text
680 s    362/800    45.25%    0 errors
```

这是所有已完成实验中 wall time 最小的一轮，但它属于历史实现，不是当前 HEAD 的正式运行入口。

### 3. 当前最快：716 秒

当前 `orchestration/run_8gpu_fastest.sh` 保留 680 秒方案的并行拓扑和调度方式，但修复了 B=1 policy path：

```text
GPU 数量                  8
每卡 server               1
总 server                 8
每卡 env                  4
总 env                    32
MAX_BATCH_SIZE            1
MAX_WAIT_MS               0
history encode            eager
LAZY_HISTORY_ENCODE       false
LEGACY_EXACT_ENCODE       false
history encode shape      bucket padding
encode batch / wait       1 / 0
sharding                  balanced_steal_v1
video                     off
CPU affinity              per_gpu_v1
transport dtype           float32
XLA autotune              开启
deterministic prewarm     开启
```

B=1 时直接把 legacy sample key 传给 `_sample_actions`，由模型内部生成 noise，与 legacy wrapper 使用相同的 RNG/JAX graph。launcher 先由单一进程预热 action graph，再启动八个 server。

```text
716 s    366/800    45.75%    0 errors
```

相对历史 680 秒版本，当前方案慢 36 秒（约 5.3%），但增加 4 个 success，SR 提升 0.50 个百分点。36 秒差异不能全部解释为模型开销，因为两个数值路径会产生不同轨迹、episode 长度和尾部调度负载。

运行入口：

```bash
bash orchestration/run_8gpu_fastest.sh
```

### 4. Deterministic exact：894–912 秒

正式可复现方案继续使用 8 servers / 32 envs，但同时控制 history encode 和 XLA 编译：

```text
GPU 数量                  8
每卡 server               1
总 server                 8
每卡 env                  4
总 env                    32
MAX_BATCH_SIZE            1
MAX_WAIT_MS               0
history encode            eager
LAZY_HISTORY_ENCODE       false
LEGACY_EXACT_ENCODE       true
history encode padding    无
encode batch / wait       1 / 0
ENCODE_MAX_TOTAL_FRAMES   4096
sharding                  balanced_steal_v1
video                     off
CPU affinity              per_gpu_v1
transport dtype           float32
SERVER_XLA_FLAGS          --xla_gpu_autotune_level=0
deterministic prewarm     开启
JAX cache                 独立 autotune0 cache
```

两个完全独立的空 cache 运行结果：

```text
Run 1    912 s    370/800    46.25%    0 errors
Run 2    894 s    370/800    46.25%    0 errors
```

两轮 800 个 episode outcome 完全一致，0 flips。相对 legacy 平均 1413 秒，deterministic exact 平均 903 秒，节省约 510 秒（8.5 分钟），wall time 降低约 36%，吞吐提升约 1.56×。

运行入口：

```bash
bash orchestration/run_8gpu_deterministic_parity.sh
```

### Lazy history encode

上述三个 stateless 方案默认均不开 lazy：

```text
LAZY_HISTORY_ENCODE=false
```

lazy 完整实验为 681 秒、357/800（44.625%）。它没有比 eager 更快，SR 反而更低，因此不列为推荐方案。

无状态 server 支持真正的动态 batching，但在这个 workload 上，环境步进、episode 长度和 history encode 使 infer 请求错峰。实测等待合批没有抵消 queue delay，`B=1 / wait=0` 最快。主要收益来自一个模型实例安全共享给多个 env、增加环境并行度、关闭视频 I/O、CPU/NUMA 隔离，以及消除长尾。

## 调度方案

### 一张卡一个 server，四个 env

legacy 方案为每个 env 启动独立 stateful server，8 张卡上共 16 servers / 16 envs。新 policy 把 episode 历史放到 client 后，同一张卡上的四个 env 可以安全共享一个 server，所以最终为 8 servers / 32 envs。

每个 worker 通过 `shard % NUM_GPUS` 连接对应 GPU 的 WebSocket 端口。server 内部将 encode 和 infer 放在同一条 GPU execution lane，避免多个线程同时驱动同一 JAX 模型。

### `balanced_steal_v1`

800 个 `(task, episode)` job 的耗时差异很大，单纯 round-robin 会在尾部等待少数长 episode。当前调度分两步：

1. 根据 `orchestration/profiles/seed7_lazy_numa_warm_episode_seconds.json` 的历史耗时，用 LPT（longest processing time first）计算每个 worker 的初始 owner，使估计总负载接近。
2. worker 先按 task-major 顺序处理自己的 job，保留 simulator/task 初始化局部性；自己的队列清空后，从其他 worker 的 pending 目录原子 claim job，消除尾部 straggler。

这个 profile 文件名保留了它最初采样时的 `lazy` 标记，但也用于当前 eager 配置。716 秒的吞吐结果和 894–912 秒的 deterministic exact 结果都使用该 profile。换 seed、任务集、机器或 checkpoint 后，建议重新生成 profile。

### JAX persistent cache

launcher 设置：

```text
JAX_COMPILATION_CACHE_DIR
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
```

首次出现的新 shape 会编译。默认 XLA GPU autotune 会根据实测候选 kernel 选 executable；本机上即使两个进程在同一 GPU 上完全串行、各用一个独立空 cache，也会选出数值略有差异的结果。shared cache 可以固定某一次选择，但删 cache 或换机器后仍可能改变。

strict preset 因此设置 `--xla_gpu_autotune_level=0`，直接消除候选 kernel 的计时选择；launcher 仍先执行一次隔离的 `--prewarm-only`，再让八个 server 复用该 persistent cache。marker 摘要覆盖 driver、checkpoint、代码、JAX、batch 配置和 XLA flags。吞吐 preset 保留 autotune 以获得更低的单次 infer latency，并依靠 prewarm/cache 固定当前机器上的选择。

strict preset 默认使用独立的 `.jax_compilation_cache_autotune0`，避免和历史 autotuned executable 混放；显式覆盖 `JAX_CACHE_DIR` 时也应保持两种 compile regime 分离。

### legacy flip 的首分叉点

诊断并不是只比较最终 SR。对三轮 legacy 中会翻转的 `BinFill episode 5`，同时启动 16 个独立 legacy server 并记录了 seed、初始轨迹、图像哈希、state、memory、RNG、模型原始 action 和每一步环境状态：

- 相同 episode 得到 12 次成功、4 次失败；
- seed、reset 后的所有图像/state、buffer、vision `image_emb_4x4`、tokenized prompt、`(1,512,2048)` memory 和 sample RNG 全部逐位一致；
- 第一次 `_sample_actions` 的 raw `(1,20,32)` 输出已经不同，最大差 `0.0048828125`；最终 8-D action 最大差约 `0.00293`；
- 每个 server 使用独立空 cache 并发首次编译时差异复现；命中同一个 shared cache 后，4 个副本的完整 1103 个 trace event 全部一致；
- 两个单 server 在同一 GPU 上串行使用独立空 cache，首个 infer 的输入仍逐位一致，但 raw action 有 394/640 个值不同，最大差 `0.001953125`，说明并发不是必要条件；
- 设置 `--xla_gpu_autotune_level=0` 后，4 个独立空 cache（其中两次额外设置 deterministic ops）的完整 1311-event rollout 全部一致，证明只关闭 autotune 已足够；
- 两个独立空 cache 的 800-episode stateless-exact 完整运行分别耗时 912 和 894 秒，均为 370/800，并且 outcome **800/800 完全一致**；
- 作为 wrapper parity 对照，固定另一份 autotuned cache 的原始 legacy、32-env stateless-exact、16-env stateless-exact 均为 363/800，三者同样在 800/800 上一致。

所以 44.625%、45.375% 和 46.25% 并不是 simulator seed 失效，而是 BF16 临界 episode 放大了不同编译/encode 数值路径。新的 strict 配置从空 cache 开始可重复得到 46.25%；它与旧 legacy 的总 SR 对齐，但相对旧三轮仍各有 46–62 个具体 episode 翻转，因此不能声称旧轨迹被逐 episode 复刻。

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
OUT=/path/to/eval_out/8gpu_deterministic_seed7 \
bash orchestration/run_8gpu_deterministic_parity.sh
```

若只追求吞吐，将最后一行换为 `bash orchestration/run_8gpu_fastest.sh`。

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
| deterministic exact run 1 | 独立空 cache，autotune=0，8 servers / 32 envs | 912 s | 370/800 | 46.250% | 0 |
| deterministic exact run 2 | 第二个独立空 cache，同配置 | 894 s | 370/800 | 46.250% | 0 |
| pinned-cache legacy | 原始 16 servers / 16 envs，固定另一份 autotuned cache，保存视频 | 987 s | 363/800 | 45.375% | 0 |
| pinned-cache stateless | 8 servers / 32 envs，复用上行 cache，exact encode，B1 | 915 s | 363/800 | 45.375% | 0 |
| pinned-cache stateless 16 env | 8 servers / 16 envs，global RR，exact encode | 974 s | 363/800 | 45.375% | 0 |
| stateless throughput | 8 servers / 32 envs，bucket encode，B1 | **716 s** | 366/800 | 45.750% | 0 |
| 较早的 stateless eager warm | parity 修复前 | 680 s | 362/800 | 45.250% | 0 |
| stateless lazy warm | lazy history encode | 681 s | 357/800 | 44.625% | 0 |

注意：legacy preset 保存视频，而两个新 preset 关闭视频。相对三轮 legacy 平均 1413 秒，deterministic exact 平均 903 秒（约 1.56× 吞吐），最快 preset 为 716 秒（约 1.97×）；这些是实际端到端配置比较，不应解读为仅由无状态 policy 带来的纯模型加速。

三次旧 legacy 两两存在 55、65、64 个 episode 翻转，共 92 个 episode 至少翻转一次。固定同一 cache 时 legacy 与两种 stateless worker 布局之间均为 0 flips；关闭 autotune 后两个独立空 cache 之间也为 0 flips。这排除了视频、env 数量、sharding 和 simulator 作为主因，并表明正式比较必须记录 XLA flags 和 cache regime。

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
    最大吞吐 preset

orchestration/run_8gpu_deterministic_parity.sh
    从空 cache 可复现 370/800 的 strict preset

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

orchestration/run_legacy_trace_replicas.sh
examples/robomme/compare_episode_traces.py（policy 仓库）
    legacy flip 的多副本 trace 与首分叉诊断工具

analysis/compare_episode_outcomes.py
results/FINDINGS_determinism.md
    episode-level flip 比较工具与本次确定性诊断记录
```

## 已知限制

- launcher 目前是一张 GPU 一个 checkpoint/server，没有跨 GPU router。
- `per_gpu_v1` 是源机器定制拓扑，不可直接假设适用于其他主机。
- 最快 preset 关闭视频；需要视频时设置 `VIDEO_MODE=save` 并重新测量容量和耗时。
- timing profile 与 workload 相关；改 seed、episode 数、checkpoint 或任务列表后应重新校准。
- marker 覆盖 driver/checkpoint/代码/JAX/batch/XLA flags；升级其中任一项会自动重新预热，并应重新跑 parity 验证。
- `autotune_level=0` 的单次 warm infer 在本机约 81 ms，而 autotuned executable 约 59–60 ms；strict preset 以跨空-cache一致性换取模型 latency，端到端时间仍受 env 并行和首次编译共同影响。
- 本仓库不包含 checkpoint、OpenPI assets、RoboMME 数据或 NVIDIA runtime。
