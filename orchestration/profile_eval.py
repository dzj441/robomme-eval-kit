"""Profile where RoboMME evaluation actually spends wall-clock.

The parallel-eval playbook's biggest win (server-side dynamic batching) only pays
off when the GPU forward dominates. Here the suspicion is the opposite: CPU
physics plus motion planning in reset() dominate, which would cap any batching
speedup by Amdahl. This measures the split before we commit to a refactor.

Patches the real client/env objects with timers, then drives the real
EpisodeEvaluator path for a few episodes per task.
"""

from __future__ import annotations

import collections
import sys
import time

sys.path.insert(0, "/home/qid/dzj/RoboMME_policy/examples/robomme")

import tyro  # noqa: E402
from openpi_client import websocket_client_policy as wcp  # noqa: E402

import eval as eval_mod  # noqa: E402
from env_runner import EnvRunner  # noqa: E402

T = collections.defaultdict(float)
C = collections.defaultdict(int)


def timed(bucket, fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            T[bucket] += time.perf_counter() - t0
            C[bucket] += 1

    return wrapper


CLS = wcp.MMEVLAWebsocketClientPolicy
CLS.infer = timed("policy.infer (GPU)", CLS.infer)
CLS.add_buffer = timed("policy.add_buffer (GPU)", CLS.add_buffer)
CLS.reset = timed("policy.reset", CLS.reset)
EnvRunner.get_init_obs = timed("env.reset (CPU sim + planner)", EnvRunner.get_init_obs)
EnvRunner.step = timed("env.step (CPU sim)", EnvRunner.step)
EnvRunner.make_env = timed("env.make (CPU build)", EnvRunner.make_env)


def main(tasks: str, episodes: int, port: int, save_dir: str):
    args = eval_mod.Args(
        host="127.0.0.1", port=port, policy_name="profile",
        model_seed=7, model_ckpt_id=79999, save_dir=save_dir,
    )
    from pathlib import Path
    sd = Path(save_dir)
    (sd / "videos").mkdir(parents=True, exist_ok=True)
    predictor = eval_mod.build_subgoal_predictor(args, sd)
    evaluator = eval_mod.EpisodeEvaluator(args, sd)

    grand0 = time.perf_counter()
    per_task = {}
    for task in tasks.split(","):
        runner = EnvRunner(task, str(sd / "videos"), max_steps=args.max_steps)
        t0 = time.perf_counter()
        outcomes = []
        for ep in range(episodes):
            runner.make_env(ep)
            outcomes.append(evaluator.eval_each_episode(runner, predictor, sd / "videos"))
        per_task[task] = (time.perf_counter() - t0, outcomes)
        runner.close_env()
        print(f"[{task}] {per_task[task][0]:.1f}s for {episodes} eps -> {outcomes}", flush=True)

    total = time.perf_counter() - grand0
    print("\n" + "=" * 78)
    print(f"{'bucket':<32} {'total_s':>9} {'calls':>8} {'ms/call':>10} {'% wall':>8}")
    print("-" * 78)
    for k in sorted(T, key=lambda x: -T[x]):
        print(f"{k:<32} {T[k]:9.1f} {C[k]:8d} {T[k]/max(C[k],1)*1000:10.1f} {100*T[k]/total:7.1f}%")
    gpu = T["policy.infer (GPU)"] + T["policy.add_buffer (GPU)"]
    cpu = T["env.reset (CPU sim + planner)"] + T["env.step (CPU sim)"] + T["env.make (CPU build)"]
    print("-" * 78)
    print(f"{'GPU (policy)':<32} {gpu:9.1f} {'':8} {'':10} {100*gpu/total:7.1f}%")
    print(f"{'CPU (sim)':<32} {cpu:9.1f} {'':8} {'':10} {100*cpu/total:7.1f}%")
    print(f"{'wall':<32} {total:9.1f}")
    print("=" * 78)
    if gpu > 0:
        print(f"\nAmdahl: even with infinitely fast batched GPU inference, the best")
        print(f"possible speedup on this workload is {total/(total-gpu):.2f}x")


if __name__ == "__main__":
    tyro.cli(main)
