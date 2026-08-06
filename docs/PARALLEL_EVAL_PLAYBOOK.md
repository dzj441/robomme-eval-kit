# Parallel Batched-Inference Evaluation — Architecture & Porting Playbook

> **Purpose of this document.** It describes a system that made LIBERO simulator
> evaluation of a VLA policy dramatically faster (from "one env stepped at a time,
> GPU idle" to "GPU running near-saturated batched inference"). It is written so a
> **downstream agent can re-implement the same parallel evaluation loop on a
> DIFFERENT benchmark**. The full source of every core file is embedded verbatim at
> the end. Read §4 (Interface contracts) and §5 (Porting checklist) first if your
> goal is to port — those are the seams you must change; everything else is generic.

Source lives in `examples/simBenchmarks/LIBERO/eval_files/bar/` (+ the model-agnostic
server wrapper in `deployment/model_server/`).

---

## 1. The performance problem this solves

A VLA policy forward pass on a 0.8B model costs roughly the same wall-clock at batch
size 1 as at batch size 30 — it is **latency-bound, not throughput-bound**, at small
batch. Naive sim eval steps ONE environment, calls the policy (batch 1), steps the
env, repeats. The GPU sits at single-digit utilization; the sim (CPU physics +
software rendering) blocks the GPU and vice-versa. On hundreds of rollouts × tasks ×
suites this is brutally slow.

The fix is three ideas stacked:
1. **Batch inference across many environments** so one GPU forward serves N envs.
2. **Chunked action prediction** so each forward yields T timesteps (fewer forwards).
3. **Vectorized envs + async pipeline** so CPU sim never idles waiting on the GPU and
   the GPU never idles waiting on the slowest env.

---

## 2. Three-layer architecture

```
                         ┌───────────────────────────────────────────────┐
                         │  ORCHESTRATION   run_eval_multigpu.sh           │
                         │  one server per GPU + client shards on a queue  │
                         └───────────────────────────────────────────────┘
        GPU 0                         GPU 1                     ... GPU 7
   ┌─────────────┐               ┌─────────────┐
   │ POLICY      │  ws://:10300  │ POLICY      │   ← starVLA conda env (torch, GPU)
   │ SERVER      │◄──────────────│ SERVER      │     serve_policy_batched.py
   │ DynamicBatcher              │             │     + PolicyServerWrapper
   └─────▲───────┘               └─────────────┘
         │ merged batch of examples from ALL connected clients
         │ (msgpack over websocket)
   ┌─────┴───────────────────────────────────────────┐
   │ CLIENT (one per (suite,task) shard)               │  ← libero conda env (sim, CPU, OSMesa)
   │  MultiPolicyClient  — N ws connections, per-env    │     eval_libero_parallel.py
   │                       action-chunk cache           │
   │  LiberoAsyncVectorEnv — N sim envs in subprocesses │     libero_vector_env.py
   └────────────────────────────────────────────────────┘
```

- **Server** (`serve_policy_batched.py`, starVLA env, GPU): one websocket server per
  GPU. A `DynamicBatcher` collects inference requests from **all connected clients**
  and dispatches them as micro-batches to a single GPU forward.
- **Client** (`eval_libero_parallel.py` + `libero_vector_env.py`, libero env, CPU):
  each client owns one (suite, task) shard and runs `num_parallel` sim environments
  in subprocesses, keeping them all busy against the shared server.
- **Orchestration** (`run_eval_multigpu.sh`): boots one server per GPU, builds a
  round-robin queue of (suite, task) shards, launches client workers, aggregates a
  `summary.json`.

**Why two conda envs / two processes:** the policy stack (torch/CUDA, transformers
5.x) and the simulator stack (LIBERO/robosuite/mujoco, gym, numpy pinned) have
conflicting deps AND compete for CPU. Splitting them into separate processes (talking
over a local websocket) lets sim physics and GPU inference overlap instead of
blocking each other, and sidesteps the dependency conflict entirely.

---

## 3. The five core mechanisms

### 3.1 Server-side dynamic batching (the biggest win)
`DynamicBatcher` (in `serve_policy_batched.py`) holds an `asyncio.Queue` of pending
requests. A single worker loop dispatches a batch when **either**:
- total queued examples ≥ `max_batch_size`, **or**
- `max_wait_ms` has elapsed since the first request in the current batch.

It then concatenates all requests' `examples` into one list, runs **one**
`wrapper.predict_action(examples=merged)` on a `ThreadPoolExecutor(max_workers=1)`
(keeps the GPU single-threaded/deterministic), and scatters the per-request slices
back to each caller's `asyncio.Future`. Requests from *different clients on the same
GPU* are merged too — so a GPU serving several (suite,task) clients batches across all
of them.

### 3.2 Chunked action prediction
The policy returns a `(T, action_dim)` **chunk**, not a single action. The client
caches it (`MultiPolicyClient.chunks[idx]`, `chunk_steps[idx]`) and steps the env `T`
times, consuming one cached action per step, before asking for a recompute. This cuts
the number of GPU forwards by ~T× (here T = action_horizon = 8).

### 3.3 Vectorized envs with selective stepping
`LiberoAsyncVectorEnv` runs `num_parallel` sim envs, each in its **own subprocess**
connected by a `Pipe` (`ctx.Process` + `CloudpickleWrapper` to ship the env factory).
`step_selective(active_mask, actions)` steps only the subset of envs that have an
action ready this tick, leaving the rest untouched — essential because episodes have
different lengths and different chunk-refresh phases.

### 3.4 Drain-then-recompute (batch-maximizing synchronization)
Per outer tick, the client loop:
1. **Drain:** repeatedly step every env that still has cached actions (no server
   call), plus warmup envs, *until every "ready" env has exhausted its chunk*.
2. **Recompute:** now that a maximal set of envs all need a fresh chunk, build one
   `example` per env and send them **concurrently** (`asyncio.gather` in
   `infer_batch`) so the server's `DynamicBatcher` sees them as a **single big
   batch**, not a dribble of size-1 requests.

This alignment is what keeps the server batch full.

### 3.5 Auto-reset trial pipeline
Each of the `num_parallel` env slots is a worker that cycles through the task's trials
(`assign_trial` / `finish_episode`). The instant an episode ends (success via
`info["_done"]`, or failure via step-budget exceeded), the slot records the result and
**immediately picks up the next trial** (`set_init_state_single`) — no slot waits for
the slowest episode, so CPU and GPU stay saturated across all `num_trials`.

---

## 4. Interface contracts (the seams — reimplement these per benchmark)

The system is deliberately split at four clean seams. To port, you swap the
bench-specific pieces and **keep the generic core** (§5 lists which is which).

### 4.1 Policy contract — `predict_action(examples) -> {"normalized_actions": (B,T,D)}`
`PolicyServerWrapper` (`deployment/model_server/policy_wrapper.py`) is **model-
agnostic**: it loads any framework via `baseframework.from_pretrained(ckpt)` and calls
`framework.predict_action(examples=[...])`, then un-normalizes with a
`PolicyNormProcessor`. **Any policy that accepts a list of `example` dicts and returns
a batched action chunk drops in unchanged.** The server never knows what LIBERO is.

### 4.2 The `example` dict — contract between client `build_example` and the framework
The unit of batching. Client `build_example(obs, task_description)` produces it; the
framework's `predict_action` consumes it. In this repo an example is roughly:
`{"image": [main_view_PIL_or_HWC_uint8, wrist_view_...], "lang": task_description}`.
Whatever keys your policy expects, `build_example` must emit — this is the ONLY schema
both sides must agree on.

### 4.3 Env worker contract — Pipe command protocol
`LiberoAsyncVectorEnv._worker` speaks a tiny command protocol over the Pipe:
`("reset", kwargs)`, `("step", action) -> (obs, reward, done, info)`,
`("set_init_state", state)`, `close`. Errors go to an `error_queue` + a per-worker
`/tmp/*_error.log`. Swap the env construction and these four handlers; keep the
subprocess/Pipe machinery.

### 4.4 Wire protocol — websocket + msgpack_numpy
`deployment/model_server/tools/msgpack_numpy.py` packs numpy arrays over a websocket.
Handshake: on connect the server sends `wrapper.metadata` (action_chunk_size, unnorm
keys, etc.); message types are `infer`/`predict_action` and `ping`. Generic — do not
touch.

---

## 5. Porting checklist — LIBERO-specific vs generic

**REPLACE (bench-specific):**
1. **Env factory + worker** (`_make_env_fn`, `_worker` in `libero_vector_env.py`,
   and `_make_env_fn` in `eval_libero_parallel.py`): create your sim env, its
   `reset`, `step -> (obs,reward,done,info)`, and init-state application.
2. **Obs → example** (`extract_rotated_images`, `build_example`): your camera keys,
   image resolution, any 180° rotation (LIBERO-specific!), proprio, and the exact
   `example` dict keys your policy wants.
3. **Model action → env action** (`to_libero_action`): action-space mapping and
   **gripper convention** (LIBERO uses dim6 ∈ [-1,1]; your bench/policy may differ —
   this is the single easiest thing to get silently wrong).
4. **Success signal & step budgets** (`info["_done"]`, `MAX_STEPS_PER_SUITE`): how you
   detect success and cap episode length per task family.
5. **Init states / seeding**: LIBERO ships `initial_states` per task; provide your
   bench's deterministic reset scheme.
6. **Task/suite iteration & shard queue** (`run_eval_multigpu.sh`): the list of
   (suite, task) shards and per-GPU queue.
7. **Render backend**: 4090 EGL is broken → `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
   Your sim may prefer EGL; warm the GL stack **before** forking workers either way.

**KEEP (generic core — reuse verbatim):**
- `DynamicBatcher` and the whole server (`serve_policy_batched.py`).
- `PolicyServerWrapper` (`policy_wrapper.py`) — as long as your policy exposes
  `predict_action(examples)`.
- `MultiPolicyClient` (chunk cache + `infer_batch` concurrent send).
- The `run_parallel_task` control loop skeleton: drain-then-recompute + auto-reset
  trial pipeline + selective stepping.
- Websocket + `msgpack_numpy` transport and the handshake.

---

## 6. Data flow of one control tick (client side)

```
for each outer tick (until all trials done):
  1. any env past its step budget  -> finish_episode(fail); assign next trial
  2. DRAIN: while some ready env still has a cached action:
        step_selective(those envs, their cached actions)   # no GPU call
        (warmup envs step a dummy action in parallel)
  3. RECOMPUTE: collect all ready envs whose chunk is exhausted
        examples = [build_example(obs_i, task) for i in those]
        client.infer_batch(those, examples)   # concurrent ws sends -> ONE server batch
  4. consume first action of each fresh chunk -> step_selective(...)
  5. on info["_done"]: finish_episode(success); assign next trial (auto-reset)
```

Server side, concurrently: `DynamicBatcher` merges whatever arrived within
`max_wait_ms` (or up to `max_batch_size`) into one `predict_action` forward and
scatters results back.

---

## 7. Tuning knobs & gotchas

- `num_parallel` — sim envs per client (↑ batch fullness, ↑ RAM/CPU).
- `max_batch_size` — server dispatch threshold (default = `num_parallel × clients_per_gpu`).
- `max_wait_ms` — max wait for a partial batch. Clients send recompute requests in a
  **burst**, so a long wait is wasted; default 200ms.
- `clients_per_gpu` — how many (suite,task) shards share one GPU server.
- **OSMesa** required on 4090 (EGL lacks PLATFORM_DEVICE); warm it before fork.
- **Two interpreters**: `PYTHON_SERVER` = starVLA env, `PYTHON_CLIENT` = libero env.
- **Gripper convention** mismatch is the classic silent bug — assert your oracle/raw
  replay hits ~100% success as a harness sanity check before trusting any policy number.
- One ckpt can still use all GPUs: the server design shards (suite,task) across GPUs.

---

## 8. How to run (reference)

```bash
# one checkpoint, spread across GPUs (each (suite,task) shard -> a GPU's server)
bash examples/simBenchmarks/LIBERO/eval_files/bar/run_eval_multigpu.sh \
    playground/Checkpoints/<run_id>/checkpoints/steps_30000_pytorch_model.pt \
    --gpu_ids 0,1,2,3,4,5,6,7 --num_trials 50 --num_parallel 10
# results: playground/Checkpoints/<run_id>/parallel_eval/steps_30000/summary.json
```

---

## 9. Full source code (verbatim)

The files below are the complete implementation, copied verbatim. Order:
1. `serve_policy_batched.py`      — batched websocket server + DynamicBatcher
2. `deployment/model_server/policy_wrapper.py` — model-agnostic policy wrapper (the seam)
3. `deployment/model_server/tools/msgpack_numpy.py` — wire serialization
4. `libero_vector_env.py`         — subprocess vector env + Pipe protocol
5. `eval_libero_parallel.py`      — vector-env client: chunk cache, drain-then-recompute, auto-reset
6. `run_eval_multigpu.sh`         — orchestration: one server per GPU + shard queue

### `examples/simBenchmarks/LIBERO/eval_files/bar/serve_policy_batched.py`

```python
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Websocket policy server with dynamic cross-client batch inference.

Drop-in alternative to `deployment/model_server/server_policy.py` for parallel
eval: same msgpack protocol (metadata handshake, then
``{"type": "infer", "examples": [...], "unnorm_key": ...}`` ->
``{"status": "ok", "data": {"actions": (B, T, D)}}``), so the existing serial
`ModelClient` also works against it unchanged.

The difference is throughput: requests from *all* connected clients are pushed
into a DynamicBatcher (design ported from GalaxeaVLA
`scripts/serve_policy_batched.py`) and dispatched as one
``PolicyServerWrapper.predict_action(examples=[...])`` call — starVLA
frameworks natively accept a list of examples — once either trigger fires:
  - total queued examples >= --max_batch_size, or
  - --max_wait_ms elapsed since the oldest queued request.

GPU inference runs in a single ThreadPoolExecutor worker so the asyncio loop
stays responsive (websocket pings keep flowing during long forwards).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import websockets.asyncio.server
import websockets.frames

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools import msgpack_numpy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PendingRequest:
    examples: List[dict]
    unnorm_key: Optional[str]
    future: asyncio.Future
    enqueued_at: float


@dataclasses.dataclass
class RequestOutcome:
    request: PendingRequest
    actions: Optional[np.ndarray] = None  # (B, T, D) for this request's examples
    error: Optional[Exception] = None


class QueueFullError(RuntimeError):
    pass


class DynamicBatcher:
    """Collects inference requests and dispatches them as micro-batches.

    A single asyncio task (`_worker_loop`) consumes from an asyncio.Queue.
    Requests accumulate until either
      - total examples >= max_batch_size, or
      - max_wait_ms elapsed since the first queued request,
    then the whole batch goes to a ThreadPoolExecutor for one GPU forward and
    each client's future is resolved with its slice of the result.
    """

    def __init__(
        self,
        wrapper: PolicyServerWrapper,
        *,
        max_batch_size: int = 30,
        max_wait_ms: float = 1000.0,
        max_queue_size: int = 256,
    ) -> None:
        self.wrapper = wrapper
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue(maxsize=max(1, int(max_queue_size)))
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="policy-batcher"
        )
        self.worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)
        while not self.queue.empty():
            req = self.queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("Dynamic batcher stopped"))

    def submit(self, examples: List[dict], unnorm_key: Optional[str]) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        req = PendingRequest(
            examples=examples, unnorm_key=unnorm_key, future=future, enqueued_at=time.monotonic()
        )
        try:
            self.queue.put_nowait(req)
        except asyncio.QueueFull as exc:
            future.cancel()
            raise QueueFullError("Inference queue is full") from exc
        return future

    @staticmethod
    def _n_examples(pending: List[PendingRequest]) -> int:
        return sum(len(r.examples) for r in pending)

    async def _worker_loop(self) -> None:
        while True:
            pending: List[PendingRequest] = []
            try:
                pending.append(await self.queue.get())

                # Drain whatever already accumulated while the previous batch
                # occupied the GPU.
                while not self.queue.empty() and self._n_examples(pending) < self.max_batch_size:
                    pending.append(self.queue.get_nowait())

                # Below threshold: wait up to max_wait_s for more to arrive.
                if self._n_examples(pending) < self.max_batch_size:
                    deadline = time.monotonic() + self.max_wait_s
                    while self._n_examples(pending) < self.max_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            pending.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                        except asyncio.TimeoutError:
                            break

                t0 = time.monotonic()
                loop = asyncio.get_running_loop()
                outcomes = await loop.run_in_executor(self.executor, self._process_batch, pending)
                logger.info(
                    "Batch forward: %d examples from %d requests, waited=%.0fms, infer=%.0fms",
                    self._n_examples(pending),
                    len(pending),
                    (t0 - pending[0].enqueued_at) * 1000.0,
                    (time.monotonic() - t0) * 1000.0,
                )

                for outcome in outcomes:
                    if outcome.request.future.done():
                        continue
                    if outcome.error is not None:
                        outcome.request.future.set_exception(outcome.error)
                    else:
                        outcome.request.future.set_result(outcome.actions)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("DynamicBatcher worker loop error")
                for req in pending:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("Batcher worker loop error"))
                await asyncio.sleep(0.1)

    def _process_batch(self, requests: List[PendingRequest]) -> List[RequestOutcome]:
        """Runs in the executor thread. Groups by unnorm_key (in practice all
        LIBERO clients share the single 'franka' key, so this is one group)."""
        outcomes: List[RequestOutcome] = []
        groups: Dict[Any, List[PendingRequest]] = {}
        for req in requests:
            groups.setdefault(req.unnorm_key, []).append(req)

        for unnorm_key, group in groups.items():
            merged: List[dict] = []
            for req in group:
                merged.extend(req.examples)
            try:
                out = self.wrapper.predict_action(examples=merged, unnorm_key=unnorm_key)
                actions = np.asarray(out["actions"])  # (sum_B, T, D)
            except Exception as exc:
                logger.exception("Batched inference failed (unnorm_key=%s)", unnorm_key)
                outcomes.extend(RequestOutcome(request=r, error=exc) for r in group)
                continue
            offset = 0
            for req in group:
                n = len(req.examples)
                outcomes.append(RequestOutcome(request=req, actions=actions[offset : offset + n]))
                offset += n
        return outcomes


async def handler(
    ws: websockets.asyncio.server.ServerConnection,
    batcher: DynamicBatcher,
    metadata: dict,
    request_timeout_s: float,
) -> None:
    packer = msgpack_numpy.Packer()
    logger.info("Connection from %s opened", ws.remote_address)
    await ws.send(packer.pack(metadata))

    try:
        async for raw in ws:
            msg = msgpack_numpy.unpackb(raw)
            req_id = msg.get("request_id", "default")
            mtype = msg.get("type", "infer")
            payload = msg.get("payload", msg)

            if mtype == "ping":
                await ws.send(packer.pack({"status": "ok", "ok": True, "type": "ping", "request_id": req_id}))
                continue

            if mtype not in ("infer", "predict_action"):
                await ws.send(
                    packer.pack(
                        {
                            "status": "error",
                            "ok": False,
                            "type": "unknown",
                            "request_id": req_id,
                            "error": {"message": f"Unsupported message type '{mtype}'"},
                        }
                    )
                )
                continue

            try:
                examples = payload["examples"]
                if not isinstance(examples, list) or not examples:
                    raise ValueError("`examples` must be a non-empty list")
                future = batcher.submit(examples, payload.get("unnorm_key"))
                actions = await asyncio.wait_for(future, timeout=request_timeout_s)
                resp = {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": req_id,
                    "data": {"actions": actions},
                }
            except Exception as exc:
                logger.exception("Inference request failed (request_id=%s)", req_id)
                resp = {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": str(exc)},
                }
            await ws.send(packer.pack(resp))
    except websockets.ConnectionClosed:
        logger.info("Connection from %s closed", ws.remote_address)


async def serve(args) -> None:
    import functools
    import torch  # noqa: F401  (ensures CUDA context init happens before serving)

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
    )
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    batcher = DynamicBatcher(
        wrapper,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        max_queue_size=args.max_queue_size,
    )
    batcher.start()

    bound = functools.partial(
        handler,
        batcher=batcher,
        metadata=wrapper.metadata,
        request_timeout_s=args.request_timeout_ms / 1000.0,
    )
    try:
        async with websockets.asyncio.server.serve(
            bound,
            args.host,
            args.port,
            compression=None,
            max_size=None,
            ping_interval=30,
            ping_timeout=300,
        ) as server:
            logger.info(
                "Batched policy server listening on ws://%s:%d (max_batch=%d, max_wait=%.0fms)",
                args.host,
                args.port,
                args.max_batch_size,
                args.max_wait_ms,
            )
            await server.serve_forever()
    finally:
        await batcher.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="starVLA batched websocket policy server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to steps_XXXX_pytorch_model.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--max_batch_size", type=int, default=30, help="Dispatch when this many examples queued")
    parser.add_argument("--max_wait_ms", type=float, default=1000.0, help="Max wait before dispatching a partial batch")
    parser.add_argument("--max_queue_size", type=int, default=256)
    parser.add_argument("--request_timeout_ms", type=float, default=300000.0)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    asyncio.run(serve(build_argparser().parse_args()))
```

### `deployment/model_server/policy_wrapper.py`

```python
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Return training image size metadata when it is explicitly configured.

    This is only a consistency check for eval clients. It is not used to infer
    camera order or to rewrite eval observations.
    """
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if size is None and "default_image_resolution" in vla_data_cfg:
        default_resolution = vla_data_cfg["default_image_resolution"]
        if isinstance(default_resolution, (list, tuple)) and len(default_resolution) >= 2:
            size = default_resolution[-2:]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "training_data_mix": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
            "training_obs_image_size": _training_obs_image_size(self._model_cfg),
            "eval_image_contract": (
                "Eval clients must explicitly choose image count and order. "
                "The server does not infer or reorder camera views from training config."
            ),
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm}
```

### `deployment/model_server/tools/msgpack_numpy.py`

```python
"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
```

### `examples/simBenchmarks/LIBERO/eval_files/bar/libero_vector_env.py`

```python
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Async vectorized LIBERO env over forkserver subprocesses.

Ported from GalaxeaVLA `experiments/libero/libero_vector_env.py` (near
verbatim). One OffScreenRenderEnv per subprocess; the parent drives them
through Pipes. `step_selective` lets a subset of envs step while the rest
receive a nop — this is what enables per-env episode slot-reuse in
`eval_libero_parallel.py` without a barrier on episode boundaries.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Any

import cloudpickle
import numpy as np

__all__ = ["LiberoAsyncVectorEnv"]


class CloudpickleWrapper:
    """Serialize an env-factory closure with cloudpickle (plain pickle can't)."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, blob):
        import pickle

        self.fn = pickle.loads(blob)

    def __call__(self):
        return self.fn()


class LiberoAsyncVectorEnv:
    def __init__(
        self,
        env_fns: list,
        dummy_env_fn=None,
        daemon: bool = True,
    ):
        try:
            mp.set_start_method("forkserver")
        except RuntimeError:
            pass

        ctx = mp.get_context("forkserver")
        self.num_envs = len(env_fns)
        self.closed = False

        # Build one env in the parent first so LIBERO/mujoco asset caches and
        # the OSMesa stack are warm before the workers fork.
        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        dummy_env.close()
        del dummy_env

        self.parent_pipes: list = []
        self.processes: list = []
        self.error_queue = ctx.Queue()

        for idx, env_fn in enumerate(env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                name=f"LiberoWorker-{idx}",
                args=(idx, CloudpickleWrapper(env_fn), child_pipe, parent_pipe, self.error_queue),
            )
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)
            process.daemon = daemon
            process.start()
            child_pipe.close()

    def reset(self, **kwargs) -> list:
        for pipe in self.parent_pipes:
            pipe.send(("reset", kwargs))
        return list(self._recv_all())

    def step(self, actions) -> tuple[list, np.ndarray, np.ndarray, list]:
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        results = self._recv_all()
        observations_list, rewards, dones, infos = zip(*results)
        return (
            list(observations_list),
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            list(infos),
        )

    def step_selective(self, active_mask: list[bool], actions: list) -> tuple:
        for i in range(self.num_envs):
            if active_mask[i]:
                self.parent_pipes[i].send(("step", actions[i]))
            else:
                self.parent_pipes[i].send(("nop", None))
        raw_results = self._recv_all()
        obs_list = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            if active_mask[i]:
                obs, reward, done, info = raw_results[i]
                info = dict(info) if info else {}
                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            else:
                obs_list.append(None)
                rewards.append(0.0)
                dones.append(False)
                infos.append({})
        return (
            obs_list,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def set_init_state_each(self, states: list) -> list:
        for pipe, state in zip(self.parent_pipes, states):
            pipe.send(("set_init_state", state))
        return list(self._recv_all())

    def set_init_state_single(self, idx: int, state) -> Any:
        self.parent_pipes[idx].send(("set_init_state", state))
        result, _success = self.parent_pipes[idx].recv()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pipe in self.parent_pipes:
            if pipe is not None and not pipe.closed:
                pipe.send(("close", None))
        for pipe in self.parent_pipes:
            if pipe is not None and not pipe.closed:
                try:
                    pipe.recv()
                except Exception:
                    pass
        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for p in self.processes:
            p.join(timeout=5)

    def _recv_all(self, timeout=None):
        results = []
        for i, pipe in enumerate(self.parent_pipes):
            if pipe is None or pipe.closed:
                raise RuntimeError(f"Worker {i} pipe is closed")
            if timeout is not None:
                if not pipe.poll(timeout):
                    raise mp.TimeoutError(f"Worker {i} response timed out after {timeout}s")
            result, success = pipe.recv()
            if not success:
                self._flush_errors()
            results.append(result)
        return results

    def _flush_errors(self):
        errors = []
        while not self.error_queue.empty():
            index, exctype, value = self.error_queue.get()
            errors.append((index, exctype, value))
            if index < len(self.parent_pipes) and self.parent_pipes[index] is not None:
                self.parent_pipes[index].close()
                self.parent_pipes[index] = None
        if errors:
            _, exctype, value = errors[-1]
            raise exctype(value)


def _worker(index, env_fn, pipe, parent_pipe, error_queue):
    try:
        env = env_fn()
    except Exception:
        import traceback

        with open(f"/tmp/libero_worker_{index}_error.log", "w") as f:
            f.write(f"Worker {index} env creation failed:\n")
            f.write(traceback.format_exc())
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
        return
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                obs = env.reset(**data)
                pipe.send((obs, True))
            elif command == "step":
                obs, reward, done, info = env.step(data)
                info = dict(info) if info else {}
                if done:
                    info["_done"] = True
                pipe.send(((obs, reward, done, info), True))
            elif command == "set_init_state":
                if data is not None:
                    env.reset()  # zero velocities from previous episode before applying init state
                    obs = env.set_init_state(data)
                    pipe.send((obs, True))
                else:
                    pipe.send((None, True))
            elif command == "nop":
                pipe.send((None, True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                fn = getattr(env, name)
                pipe.send((fn(*args, **kwargs) if callable(fn) else fn, True))
            else:
                raise RuntimeError(f"Unknown command: {command}")
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
```

### `examples/simBenchmarks/LIBERO/eval_files/bar/eval_libero_parallel.py`

```python
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Parallel LIBERO eval client for the starVLA batched policy server.

Scheduling design ported from GalaxeaVLA `experiments/libero/eval_libero_parallel.py`:
  - N env subprocesses (LiberoAsyncVectorEnv) run one task's trials with
    slot-reuse: a finished env immediately takes the next trial's init state.
  - Chunk caching is CLIENT-side (the starVLA server returns the full
    unnormalized (T, 7) chunk). Envs with cached actions are stepped first
    ("drain"), then all recompute requests are sent concurrently so the
    server's DynamicBatcher sees them as one batch.

Observation contract — must match training exactly (audited 2026-07-16 against
`gr00t_lerobot/datasets.py::_pack_sample` and the serial `eval_libero.py`):
  - both cameras rendered at 256 and rotated 180 deg ([::-1, ::-1])
  - resized to 224x224 with PIL's DEFAULT resample (= BICUBIC, same as training)
  - image order [agentview, wrist]; NO proprioceptive state
  - gripper: raw model output in [0,1] (1=open) -> env command 1 - 2*(v > 0.5)

Run inside the `libero` conda env with MUJOCO_GL=osmesa (EGL is broken on the
4090 cluster) and the starVLA repo root on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import time
from typing import Any, List, Optional

import imageio
import numpy as np
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from deployment.model_server.tools import msgpack_numpy
from examples.simBenchmarks.LIBERO.eval_files.bar.libero_vector_env import LiberoAsyncVectorEnv

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render the training data
MODEL_IMAGE_SIZE = (224, 224)  # (H, W), matches training-time resize

MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


def _json_default(obj: Any):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Observation / action adapters (the train/test-consistency-critical part)
# ---------------------------------------------------------------------------


def extract_rotated_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """256x256 uint8, rotated 180 deg to match training preprocessing."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return img, wrist


def build_example(obs: dict, task_description: str) -> dict:
    img, wrist = extract_rotated_images(obs)
    # NOTE: no explicit resample filter — PIL defaults to BICUBIC, matching the
    # training-side resize in gr00t_lerobot/datasets.py `_pack_sample`.
    img = np.asarray(Image.fromarray(img).resize((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0])))
    wrist = np.asarray(Image.fromarray(wrist).resize((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0])))
    return {"image": [img, wrist], "lang": str(task_description)}


def to_libero_action(action7: np.ndarray) -> List[float]:
    """Unnormalized model action (7,) -> LIBERO env action.

    x/y/z/roll/pitch/yaw pass through; gripper is binarized with the same
    openvla-lineage inverse transform as the serial eval client:
    v > 0.5 (open in dataset convention) -> -1.0 (open in LIBERO).
    """
    a = np.asarray(action7, dtype=np.float32).reshape(-1)
    if a.size != 7:
        raise ValueError(f"Expected 7-dim action, got shape {a.shape}")
    gripper = 1.0 - 2.0 * (float(a[6]) > 0.5)
    return [float(x) for x in a[:6]] + [float(gripper)]


# ---------------------------------------------------------------------------
# Async websocket client with per-env chunk cache
# ---------------------------------------------------------------------------


class PolicyConnection:
    def __init__(self, uri: str, timeout_s: float):
        self.uri = uri
        self.timeout_s = timeout_s
        self._ws = None
        self._packer = msgpack_numpy.Packer()
        self.metadata: Optional[dict] = None

    async def connect(self) -> dict:
        import websockets.asyncio.client

        self._ws = await websockets.asyncio.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            ping_interval=30,
            ping_timeout=300,
        )
        handshake = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        self.metadata = msgpack_numpy.unpackb(handshake)
        return self.metadata

    async def infer(self, examples: List[dict], unnorm_key: Optional[str]) -> np.ndarray:
        """Returns the unnormalized action chunk (T, 7) for a single example."""
        payload = {"type": "infer", "examples": examples, "unnorm_key": unnorm_key}
        await asyncio.wait_for(self._ws.send(self._packer.pack(payload)), timeout=self.timeout_s)
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        if isinstance(raw, str):
            raise RuntimeError(f"Server error:\n{raw}")
        resp = msgpack_numpy.unpackb(raw)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Inference failed: {resp.get('error')}")
        return np.asarray(resp["data"]["actions"])[0]  # (T, 7)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class MultiPolicyClient:
    """One ws connection per parallel env + client-side chunk caches."""

    def __init__(self, uri: str, num_connections: int, timeout_s: float, unnorm_key: Optional[str]):
        self.connections = [PolicyConnection(uri, timeout_s) for _ in range(num_connections)]
        self.unnorm_key = unnorm_key
        self.chunks: List[Optional[np.ndarray]] = [None] * num_connections
        self.chunk_steps: List[int] = [0] * num_connections
        self.metadata: Optional[dict] = None

    async def connect_all(self) -> dict:
        metas = await asyncio.gather(*(c.connect() for c in self.connections))
        self.metadata = metas[0]
        return self.metadata

    async def close_all(self) -> None:
        await asyncio.gather(*(c.close() for c in self.connections))

    def needs_recompute(self, idx: int) -> bool:
        chunk = self.chunks[idx]
        return chunk is None or self.chunk_steps[idx] >= chunk.shape[0]

    def get_cached_action(self, idx: int) -> Optional[np.ndarray]:
        if self.needs_recompute(idx):
            return None
        action = self.chunks[idx][self.chunk_steps[idx]]
        self.chunk_steps[idx] += 1
        return action

    def clear_chunk(self, idx: int) -> None:
        self.chunks[idx] = None
        self.chunk_steps[idx] = 0

    async def infer_batch(self, indices: List[int], examples_list: List[dict]) -> None:
        chunks = await asyncio.gather(
            *(
                self.connections[i].infer([examples_list[j]], self.unnorm_key)
                for j, i in enumerate(indices)
            )
        )
        for j, i in enumerate(indices):
            self.chunks[i] = np.asarray(chunks[j])
            self.chunk_steps[i] = 0


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def _make_env_fn(task, resolution: int, seed: int):
    from libero.libero import get_libero_path

    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    def _fn():
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
        # IMPORTANT: seed affects object positions even with fixed init states
        # (same convention as the serial eval_libero.py).
        env.seed(seed)
        return env

    return _fn


def save_rollout_video(video_dir: pathlib.Path, task_description: str, trial_id: int, success: bool, frames: list) -> None:
    if not frames:
        return
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")
    imageio.mimwrite(
        video_dir / f"rollout_{task_segment}_episode{trial_id}_{suffix}.mp4",
        [np.asarray(f) for f in frames],
        fps=10,
    )


# ---------------------------------------------------------------------------
# Parallel rollout for one task
# ---------------------------------------------------------------------------


async def run_parallel_task(
    task_id: int,
    task_description: str,
    initial_states,
    num_trials: int,
    max_steps: int,
    num_steps_wait: int,
    num_parallel: int,
    client: MultiPolicyClient,
    vec_env: LiberoAsyncVectorEnv,
    save_videos: bool,
    output_dir: pathlib.Path,
) -> dict:
    trial_successes: dict[int, bool] = {}
    trial_replay_images: dict[int, list] = {}
    completed = 0
    next_trial_idx = 0
    total_infer_calls = 0

    env_trial_id = [0] * num_parallel  # -1 = idle (no trials left)
    env_step_count = [0] * num_parallel
    env_replay: List[list] = [[] for _ in range(num_parallel)]

    total_steps = max_steps + num_steps_wait

    def assign_trial(env_idx: int):
        nonlocal next_trial_idx
        trial_id = next_trial_idx
        state_idx = trial_id % len(initial_states)
        next_trial_idx += 1
        env_trial_id[env_idx] = trial_id
        env_step_count[env_idx] = 0
        env_replay[env_idx] = []
        return trial_id, initial_states[state_idx]

    def finish_episode(env_idx: int, success: bool):
        """Record the outcome; return the next trial's init state (or None)."""
        nonlocal completed
        tid = env_trial_id[env_idx]
        if tid < 0 or tid in trial_successes:
            return None
        trial_successes[tid] = success
        if save_videos:
            trial_replay_images[tid] = list(env_replay[env_idx])
        completed += 1
        logger.info(
            "task %d trial %d done: success=%s (%d/%d completed)",
            task_id, tid, success, completed, num_trials,
        )
        env_step_count[env_idx] = 0
        env_replay[env_idx] = []
        client.clear_chunk(env_idx)
        if next_trial_idx < num_trials:
            _, state = assign_trial(env_idx)
            return state
        env_trial_id[env_idx] = -1
        return None

    async def step_envs(active_mask: List[bool], actions: list, per_env_obs: list):
        """Step the masked envs; handle done/new-trial bookkeeping."""
        obs_batch, _, _, infos = await asyncio.to_thread(vec_env.step_selective, active_mask, actions)
        for i in range(num_parallel):
            if not active_mask[i]:
                continue
            if save_videos and env_step_count[i] >= num_steps_wait:
                # frame the policy saw at this step (pre-step obs, rotated, 256px)
                env_replay[i].append(extract_rotated_images(per_env_obs[i])[0])
            if obs_batch[i] is not None:
                per_env_obs[i] = obs_batch[i]
            env_step_count[i] += 1
            if infos[i].get("_done"):
                client.clear_chunk(i)
                new_state = finish_episode(i, True)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(vec_env.set_init_state_single, i, new_state)

    # --- seed the first num_parallel trials ---
    init_states = []
    for i in range(num_parallel):
        _, state = assign_trial(i)
        init_states.append(state)
    await asyncio.to_thread(vec_env.reset)
    obs_batch = await asyncio.to_thread(vec_env.set_init_state_each, init_states)
    per_env_obs: List[Optional[dict]] = list(obs_batch)

    started = time.time()
    while completed < num_trials:
        if next_trial_idx >= num_trials and all(t < 0 for t in env_trial_id):
            break

        # --- episodes that ran out of steps -> failure, reuse the slot ---
        for i in range(num_parallel):
            if env_trial_id[i] >= 0 and env_step_count[i] >= total_steps:
                new_state = finish_episode(i, False)
                if new_state is not None:
                    per_env_obs[i] = await asyncio.to_thread(vec_env.set_init_state_single, i, new_state)

        active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
        if not active_envs:
            continue

        dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
        ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # --- drain: step every env that still has cached actions (no server
        # call), stepping warmup envs alongside, until all ready envs need a
        # recompute. This maximizes the batch the server sees next. ---
        while True:
            cached_envs = [
                i
                for i in ready_envs
                if env_trial_id[i] >= 0
                and env_step_count[i] < total_steps
                and not client.needs_recompute(i)
            ]
            if not cached_envs and not (dummy_envs and not ready_envs):
                break

            active_mask = [False] * num_parallel
            actions: list = [LIBERO_DUMMY_ACTION] * num_parallel
            for i in dummy_envs:
                if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                    active_mask[i] = True
            for i in cached_envs:
                action = client.get_cached_action(i)
                if action is None:
                    continue
                actions[i] = to_libero_action(action)
                active_mask[i] = True
            if not any(active_mask):
                break

            await step_envs(active_mask, actions, per_env_obs)

            active_envs = [i for i in range(num_parallel) if env_trial_id[i] >= 0]
            dummy_envs = [i for i in active_envs if env_step_count[i] < num_steps_wait]
            ready_envs = [i for i in active_envs if env_step_count[i] >= num_steps_wait]

        # --- recompute: all ready envs have exhausted their cache; send every
        # request concurrently so the server batches them in one forward. ---
        recompute_envs = [
            i
            for i in range(num_parallel)
            if env_trial_id[i] >= 0
            and num_steps_wait <= env_step_count[i] < total_steps
            and client.needs_recompute(i)
        ]
        if not recompute_envs:
            await asyncio.sleep(0)
            continue

        examples = [build_example(per_env_obs[i], task_description) for i in recompute_envs]
        await client.infer_batch(recompute_envs, examples)
        total_infer_calls += len(recompute_envs)

        # consume the first action of each fresh chunk (+ step warmup envs)
        active_mask = [False] * num_parallel
        actions = [LIBERO_DUMMY_ACTION] * num_parallel
        for i in dummy_envs:
            if env_trial_id[i] >= 0 and env_step_count[i] < num_steps_wait:
                active_mask[i] = True
        for i in recompute_envs:
            action = client.get_cached_action(i)
            if action is None:
                logger.warning("env %d has no cached action after infer, skipping step", i)
                continue
            actions[i] = to_libero_action(action)
            active_mask[i] = True
        if not any(active_mask):
            await asyncio.sleep(0)
            continue
        await step_envs(active_mask, actions, per_env_obs)

    successes = sum(1 for v in trial_successes.values() if v)
    task_result = {
        "task_id": task_id,
        "task_description": task_description,
        "successes": successes,
        "total_episodes": num_trials,
        "success_rate": successes / num_trials if num_trials else 0.0,
        "infer_calls": total_infer_calls,
        "wall_seconds": round(time.time() - started, 1),
    }

    if save_videos:
        for tid, success in trial_successes.items():
            save_rollout_video(
                output_dir / "videos", task_description, tid, success, trial_replay_images.get(tid, [])
            )

    return task_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def evaluate(args) -> dict:
    from libero.libero import benchmark

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS_BY_SUITE[args.task_suite_name]
    task_ids = list(range(task_suite.n_tasks)) if args.task_id is None else [args.task_id]

    summary: dict = {
        "server_uri": args.server_uri,
        "task_suite_name": args.task_suite_name,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "num_parallel": args.num_parallel,
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "total_infer_calls": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    client = MultiPolicyClient(
        args.server_uri, args.num_parallel, timeout_s=args.timeout_s, unnorm_key=args.unnorm_key
    )
    meta = await client.connect_all()
    logger.info(
        "Connected to %s with %d connections; server metadata: %s",
        args.server_uri, args.num_parallel, meta,
    )
    summary["server_metadata"] = {
        k: meta.get(k)
        for k in ("ckpt_path", "action_chunk_size", "default_unnorm_key", "training_data_mix")
    }

    try:
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_description = task.language
            logger.info("=== task %d: %s ===", task_id, task_description)

            env_fn = _make_env_fn(task, args.env_resolution, args.seed)
            vec_env = await asyncio.to_thread(
                LiberoAsyncVectorEnv, env_fns=[env_fn] * args.num_parallel
            )
            try:
                task_result = await run_parallel_task(
                    task_id=task_id,
                    task_description=task_description,
                    initial_states=initial_states,
                    num_trials=args.num_trials_per_task,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    num_parallel=args.num_parallel,
                    client=client,
                    vec_env=vec_env,
                    save_videos=args.save_videos,
                    output_dir=pathlib.Path(args.output_dir),
                )
            finally:
                await asyncio.to_thread(vec_env.close)

            summary["tasks"].append(task_result)
            summary["total_successes"] += task_result["successes"]
            summary["total_episodes"] += task_result["total_episodes"]
            summary["total_infer_calls"] += task_result["infer_calls"]
            logger.info(
                "task %d success rate: %.3f (%d/%d) in %.0fs",
                task_id,
                task_result["success_rate"],
                task_result["successes"],
                task_result["total_episodes"],
                task_result["wall_seconds"],
            )
    finally:
        await client.close_all()

    summary["success_rate"] = (
        summary["total_successes"] / summary["total_episodes"] if summary["total_episodes"] else 0.0
    )
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel LIBERO eval against a starVLA policy server")
    parser.add_argument("--server_uri", default="ws://127.0.0.1:10093")
    parser.add_argument("--task_suite_name", default="libero_goal", choices=sorted(MAX_STEPS_BY_SUITE))
    parser.add_argument("--task_id", type=int, default=None, help="Single task id; default = all tasks in suite")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_parallel", type=int, default=10)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--timeout_s", type=float, default=600.0)
    parser.add_argument("--env_resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    parser.add_argument("--output_dir", default="outputs/libero_parallel_eval")
    parser.add_argument("--save_videos", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(evaluate(args))

    summary_path = output_dir / f"{args.task_suite_name}_parallel_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info(
        "Total success rate: %.3f (%d/%d)",
        summary["success_rate"], summary["total_successes"], summary["total_episodes"],
    )
    logger.info("Results saved to %s", summary_path)


if __name__ == "__main__":
    main()
```

### `examples/simBenchmarks/LIBERO/eval_files/bar/run_eval_multigpu.sh`

```bash
#!/bin/bash
# Multi-GPU parallel LIBERO evaluation for ONE starVLA checkpoint.
# Design ported from GalaxeaVLA scripts/run/eval_libero_multigpu.sh:
#   one batched policy server per GPU  +  per-GPU client slots that pull
#   (suite, task) shards off a tsv queue, each running num_parallel envs.
#
# Usage:
#   bash examples/simBenchmarks/LIBERO/eval_files/bar/run_eval_multigpu.sh \
#       playground/Checkpoints/<run_id>/checkpoints/steps_30000_pytorch_model.pt [options]
#
# Options:
#   --output_dir DIR       (default: <run_dir>/parallel_eval/<steps_XXXX>)
#   --base_port PORT       (default: 10300)
#   --num_gpus N           (default: all visible)
#   --gpu_ids IDS          space/comma-separated physical GPU ids
#   --clients_per_gpu N    concurrent task clients per GPU (default: 1)
#   --num_trials N         trials per task (default: 50)
#   --num_parallel N       parallel envs per task client (default: 10)
#   --num_steps_wait N     warmup steps (default: 10, matches serial eval)
#   --max_batch_size N     server batch trigger (default: num_parallel*clients_per_gpu)
#   --max_wait_ms N        server partial-batch wait (default: 200; client bursts
#                          all recompute requests together so a long wait is waste)
#   --suites "A B"         (default: libero_spatial libero_object libero_goal libero_10)
#   --task_ids IDS         (default: 0..9)
#   --save_videos          save rollout mp4s
#
# The sim client MUST use OSMesa: EGL headless is broken on the 4090 cluster
# (driver lacks PLATFORM_DEVICE ext) — same workaround GalaxeaVLA used on H100.

set -euo pipefail

export HOME=/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"
# $REPO_ROOT/LIBERO: the `libero` pip editable install in the libero env has a
# broken (empty) finder mapping, so the package must come from PYTHONPATH; this
# tree is also what ~/.libero/config.yaml points bddl/init_states at.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_SERVER="$HOME/miniconda3/envs/starVLA/bin/python"
PYTHON_CLIENT="$HOME/miniconda3/envs/libero/bin/python"

CKPT_PATH=""
OUTPUT_DIR=""
BASE_PORT=10300
NUM_GPUS=""
GPU_IDS_RAW=""
CLIENTS_PER_GPU=1
NUM_TRIALS=50
NUM_PARALLEL=10
NUM_STEPS_WAIT=10
MAX_BATCH_SIZE=""
MAX_WAIT_MS=200
SAVE_VIDEOS=false
SUITES=(libero_spatial libero_object libero_goal libero_10)
TASK_IDS_RAW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --base_port|--port) BASE_PORT="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpu_ids) GPU_IDS_RAW="$2"; shift 2 ;;
        --clients_per_gpu) CLIENTS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --num_parallel) NUM_PARALLEL="$2"; shift 2 ;;
        --num_steps_wait) NUM_STEPS_WAIT="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_wait_ms) MAX_WAIT_MS="$2"; shift 2 ;;
        --save_videos) SAVE_VIDEOS=true; shift ;;
        --suites) read -ra SUITES <<< "$2"; shift 2 ;;
        --task_ids) TASK_IDS_RAW="$2"; shift 2 ;;
        *)
            if [[ -z "$CKPT_PATH" ]]; then
                CKPT_PATH="$1"
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$CKPT_PATH" || ! -f "$CKPT_PATH" ]]; then
    echo "Usage: bash $0 <path/to/steps_XXXX_pytorch_model.pt> [options]" >&2
    exit 1
fi

if [[ -z "$MAX_BATCH_SIZE" ]]; then
    MAX_BATCH_SIZE=$((NUM_PARALLEL * CLIENTS_PER_GPU))
fi

DETECTED_GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
GPU_IDS=()
if [[ -n "$GPU_IDS_RAW" ]]; then
    GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
    read -ra GPU_IDS <<< "$GPU_IDS_RAW"
    NUM_GPUS="${#GPU_IDS[@]}"
else
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS="$DETECTED_GPU_COUNT"
    fi
    for ((i = 0; i < NUM_GPUS; i++)); do
        GPU_IDS+=("$i")
    done
fi
if (( NUM_GPUS < 1 )); then
    echo "ERROR: no GPUs selected" >&2
    exit 1
fi

if [[ -z "$TASK_IDS_RAW" ]]; then
    TASK_IDS=(0 1 2 3 4 5 6 7 8 9)
else
    TASK_IDS_RAW="${TASK_IDS_RAW//,/ }"
    read -ra TASK_IDS <<< "$TASK_IDS_RAW"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    CKPT_FILE="$(basename "$CKPT_PATH")"           # steps_30000_pytorch_model.pt
    CKPT_TAG="${CKPT_FILE%_pytorch_model.pt}"      # steps_30000
    RUN_DIR="$(dirname "$(dirname "$CKPT_PATH")")" # <run_dir>
    OUTPUT_DIR="$RUN_DIR/parallel_eval/$CKPT_TAG"
fi
mkdir -p "$OUTPUT_DIR/servers" "$OUTPUT_DIR/shards" "$OUTPUT_DIR/queues"

echo "=============================================="
echo " starVLA LIBERO Multi-GPU Parallel Evaluation"
echo "=============================================="
echo " Checkpoint:       $CKPT_PATH"
echo " Output dir:       $OUTPUT_DIR"
echo " GPU ids:          ${GPU_IDS[*]}"
echo " Base port:        $BASE_PORT"
echo " Suites:           ${SUITES[*]}"
echo " Task ids:         ${TASK_IDS[*]}"
echo " Trials/task:      $NUM_TRIALS"
echo " Parallel/client:  $NUM_PARALLEL"
echo " Clients/GPU:      $CLIENTS_PER_GPU"
echo " Max batch:        $MAX_BATCH_SIZE"
echo " Max wait ms:      $MAX_WAIT_MS"
echo " Save videos:      $SAVE_VIDEOS"
echo "=============================================="

port_is_listening() {
    "$PYTHON_CLIENT" - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

SERVER_PIDS=()
SERVER_PORTS=()

cleanup_servers() {
    if [[ ${#SERVER_PIDS[@]} -gt 0 ]]; then
        echo ""
        echo "Stopping policy servers ..."
        for pid in "${SERVER_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for pid in "${SERVER_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup_servers EXIT INT TERM

echo ""
echo "[1/4] Starting ${#GPU_IDS[@]} batched policy server(s) ..."
for gpu_ordinal in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$gpu_ordinal]}"
    port=$((BASE_PORT + gpu_ordinal))
    SERVER_PORTS[$gpu_ordinal]="$port"
    log_path="$OUTPUT_DIR/servers/gpu${gpu_id}_port${port}.log"

    echo "  GPU $gpu_id -> port $port (log: $log_path)"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_SERVER" \
        examples/simBenchmarks/LIBERO/eval_files/bar/serve_policy_batched.py \
        --ckpt_path "$CKPT_PATH" \
        --host 0.0.0.0 \
        --port "$port" \
        --use_bf16 \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --max_wait_ms "$MAX_WAIT_MS" \
        &> "$log_path" &
    SERVER_PIDS[$gpu_ordinal]=$!
done

echo ""
echo "[2/4] Waiting for servers to listen ..."
for gpu_ordinal in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$gpu_ordinal]}"
    port="${SERVER_PORTS[$gpu_ordinal]}"
    pid="${SERVER_PIDS[$gpu_ordinal]}"
    ready=false
    for _ in $(seq 1 300); do
        if port_is_listening "$port"; then
            echo "  GPU $gpu_id server is ready on port $port"
            ready=true
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: server for GPU $gpu_id died. Check $OUTPUT_DIR/servers/gpu${gpu_id}_port${port}.log" >&2
            exit 1
        fi
        sleep 2
    done
    if [[ "$ready" != "true" ]]; then
        echo "ERROR: server for GPU $gpu_id not ready after 600s" >&2
        exit 1
    fi
done

echo ""
echo "[3/4] Building task queues ..."
rm -f "$OUTPUT_DIR"/queues/*.tsv
GPU_SHARD_COUNTS=()
for ((i = 0; i < NUM_GPUS; i++)); do
    GPU_SHARD_COUNTS[$i]=0
    for ((slot = 0; slot < CLIENTS_PER_GPU; slot++)); do
        : > "$OUTPUT_DIR/queues/gpu${i}_slot${slot}.tsv"
    done
done

shard_index=0
for suite in "${SUITES[@]}"; do
    for task_id in "${TASK_IDS[@]}"; do
        gpu_ordinal=$((shard_index % NUM_GPUS))
        slot=$((GPU_SHARD_COUNTS[$gpu_ordinal] % CLIENTS_PER_GPU))
        GPU_SHARD_COUNTS[$gpu_ordinal]=$((GPU_SHARD_COUNTS[$gpu_ordinal] + 1))
        shard_name="$(printf "%s_task%02d" "$suite" "$task_id")"
        printf "%s\t%s\t%s\n" "$suite" "$task_id" "$shard_name" >> "$OUTPUT_DIR/queues/gpu${gpu_ordinal}_slot${slot}.tsv"
        shard_index=$((shard_index + 1))
    done
done

for gpu_ordinal in "${!GPU_IDS[@]}"; do
    echo "  GPU ${GPU_IDS[$gpu_ordinal]} shards: ${GPU_SHARD_COUNTS[$gpu_ordinal]}"
done

run_client_queue() {
    local gpu_ordinal="$1"
    local slot="$2"
    local gpu_id="${GPU_IDS[$gpu_ordinal]}"
    local port="${SERVER_PORTS[$gpu_ordinal]}"
    local queue_file="$OUTPUT_DIR/queues/gpu${gpu_ordinal}_slot${slot}.tsv"
    local failed=0
    local video_args=()
    if [[ "$SAVE_VIDEOS" == "true" ]]; then
        video_args=(--save_videos)
    fi

    while IFS=$'\t' read -r suite task_id shard_name; do
        if [[ -z "${suite:-}" ]]; then
            continue
        fi
        local shard_out="$OUTPUT_DIR/shards/$shard_name"
        if [[ -f "$shard_out/${suite}_parallel_results.json" ]]; then
            echo "  [GPU $gpu_id slot $slot] skip $suite task $task_id (results exist)"
            continue
        fi
        mkdir -p "$shard_out"
        echo "  [GPU $gpu_id slot $slot] start $suite task $task_id -> $shard_out"

        if env -u CUDA_VISIBLE_DEVICES \
            MUJOCO_GL=osmesa \
            PYOPENGL_PLATFORM=osmesa \
            OMP_NUM_THREADS=1 \
            "$PYTHON_CLIENT" examples/simBenchmarks/LIBERO/eval_files/bar/eval_libero_parallel.py \
            --server_uri "ws://127.0.0.1:${port}" \
            --task_suite_name "$suite" \
            --task_id "$task_id" \
            --num_trials_per_task "$NUM_TRIALS" \
            --num_steps_wait "$NUM_STEPS_WAIT" \
            --num_parallel "$NUM_PARALLEL" \
            --output_dir "$shard_out" \
            "${video_args[@]}" \
            &> "$shard_out/client.log"; then
            echo "  [GPU $gpu_id slot $slot] done $suite task $task_id"
        else
            local status=$?
            echo "  [GPU $gpu_id slot $slot] fail $suite task $task_id (status $status); check $shard_out/client.log" >&2
            printf "%s\t%s\t%s\t%s\n" "$gpu_id" "$slot" "$suite" "$task_id" >> "$OUTPUT_DIR/failed_shards.tsv"
            failed=1
        fi
    done < "$queue_file"

    return "$failed"
}

echo ""
echo "[4/4] Launching task client workers ..."
rm -f "$OUTPUT_DIR/failed_shards.tsv"
WORKER_PIDS=()
WORKER_LABELS=()
for gpu_ordinal in "${!GPU_IDS[@]}"; do
    for ((slot = 0; slot < CLIENTS_PER_GPU; slot++)); do
        queue_file="$OUTPUT_DIR/queues/gpu${gpu_ordinal}_slot${slot}.tsv"
        if [[ ! -s "$queue_file" ]]; then
            continue
        fi
        run_client_queue "$gpu_ordinal" "$slot" &
        WORKER_PIDS+=("$!")
        WORKER_LABELS+=("gpu${GPU_IDS[$gpu_ordinal]}_slot${slot}")
    done
done

FAILED_WORKERS=()
for idx in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$idx]}"
    label="${WORKER_LABELS[$idx]}"
    if wait "$pid"; then
        echo "  [DONE] worker $label"
    else
        echo "  [FAIL] worker $label" >&2
        FAILED_WORKERS+=("$label")
    fi
done

cleanup_servers
SERVER_PIDS=()

echo ""
echo "Generating summary ..."
"$PYTHON_CLIENT" - "$OUTPUT_DIR" "$CKPT_PATH" "${SUITES[*]}" "${TASK_IDS[*]}" "$NUM_GPUS" "$CLIENTS_PER_GPU" "$NUM_PARALLEL" "$NUM_TRIALS" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
ckpt_path = sys.argv[2]
suite_names = sys.argv[3].split()
task_ids = [int(x) for x in sys.argv[4].split()]
num_gpus = int(sys.argv[5])
clients_per_gpu = int(sys.argv[6])
num_parallel = int(sys.argv[7])
num_trials = int(sys.argv[8])

suite_order = {suite: i for i, suite in enumerate(suite_names)}
all_suite_results = {}
per_task_rows = []
missing = []

for suite in suite_names:
    suite_successes = 0
    suite_total = 0
    for task_id in task_ids:
        shard_name = f"{suite}_task{task_id:02d}"
        json_path = output_dir / "shards" / shard_name / f"{suite}_parallel_results.json"
        if not json_path.exists():
            missing.append(str(json_path))
            continue
        with json_path.open() as f:
            data = json.load(f)
        for task in data.get("tasks", []):
            tid = int(task["task_id"])
            desc = task.get("task_description", "")
            successes = int(task["successes"])
            total = int(task["total_episodes"])
            rate = float(task["success_rate"])
            suite_successes += successes
            suite_total += total
            per_task_rows.append((suite, tid, desc, successes, total, rate))
    all_suite_results[suite] = {
        "successes": suite_successes,
        "total": suite_total,
        "success_rate": suite_successes / suite_total if suite_total else 0.0,
    }

print("\n" + "=" * 90)
print("  LIBERO MULTI-GPU EVALUATION RESULTS - PER-TASK BREAKDOWN")
print("=" * 90)
print(f"{'Suite':<18} {'TaskID':>6}  {'Success':>7}  {'Total':>5}  {'Rate':>7}  Description")
print("-" * 90)

current_suite = None
for suite, tid, desc, successes, total, rate in sorted(
    per_task_rows, key=lambda row: (suite_order.get(row[0], 999), row[1])
):
    if suite != current_suite:
        if current_suite is not None:
            sr = all_suite_results[current_suite]
            print(f"  {'-' * 38}  Suite Avg: {sr['success_rate']:6.1%}  ({sr['successes']}/{sr['total']})")
            print()
        current_suite = suite
    desc_short = desc[:35] + "..." if len(desc) > 38 else desc
    print(f"{suite:<18} {tid:>6}  {successes:>5}/{total:<3d}  {total:>5}  {rate:>6.1%}  {desc_short}")

if current_suite is not None:
    sr = all_suite_results[current_suite]
    print(f"  {'-' * 38}  Suite Avg: {sr['success_rate']:6.1%}  ({sr['successes']}/{sr['total']})")

print("\n" + "=" * 60)
print("  SUITE SUMMARY")
print("=" * 60)
print(f"{'Suite':<18} {'Success':>8}  {'Total':>6}  {'Rate':>8}")
print("-" * 60)

grand_successes = 0
grand_total = 0
for suite in suite_names:
    sr = all_suite_results[suite]
    print(f"{suite:<18} {sr['successes']:>5}/{sr['total']:<3d}  {sr['total']:>6}  {sr['success_rate']:>7.1%}")
    grand_successes += sr["successes"]
    grand_total += sr["total"]

grand_rate = grand_successes / grand_total if grand_total else 0.0
print("-" * 60)
print(f"{'OVERALL':<18} {grand_successes:>5}/{grand_total:<3d}  {grand_total:>6}  {grand_rate:>7.1%}")
print("=" * 60)

if missing:
    print("\nWARNING: missing shard result files:")
    for path in missing:
        print(f"  {path}")

failed_shards_path = output_dir / "failed_shards.tsv"
failed_shards = []
if failed_shards_path.exists():
    failed_shards = failed_shards_path.read_text().splitlines()

summary = {
    "ckpt_path": ckpt_path,
    "num_gpus": num_gpus,
    "clients_per_gpu": clients_per_gpu,
    "num_parallel": num_parallel,
    "num_trials_per_task": num_trials,
    "suites": all_suite_results,
    "grand_successes": grand_successes,
    "grand_total": grand_total,
    "grand_success_rate": grand_rate,
    "missing_shards": missing,
    "failed_shards": failed_shards,
    "per_task": [
        {
            "suite": suite,
            "task_id": task_id,
            "description": desc,
            "successes": successes,
            "total": total,
            "success_rate": rate,
        }
        for suite, task_id, desc, successes, total, rate in per_task_rows
    ],
}

summary_path = output_dir / "summary.json"
with summary_path.open("w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nSummary saved to {summary_path}")
PYTHON_SCRIPT

echo ""
echo "All done!"
if [[ ${#FAILED_WORKERS[@]} -gt 0 ]]; then
    echo "WARNING: failed workers: ${FAILED_WORKERS[*]}" >&2
    exit 1
fi
```
