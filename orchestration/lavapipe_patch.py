"""Force SAPIEN onto Mesa lavapipe (software Vulkan), bypassing the NVIDIA
graphics driver entirely.

Why: this host accumulated 412 general-protection faults in
``__kmem_cache_alloc_node``, all reached through NVIDIA driver allocations, and
crashed three times. The box ran 81 days clean on a compute-only NVIDIA stack;
the faults began roughly a day after user-space GL/EGL/Vulkan was installed and
SAPIEN started exercising the driver's graphics path. Software Vulkan avoids
that path while leaving JAX/CUDA compute (which was stable) untouched.

Ported from allenai/vla-evaluation-harness `benchmarks/robomme/benchmark.py`,
which declares RoboMME as supporting `render_backends = {"gpu", "cpu"}`.

Must be imported BEFORE `sapien.render`: the Vulkan ICD binds on first import.
"""

from __future__ import annotations

import os
import sys

ICD = os.environ.get(
    "ROBOMME_LAVAPIPE_ICD", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
)


_APPLIED: dict[str, str] | None = None


def engage() -> dict[str, str]:
    global _APPLIED
    if _APPLIED is not None:          # idempotent: the module may auto-engage
        return _APPLIED
    if "sapien.render" in sys.modules:
        raise RuntimeError(
            "sapien.render already imported; VK_ICD_FILENAMES only binds on first "
            "Vulkan init. Import this module earlier."
        )
    if not os.path.isfile(ICD):
        raise RuntimeError(f"lavapipe ICD not found: {ICD}")

    # 1) point Vulkan dispatch at lavapipe. LP_NUM_THREADS=4 with single-threaded
    #    BLAS is the harness's measured sweet spot for 256x256.
    applied = {
        "VK_ICD_FILENAMES": ICD,
        "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS", "4"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
    }
    os.environ.update(applied)

    # 2) RenderSystem("cuda:0") raises "Failed to find a supported physical
    #    device" under lavapipe, so drop the device argument.
    import sapien.render as sr

    _orig = sr.RenderSystem
    sr.RenderSystem = lambda *a, **k: _orig()

    # 3) ManiSkill resolves a "sapien_cuda" render backend; rewrite to sapien_cpu.
    #    Patch both the source module and the already-imported sapien_env, which
    #    captured the function by value at its own import time.
    from mani_skill.envs.utils.system import backend as _backend_mod

    _orig_parse = _backend_mod.parse_sim_and_render_backend

    def _patched_parse(sim_backend, render_backend):
        result = _orig_parse(sim_backend, render_backend)
        if result.render_backend == "sapien_cuda":
            result.render_backend = "sapien_cpu"
        return result

    _backend_mod.parse_sim_and_render_backend = _patched_parse
    import mani_skill.envs.sapien_env as _se

    _se.parse_sim_and_render_backend = _patched_parse

    _APPLIED = applied
    return applied


if os.environ.get("ROBOMME_USE_LAVAPIPE", "").lower() in ("1", "true", "yes"):
    print(f"[lavapipe] engaged: {engage()}", flush=True)
