"""Run RoboMME's eval.py with SAPIEN pinned to lavapipe software Vulkan.

The patch has to land before `sapien.render` is imported, so this wrapper
engages it first and only then hands over to the stock eval entrypoint —
eval.py itself is untouched, and tyro still sees the real argv.
"""

from __future__ import annotations

import runpy
import sys

sys.path.insert(0, "/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME")
import lavapipe_patch  # noqa: E402

print(f"[lavapipe] {lavapipe_patch.engage()}", flush=True)

EVAL = "/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME_policy/examples/robomme/eval.py"
sys.path.insert(0, "/inspire/hdd/global_user/lutianyi-253108120107/tylu/projects/dzj/RoboMME_policy/examples/robomme")
runpy.run_path(EVAL, run_name="__main__")
