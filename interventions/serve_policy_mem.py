"""Run the stock policy server with a memory intervention engaged.

Same shape as eval_lavapipe.py: patch first, then hand the real script its own
__main__ so its argument parsing and checkpoint loading are untouched.
"""

from __future__ import annotations

import runpy
import sys

sys.path.insert(0, "/datadrive1/dzj/RoboMME")

import mem_intervention

mem_intervention.apply()

SERVE = "/home/qid/dzj/RoboMME_policy/scripts/serve_policy.py"
sys.argv[0] = SERVE
runpy.run_path(SERVE, run_name="__main__")
