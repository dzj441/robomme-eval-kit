"""Find the real GPU-memory floor for an MME-VLA policy server.

The earlier floor (MEM_FRACTION=0.14) was measured by loading the model and
checking the port opened. That missed the actual peak: VideoPlaceButton and
VideoPlaceOrder push 700+ conditioning frames through `add_buffer` in one go, so
they allocate ~1.6 GB beyond the resident model. Twenty-two servers died on
exactly that task, mid-run.

So probe against a long-video task, not against startup.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = "/home/qid/dzj/RoboMME_policy"
ROOT = "/datadrive1/dzj/RoboMME"
CKPT = (f"{ROOT}/ckpt/perceptual-framesamp-modul/home/daiyp/MME-VLA-Suite/runs/"
        "ckpts/mme_vla_suite/perceptual-framesamp-modul/79999")
SERVER_PY = "/home/qid/dzj/miniconda3/envs/robomme-vla/bin/python"
EVAL_PY = "/home/qid/dzj/miniconda3/envs/robomme/bin/python"
SP = "/home/qid/dzj/miniconda3/envs/robomme-vla/lib/python3.11/site-packages"

LDP = subprocess.run(
    ["bash", "-c", f"find {SP}/nvidia -maxdepth 2 -name lib -type d | tr '\\n' ':'"],
    capture_output=True, text=True).stdout

# one long-video task (the one that OOMed) plus a cheap one as control
TASK = os.environ.get("PROBE_TASK", "VideoPlaceButton")
PORT = 8777


def peak_mem() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return max((int(x) for x in out.split() if x.isdigit()), default=0)


def try_frac(frac: str) -> tuple[bool, int, str]:
    env = dict(os.environ, LD_LIBRARY_PATH=LDP, CUDA_VISIBLE_DEVICES="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=frac)
    print(f"  [{frac}] starting server...", flush=True)
    t_start = time.time()
    srv = subprocess.Popen(
        [SERVER_PY, f"{REPO}/scripts/serve_policy.py", "--seed=7", f"--port={PORT}",
         "policy:checkpoint", f"--policy.dir={CKPT}", "--policy.config=mme_vla_suite"],
        cwd=REPO, env=env, stdout=open(f"/tmp/mf_{frac}_srv.log", "w"),
        stderr=subprocess.STDOUT)
    try:
        for i in range(120):
            time.sleep(5)
            if "server listening" in open(f"/tmp/mf_{frac}_srv.log").read():
                print(f"  [{frac}] server up in {time.time()-t_start:.0f}s "
                      f"({peak_mem()} MiB resident), running {TASK}...", flush=True)
                break
            if i % 6 == 5:
                print(f"  [{frac}] ...waiting for server ({(i+1)*5}s)", flush=True)
            if srv.poll() is not None:
                return False, 0, "server died at startup"
        else:
            return False, 0, "server never listened"

        # drive one real episode of the long-video task through this server
        out = f"/tmp/mf_{frac}_out"
        subprocess.run(["rm", "-rf", out])
        ev = subprocess.run(
            [EVAL_PY, f"{ROOT}/eval_lavapipe.py",
             "--args.host=127.0.0.1", f"--args.port={PORT}", "--args.model_seed=7",
             "--args.policy_name=probe", "--args.model_ckpt_id=79999",
             f"--args.only_tasks={TASK}", f"--args.save_dir={out}"],
            cwd=f"{REPO}/examples/robomme",
            env=dict(os.environ, ROBOMME_USE_LAVAPIPE="1", LP_NUM_THREADS="4",
                     CUDA_VISIBLE_DEVICES="0",
                     PYTHONPATH=f"{REPO}/examples/robomme"),
            capture_output=True, text=True, timeout=1500)
        mem = peak_mem()
        print(f"  [{frac}] episode phase done in {time.time()-t_start:.0f}s", flush=True)
        srv_log = open(f"/tmp/mf_{frac}_srv.log").read()
        if "RESOURCE_EXHAUSTED" in srv_log:
            return False, mem, "server OOM"
        n = ev.stdout.count("setup finished")
        return n > 0, mem, f"{n} episodes ran"
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        time.sleep(8)


if __name__ == "__main__":
    print(f"probing against task={TASK}\n")
    for frac in sys.argv[1:] or ["0.14", "0.20", "0.25"]:
        ok, mem, note = try_frac(frac)
        print(f"  MEM_FRACTION={frac:<5} {'OK ' if ok else 'FAIL'}  peak={mem:>6} MiB  {note}",
              flush=True)
