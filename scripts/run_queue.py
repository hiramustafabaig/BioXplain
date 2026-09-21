"""Sequential queue for the remaining discovery-side experiments (pre-specified in Addendum A2 / A6 / A7).

Order: permutation-label null (replicates 0..29), Dec-2004 subset, size-matched control, composition ablation.
Failures are logged and the queue continues; every step is resumable.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def step(name, *args):
    t = time.time()
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT)
    print(f"[queue] {name}: exit {r.returncode} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    step("null replicates 0-29", "scripts/run_null.py", "configs/null_matrix.yaml", "results/null/main", "0", "30")
    step("dec2004 subset", "scripts/run_matrix.py", "configs/date_dec2004.yaml")
    step("size-matched control", "scripts/run_matrix.py", "configs/date_control.yaml")
    step("composition ablation", "scripts/run_matrix.py", "configs/composition_ablation.yaml")
    print("[queue] done", flush=True)
