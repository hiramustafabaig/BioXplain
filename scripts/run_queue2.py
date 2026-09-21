"""Pre-freeze preprocessing sensitivity runs (decisions D6 and D7): highest-variance probe rule, filter thresholds 0.25x and 1.0x."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

for name in ("sens_collapse_var", "sens_filter_025", "sens_filter_100"):
    t = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", "scripts/run_matrix.py", f"configs/{name}.yaml"], cwd=ROOT)
    print(f"[queue2] {name}: exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
print("[queue2] done", flush=True)
