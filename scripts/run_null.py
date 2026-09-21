"""python scripts/run_null.py <base_config.yaml> <out_dir> <start> <stop>   (resumable; discovery cohort only)"""
from __future__ import annotations

import pathlib
import sys

from bioxplain.null import run_null_replicates

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    dirs = run_null_replicates(sys.argv[1], ROOT, sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    print(f"{len(dirs)} replicate directories under {sys.argv[2]}")
