"""python scripts/run_matrix.py <config.yaml> [--resume <run_dir>]   then analysed with bioxplain.analysis.analyze_run"""
from __future__ import annotations

import pathlib
import sys

from bioxplain.analysis import analyze_run
from bioxplain.matrix import run_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if len(sys.argv) not in (2, 4):
        sys.exit(__doc__)
    resume = sys.argv[3] if len(sys.argv) == 4 else None
    out = run_matrix(sys.argv[1], ROOT, resume_dir=resume)
    analyze_run(out)
    print(f"results written to {out}")
