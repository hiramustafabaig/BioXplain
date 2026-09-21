"""python scripts/run_regularization_sensitivity.py configs/sensitivity_logreg_C.yaml   (discovery cohort only)"""
from __future__ import annotations

import pathlib
import sys

from bioxplain.sensitivity import run_regularization_sensitivity

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: run_regularization_sensitivity.py <config.yaml>")
    print(f"results written to {run_regularization_sensitivity(sys.argv[1], ROOT)}")
