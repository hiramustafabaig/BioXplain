"""Run a discovery experiment from a config:  python scripts/run_experiment.py configs/slice_logreg_coef.yaml"""
from __future__ import annotations

import pathlib
import sys

from bioxplain.experiment import run_experiment

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: run_experiment.py <config.yaml>")
    out = run_experiment(sys.argv[1], ROOT)
    print(f"results written to {out}")
