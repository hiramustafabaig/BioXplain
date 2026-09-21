"""python scripts/run_external.py

Only works after the discovery freeze is committed (gated by bioxplain.external.verify_freeze).
"""
from __future__ import annotations

import pathlib

from bioxplain.external_analysis import run_external_analysis

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(f"results written to {run_external_analysis(ROOT)}")
