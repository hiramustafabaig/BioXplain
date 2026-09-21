"""Benchmark + equivalence check of the vectorised bootstrap against the reference implementation.

    python scripts/benchmark_metrics.py results/experiments/<run_dir>

Uses the predictions of an existing run. Reports timings, the maximum absolute difference between the reference
(scikit-learn based, ``evaluation.metrics.bootstrap_ci``) and the vectorised CIs, the difference to the CIs
stored in the run's ``pooled_metrics.csv``, and repeats the fast path under several PYTHONHASHSEEDs.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from bioxplain.evaluation.bootstrap import bootstrap_intervals
from bioxplain.evaluation.metrics import bootstrap_ci

TOLERANCE = 1e-12          # documented equivalence tolerance (absolute, on CI endpoints in [0, 1])


def reference_intervals(y, s, n_boot, seed):
    return {
        "roc_auc": bootstrap_ci(y, s, roc_auc_score, n_boot, seed),
        "ap_normal": bootstrap_ci(y, s, lambda a, b: average_precision_score(1 - a, 1 - b), n_boot, seed),
        "balanced_accuracy": bootstrap_ci(y, s, lambda a, b: balanced_accuracy_score(a, (b >= 0.5).astype(int)), n_boot, seed),
    }


def main(run_dir: str) -> None:
    run = pathlib.Path(run_dir)
    cfg = json.loads(json.dumps(__import__("yaml").safe_load((run / "config.yaml").read_text())))
    n_boot, seed = cfg.get("bootstrap", {}).get("n_boot", 2000), cfg["seed"]
    pred = pd.read_csv(run / "predictions.csv")
    stored = pd.read_csv(run / "pooled_metrics.csv").set_index("repeat")
    t_ref = t_new = 0.0
    worst_ref = worst_stored = 0.0
    for rep, g in pred.groupby("repeat"):
        y, s, p = g["y_true"].to_numpy(), g["score"].to_numpy(), g["pred"].to_numpy()
        t = time.perf_counter(); ref = reference_intervals(y, s, n_boot, seed); t_ref += time.perf_counter() - t
        t = time.perf_counter(); new = bootstrap_intervals(y, s, p, n_boot, seed); t_new += time.perf_counter() - t
        for k in ref:
            worst_ref = max(worst_ref, *(abs(a - b) for a, b in zip(ref[k], new[k])))
            worst_stored = max(worst_stored, abs(stored.loc[rep, f"{k}_ci_lo"] - new[k][0]), abs(stored.loc[rep, f"{k}_ci_hi"] - new[k][1]))
    print(f"reference (sklearn, per-resample):  {t_ref:7.2f} s   ({pred.repeat.nunique()} repeats x 3 metrics x {n_boot} resamples)")
    print(f"vectorised (resample weights):      {t_new:7.3f} s   speed-up x{t_ref / t_new:,.0f}")
    print(f"max |reference - vectorised| CI endpoint:      {worst_ref:.3e}")
    print(f"max |stored pilot CI - vectorised| CI endpoint: {worst_stored:.3e}")
    assert worst_ref <= TOLERANCE and worst_stored <= TOLERANCE, "equivalence tolerance exceeded"
    print(f"EQUIVALENT within tolerance {TOLERANCE:g}")

    code = ("import sys,pandas as pd\nfrom bioxplain.evaluation.bootstrap import bootstrap_intervals\n"
            "g=pd.read_csv(sys.argv[1]); g=g[g.repeat==0]\n"
            "print(repr(bootstrap_intervals(g.y_true.to_numpy(),g.score.to_numpy(),g.pred.to_numpy(),%d,%d)))" % (n_boot, seed))
    outs = {subprocess.run([sys.executable, "-c", code, str(run / "predictions.csv")], env={**os.environ, "PYTHONHASHSEED": str(h)},
                           capture_output=True, text=True, check=True).stdout for h in range(4)}
    print(f"PYTHONHASHSEED 0..3 -> {len(outs)} distinct output(s)")
    assert len(outs) == 1


if __name__ == "__main__":
    main(sys.argv[1])
