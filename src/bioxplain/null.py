"""Permutation-label null (Addendum A2): B replicates, each one 5-fold CV repeat on globally permuted labels, full pipeline.

Observed and null statistics come from the same function (`repeat_level_stability`), so they are matched in resampling number.
No p-value is produced: with B replicates the smallest attainable exceedance is 1/(B+1); results are descriptive.
"""
from __future__ import annotations

import copy
import pathlib

import pandas as pd
import yaml

from bioxplain.matrix import load_matrix_config, run_matrix
from bioxplain.stability.framework import repeat_level_stability


def replicate_config(base_cfg: dict, b: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.update({"labels": "permuted", "null": {"replicate": b}, "name": f"null_rep{b:03d}"})
    cfg["cv"] = {**cfg["cv"], "n_repeats": 1}
    return cfg


def run_null_replicates(base_config_path, root, out_dir, start: int, stop: int) -> list[pathlib.Path]:
    """Run (or resume) replicates start..stop-1, each into <out_dir>/rep_XXX; finished replicates are skipped."""
    base = load_matrix_config(base_config_path)
    out_dir = pathlib.Path(out_dir)
    dirs = []
    for b in range(start, stop):
        rd = out_dir / f"rep_{b:03d}"
        rd.mkdir(parents=True, exist_ok=True)
        cfg = replicate_config({k: v for k, v in base.items() if not k.startswith("_")}, b)
        cfg_path = rd / "replicate_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        if not (rd / "manifest.json").exists():
            run_matrix(cfg_path, root, resume_dir=rd)
        dirs.append(rd)
    return dirs


def null_repeat_level(rep_dirs, ks, d: int) -> pd.DataFrame:
    frames = []
    for rd in rep_dirs:
        rk = pd.read_csv(rd / "rankings.csv.gz")
        pipes = pd.read_csv(rd / "pipelines.csv")
        b = int(rd.name.split("_")[1])
        frames.append(repeat_level_stability(rk, pipes, ks, d).assign(repeat=b))
    return pd.concat(frames, ignore_index=True)


def compare_observed_null(observed: pd.DataFrame, null: pd.DataFrame, metric: str = "phi") -> pd.DataFrame:
    """Per dimension / group / k: observed vs null distributions of the repeat-level statistic (descriptive)."""
    rows = []
    for (dim, grp, k), o in observed.groupby(["dimension", "group", "k"], sort=False):
        n = null[(null.dimension == dim) & (null.group == grp) & (null.k == k)][metric].dropna()
        ov = o[metric].dropna()
        if len(n) == 0 or len(ov) == 0:
            continue
        hi = n.quantile(0.975)
        rows.append({"dimension": dim, "group": grp, "k": k, "n_observed_repeats": len(ov), "n_null_replicates": len(n),
                     "obs_mean": ov.mean(), "obs_median": ov.median(), "obs_min": ov.min(), "obs_max": ov.max(),
                     "null_mean": n.mean(), "null_median": n.median(), "null_p2.5": n.quantile(0.025), "null_p97.5": hi, "null_max": n.max(),
                     "median_difference": ov.median() - n.median(), "frac_obs_repeats_above_null_p97.5": float((ov > hi).mean()),
                     "obs_min_above_null_max": bool(ov.min() > n.max())})
    return pd.DataFrame(rows)
