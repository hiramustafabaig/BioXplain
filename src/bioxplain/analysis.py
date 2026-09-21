"""Turn a matrix run directory into stability tables, gene tables and frozen-list candidates (writes <run>/analysis/)."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from bioxplain.consensus import frozen_lists, gene_table
from bioxplain.stability.framework import pairwise_jaccard_matrix, pooled_resampling, repeat_level_stability, summarize_repeats


def analyze_run(run_dir, ks=(10, 25, 50)) -> pathlib.Path:
    run = pathlib.Path(run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    rk = pd.read_csv(run / "rankings.csv.gz")
    pipes = pd.read_csv(run / "pipelines.csv")
    d = json.loads((run / "manifest.json").read_text())["feature_universe"]["n_genes"]
    rl = repeat_level_stability(rk, pipes, list(ks), d)
    rl.to_csv(out / "repeat_level_stability.csv", index=False)
    summarize_repeats(rl).to_csv(out / "stability_summary.csv", index=False)
    pooled_resampling(rk, pipes, list(ks), d).to_csv(out / "pooled_resampling.csv", index=False)
    for k in ks:
        gene_table(rk, pipes, k).to_csv(out / f"gene_table_k{k}.csv", index=False)
    pairwise_jaccard_matrix(rk, pipes, 25).to_csv(out / "pairwise_jaccard_k25.csv")
    (out / "frozen_list_candidates.json").write_text(json.dumps(frozen_lists(rk, pipes), indent=2))
    if (run / "fold_metrics.csv").exists():
        m = pd.read_csv(run / "fold_metrics.csv")
        m.groupby("config_id", sort=False)[["roc_auc", "ap_normal", "balanced_accuracy", "sensitivity", "specificity", "n_features_used"]].agg(["mean", "std"]).to_csv(out / "fold_metric_summary.csv")
    return out
