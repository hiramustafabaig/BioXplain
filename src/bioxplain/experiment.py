"""Config-driven discovery experiment: data -> repeated stratified CV -> rankings -> stability -> artefacts.

Every output row carries dataset / model / explainer / repeat / fold / seed; the run directory holds a
manifest linking results to config hash, code commit, data hashes, software versions and runtimes.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import yaml

from bioxplain.data.discovery import load_discovery
from bioxplain.evaluation.bootstrap import bootstrap_intervals
from bioxplain.evaluation.metrics import classification_metrics
from bioxplain.models.factory import resolve_spec
from bioxplain.stability.summary import stability_summary
from bioxplain.utils.provenance import config_hash, git_state, software_versions, utc_now
from bioxplain.validation.cv import run_fold
from bioxplain.validation.splits import repeated_stratified_splits

REQUIRED = ("name", "dataset", "seed", "cv", "preprocessing", "model", "explainer", "top_k", "store_top")


def load_config(path: str | pathlib.Path) -> dict:
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config {path} is missing keys: {missing}")
    if cfg["dataset"] != "GSE42568":
        raise ValueError("discovery experiments may only use GSE42568; the external cohort is opened only after the freeze")
    if max(cfg["top_k"]) > cfg["store_top"]:
        raise ValueError("store_top must be >= max(top_k)")
    return cfg


def _pooled_metrics(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for rep, g in pred.groupby("repeat"):
        y, s, p = g["y_true"].to_numpy(), g["score"].to_numpy(), g["pred"].to_numpy()
        assert g["sample"].is_unique, "each sample must be predicted exactly once per repeat"
        row = {"repeat": rep, "n": len(g), **classification_metrics(y, s, p)}
        for name, (lo, hi) in bootstrap_intervals(y, s, p, n_boot, seed).items():   # vectorised, see evaluation/bootstrap.py
            row[f"{name}_ci_lo"], row[f"{name}_ci_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(config_path: str | pathlib.Path, root: str | pathlib.Path, out_root: str | pathlib.Path | None = None) -> pathlib.Path:
    root = pathlib.Path(root)
    cfg = load_config(config_path)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    started = utc_now()
    exp_id = f"{started:%Y%m%dT%H%M%SZ}_{cfg['name']}_{config_hash(cfg)[:8]}"
    out = pathlib.Path(out_root or root / "results/experiments") / exp_id
    out.mkdir(parents=True, exist_ok=False)

    t = time.perf_counter()
    data = load_discovery(root)
    timings["load_data_s"] = time.perf_counter() - t
    y = data.y.to_numpy()
    d_universe = int(data.probe_to_gene.nunique())

    model_spec, explainer = resolve_spec(cfg["model"]), cfg["explainer"]
    ranking_frames, pred_frames, run_rows, eligible = [], [], [], set()
    t = time.perf_counter()
    for sp in repeated_stratified_splits(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"]):
        tf = time.perf_counter()
        res = run_fold(data.X, y, sp.train_idx, sp.test_idx, model_spec=model_spec, explainer_name=explainer,
                       probe_to_gene=data.probe_to_gene, prep_cfg=cfg["preprocessing"], seed=cfg["seed"],
                       store_top=cfg["store_top"])
        key = {"experiment_id": exp_id, "dataset": cfg["dataset"], "model": model_spec["name"],
               "explainer": explainer, "repeat": sp.repeat, "fold": sp.fold, "seed": sp.seed}
        ranking_frames.append(res.ranking.assign(**key))
        pred_frames.append(res.predictions.assign(**key))
        run_rows.append({**key, **res.metrics, **res.diagnostics, "fold_seconds": time.perf_counter() - tf})
        eligible |= set(res.eligible_genes)
    timings["cv_loop_s"] = time.perf_counter() - t

    rankings = pd.concat(ranking_frames, ignore_index=True)
    preds = pd.concat(pred_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    t = time.perf_counter()
    pooled = _pooled_metrics(preds, cfg.get("bootstrap", {}).get("n_boot", 2000), cfg["seed"])
    stab = stability_summary(rankings, cfg["top_k"], d_universe, n_features_alt=len(eligible),
                             extra={"experiment_id": exp_id, "model": model_spec["name"], "explainer": explainer})
    timings["metrics_and_stability_s"] = time.perf_counter() - t

    rankings.to_csv(out / "rankings_top.csv.gz", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    pooled.to_csv(out / "pooled_metrics.csv", index=False)
    stab.to_csv(out / "stability.csv", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    timings["total_s"] = time.perf_counter() - t0
    manifest = {
        "experiment_id": exp_id, "started_utc": started.isoformat(), "config_sha256": config_hash(cfg),
        "config_path": str(pathlib.Path(config_path)), "git": git_state(root), "software": software_versions(),
        "data": json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE42568"],
        "feature_universe": {"n_probes": int(len(data.probe_to_gene)), "n_genes": d_universe,
                             "n_genes_eligible_in_any_fold": len(eligible)},
        "model_resolved": model_spec, "explainer": explainer,
        "n_samples": int(len(y)), "n_cancer": int(y.sum()), "n_normal": int((y == 0).sum()),
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "outputs": sorted(p.name for p in out.iterdir()) + ["manifest.json"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
