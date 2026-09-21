"""Controlled model benchmarks on the discovery cohort (GSE42568 only). NO explainers, NO stability here (Phase 4+).

    python scripts/benchmark_models.py pilot                  # ONE fold per model: sanity checks + runtime
    python scripts/benchmark_models.py benchmark              # 1 repeat x 5 folds per model: performance + runtime + model support

Checks per model (pilot): training succeeds, scores finite (probabilities in [0, 1] for LR/RF/XGBoost), predicted labels are
{0, 1}, feature dimensions consistent, the fitted model's support size is sensible, runtime. The benchmark writes
results/benchmarks/<id>/ with per-fold rows, pooled metrics + bootstrap CIs and a manifest (git commit, config, versions).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from bioxplain.data.discovery import load_discovery
from bioxplain.evaluation.bootstrap import bootstrap_intervals
from bioxplain.evaluation.metrics import classification_metrics
from bioxplain.models.factory import IMPLEMENTED, PROBABILITY_MODELS, resolve_spec
from bioxplain.utils.provenance import config_hash, git_state, software_versions, utc_now
from bioxplain.validation.cv import run_fold
from bioxplain.validation.splits import repeated_stratified_splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260921
PREP = {"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean"}     # scale decided per model (D8/D11b)


def pilot(data) -> None:
    y = data.y.to_numpy()
    sp = next(iter(repeated_stratified_splits(y, 5, 1, SEED)))
    print(f"single controlled fold: train {len(sp.train_idx)} (normals {(y[sp.train_idx] == 0).sum()}), test {len(sp.test_idx)} (normals {(y[sp.test_idx] == 0).sum()})")
    for name in IMPLEMENTED:
        t = time.perf_counter()
        out = run_fold(data.X, y, sp.train_idx, sp.test_idx, model_spec={"name": name}, explainer_name=None,
                       probe_to_gene=data.probe_to_gene, prep_cfg=PREP, seed=SEED)
        wall = time.perf_counter() - t
        pr = out.predictions
        checks = {
            "scores_finite": bool(np.isfinite(pr.score).all()),
            "probabilities_in_0_1": bool(pr.score.between(0, 1).all()) if name in PROBABILITY_MODELS else "n/a (margin)",
            "labels_binary": bool(set(pr.pred.unique()) <= {0, 1}),
            "n_predictions_equals_n_test": len(pr) == len(sp.test_idx),
            "support_within_genes": out.diagnostics["n_features_used"] <= out.diagnostics["n_genes"],
        }
        d = out.diagnostics
        print(f"\n[{name}] resolved spec: {resolve_spec({'name': name})}")
        print(f"  genes {d['n_genes']}, features used by model {d['n_features_used']} ({d['n_features_used'] / d['n_genes']:.1%}) | "
              f"prep {d['prep_seconds']:.2f}s fit {d['fit_seconds']:.2f}s total {wall:.2f}s")
        print(f"  checks: {checks}")
        print(f"  fold metrics: ROC-AUC {out.metrics['roc_auc']:.3f}, AP(normal) {out.metrics['ap_normal']:.3f}, balanced acc {out.metrics['balanced_accuracy']:.3f}, "
              f"confusion tn/fp/fn/tp = {out.metrics['tn']}/{out.metrics['fp']}/{out.metrics['fn']}/{out.metrics['tp']}")
        assert all(v is True or v == "n/a (margin)" for v in checks.values()), checks


def benchmark(data) -> None:
    y = data.y.to_numpy()
    started = utc_now()
    cfg = {"seed": SEED, "cv": {"n_splits": 5, "n_repeats": 1}, "preprocessing": PREP, "models": {n: resolve_spec({"name": n}) for n in IMPLEMENTED}}
    out = ROOT / "results/benchmarks" / f"{started:%Y%m%dT%H%M%SZ}_models_1x5_{config_hash(cfg)[:8]}"
    out.mkdir(parents=True)
    rows, preds = [], []
    for name in IMPLEMENTED:
        for sp in repeated_stratified_splits(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], SEED):
            res = run_fold(data.X, y, sp.train_idx, sp.test_idx, model_spec={"name": name}, explainer_name=None,
                           probe_to_gene=data.probe_to_gene, prep_cfg=PREP, seed=SEED)
            rows.append({"model": name, "repeat": sp.repeat, "fold": sp.fold, "seed": sp.seed, **res.metrics, **res.diagnostics})
            preds.append(res.predictions.assign(model=name, repeat=sp.repeat, fold=sp.fold))
    runs = pd.DataFrame(rows).drop(columns=["model_name"])
    preds = pd.concat(preds, ignore_index=True)
    pooled = []
    for name, g in preds.groupby("model", sort=False):
        yy, ss, pp = g.y_true.to_numpy(), g.score.to_numpy(), g.pred.to_numpy()
        ci = bootstrap_intervals(yy, ss, pp, 2000, SEED)
        m = classification_metrics(yy, ss, pp)
        r = runs[runs.model == name]
        pooled.append({"model": name, "n": len(g), **{k: m[k] for k in ["roc_auc", "ap_normal", "balanced_accuracy", "sensitivity", "specificity", "f1_cancer", "f1_normal", "tn", "fp", "fn", "tp"]},
                       "roc_auc_ci": f"{ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}", "ap_normal_ci": f"{ci['ap_normal'][0]:.3f}-{ci['ap_normal'][1]:.3f}",
                       "balanced_accuracy_ci": f"{ci['balanced_accuracy'][0]:.3f}-{ci['balanced_accuracy'][1]:.3f}",
                       "features_used_mean": r.n_features_used.mean(), "features_used_min": int(r.n_features_used.min()), "features_used_max": int(r.n_features_used.max()),
                       "n_genes_mean": r.n_genes.mean(), "fit_s_mean": r.fit_seconds.mean(), "prep_s_mean": r.prep_seconds.mean()})
    pooled = pd.DataFrame(pooled)
    runs.to_csv(out / "per_fold.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    pooled.to_csv(out / "pooled_metrics.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"id": out.name, "started_utc": started.isoformat(), "config": cfg, "config_sha256": config_hash(cfg),
                                                    "git": git_state(ROOT), "software": software_versions(), "n_samples": int(len(y)), "n_normal": int((y == 0).sum())}, indent=2, default=str))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"results written to {out}\n")
    print(pooled[["model", "roc_auc", "roc_auc_ci", "ap_normal", "ap_normal_ci", "balanced_accuracy", "sensitivity", "specificity", "f1_normal", "tn", "fp", "fn", "tp"]].round(3).to_string(index=False))
    print("\n" + pooled[["model", "features_used_mean", "features_used_min", "features_used_max", "n_genes_mean", "prep_s_mean", "fit_s_mean"]].round(2).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pilot", "benchmark"):
        sys.exit(__doc__)
    d = load_discovery(ROOT)
    (pilot if sys.argv[1] == "pilot" else benchmark)(d)
