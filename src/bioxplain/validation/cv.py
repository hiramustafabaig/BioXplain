"""Run ONE cross-validation fold in a leakage-safe way.

Contract: everything data-dependent (expression filter, probe collapse, scaling, model fit, explanation)
is computed from ``X.iloc[train_idx]`` / ``y[train_idx]`` only. ``X.iloc[test_idx]`` is used solely to
produce predictions for evaluation. tests/test_leakage.py enforces this contract (fold-perturbation
invariance, fit-index spying, and a deliberately leaky variant that the tests must catch).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bioxplain.evaluation.metrics import classification_metrics
from bioxplain.explainers.attribution import EXPLAINERS
from bioxplain.models.factory import SCALED, make_model, model_scores, model_support, resolve_spec
from bioxplain.preprocessing.transformers import FoldPreprocessor


@dataclass
class FoldOutput:
    ranking: pd.DataFrame       # top `store_top` genes: gene, probe, importance, signed, rank, n_detected_train
    predictions: pd.DataFrame   # test samples: sample, y_true, score, pred
    metrics: dict
    diagnostics: dict
    eligible_genes: pd.Index    # genes that could have been ranked in this fold
    support_genes: list         # genes the fitted model actually uses (non-zero coefficient / split feature)


def run_fold(
    X: pd.DataFrame,
    y,
    train_idx,
    test_idx,
    *,
    model_spec: dict,
    explainer_name: str | None,
    probe_to_gene: pd.Series,
    prep_cfg: dict,
    seed: int,
    store_top: int | None = 200,
) -> FoldOutput:
    y = np.asarray(y).astype(int)
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train and test indices overlap")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    spec = resolve_spec(model_spec)
    name = spec["name"]
    prep = FoldPreprocessor(
        probe_to_gene,
        min_detect_frac_of_minority=prep_cfg.get("min_detect_frac_of_minority", 0.5),
        collapse_rule=prep_cfg.get("collapse_rule", "max_mean"),
        scale=prep_cfg.get("scale", SCALED[name]),           # default per model (D8/D11b); an explicit config value wins
    )
    t0 = time.perf_counter()
    Z_train = prep.fit_transform(X_train, y_train)              # <- learned on training samples only
    t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = make_model(spec, seed, y_train).fit(Z_train.to_numpy(), y_train)
    t_fit = time.perf_counter() - t0
    genes = list(Z_train.columns)
    support = model_support(model, name, len(genes))
    if explainer_name is None:                                   # explainers for RF/XGBoost arrive in Phase 4
        attribution = pd.DataFrame({"gene": [], "importance": [], "signed": [], "rank": []})
    else:
        attribution = EXPLAINERS[explainer_name](model, genes)   # <- from the training-fit model

    Z_test = prep.transform(X.iloc[test_idx])                    # <- validation fold: transform only
    score = model_scores(model, name, Z_test.to_numpy())
    pred = model.predict(Z_test.to_numpy())
    metrics = classification_metrics(y[test_idx], score, pred)

    # D13b: only genes with strictly positive attribution are eligible for top-k sets (sparse models have exact zeros;
    # ranking those by gene name would manufacture identical, arbitrary 'selected' genes in every fold)
    positive = attribution[attribution["importance"] > 0]
    ranking = (positive if store_top is None else positive.head(store_top)).copy()
    if len(ranking):
        ranking["probe"] = prep.collapser_.selected_.loc[ranking["gene"]].to_numpy()
        if prep.filter_.floor_ is not None:   # how many TRAINING samples detect the chosen probe (artefact check, D8)
            raw = X_train.loc[:, ranking["probe"].to_numpy()].to_numpy()
            ranking["n_detected_train"] = (raw > prep.filter_.floor_ + 1e-9).sum(axis=0)
        else:
            ranking["n_detected_train"] = np.nan
    else:
        ranking["probe"], ranking["n_detected_train"] = [], []

    predictions = pd.DataFrame(
        {"sample": X.index[test_idx], "y_true": y[test_idx], "score": score, "pred": pred.astype(int)}
    )
    diagnostics = {**prep.diagnostics(), "n_train": len(train_idx), "n_test": len(test_idx),
                   "n_train_normal": int((y_train == 0).sum()), "n_test_normal": int((y[test_idx] == 0).sum()),
                   "n_positive_attribution": int(len(positive)), "n_features_used": int(len(support)),
                   "model_name": name, "prep_seconds": t_prep, "fit_seconds": t_fit}
    return FoldOutput(ranking, predictions, metrics, diagnostics, prep.genes_, [genes[i] for i in support])
