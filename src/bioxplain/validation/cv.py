"""Run ONE cross-validation fold in a leakage-safe way.

Contract: everything data-dependent (expression filter, probe collapse, scaling, model fit, explanation)
is computed from ``X.iloc[train_idx]`` / ``y[train_idx]`` only. ``X.iloc[test_idx]`` is used solely to
produce predictions for evaluation. tests/test_leakage.py enforces this contract (fold-perturbation
invariance, fit-index spying, and a deliberately leaky variant that the tests must catch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bioxplain.evaluation.metrics import classification_metrics
from bioxplain.explainers.attribution import EXPLAINERS
from bioxplain.models.factory import make_model
from bioxplain.preprocessing.transformers import FoldPreprocessor


@dataclass
class FoldOutput:
    ranking: pd.DataFrame       # top `store_top` genes: gene, probe, importance, signed, rank, n_detected_train
    predictions: pd.DataFrame   # test samples: sample, y_true, score, pred
    metrics: dict
    diagnostics: dict
    eligible_genes: pd.Index    # genes that could have been ranked in this fold


def run_fold(
    X: pd.DataFrame,
    y,
    train_idx,
    test_idx,
    *,
    model_spec: dict,
    explainer_name: str,
    probe_to_gene: pd.Series,
    prep_cfg: dict,
    seed: int,
    store_top: int = 200,
) -> FoldOutput:
    y = np.asarray(y).astype(int)
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train and test indices overlap")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    prep = FoldPreprocessor(
        probe_to_gene,
        min_detect_frac_of_minority=prep_cfg.get("min_detect_frac_of_minority", 0.5),
        collapse_rule=prep_cfg.get("collapse_rule", "max_mean"),
        scale=prep_cfg.get("scale", True),
    )
    Z_train = prep.fit_transform(X_train, y_train)              # <- learned on training samples only
    model = make_model(model_spec, seed).fit(Z_train.to_numpy(), y_train)
    attribution = EXPLAINERS[explainer_name](model, list(Z_train.columns))   # <- from the training-fit model

    Z_test = prep.transform(X.iloc[test_idx])                    # <- validation fold: transform only
    score = model.predict_proba(Z_test.to_numpy())[:, 1]
    pred = model.predict(Z_test.to_numpy())
    metrics = classification_metrics(y[test_idx], score, pred)

    ranking = attribution.head(store_top).copy()
    ranking["probe"] = prep.collapser_.selected_.loc[ranking["gene"]].to_numpy()
    if prep.filter_.floor_ is not None:   # how many TRAINING samples detect the chosen probe (artefact check, D8)
        raw = X_train.loc[:, ranking["probe"].to_numpy()].to_numpy()
        ranking["n_detected_train"] = (raw > prep.filter_.floor_ + 1e-9).sum(axis=0)
    else:
        ranking["n_detected_train"] = np.nan

    predictions = pd.DataFrame(
        {"sample": X.index[test_idx], "y_true": y[test_idx], "score": score, "pred": pred.astype(int)}
    )
    diagnostics = {**prep.diagnostics(), "n_train": len(train_idx), "n_test": len(test_idx),
                   "n_train_normal": int((y_train == 0).sum()), "n_test_normal": int((y[test_idx] == 0).sum())}
    return FoldOutput(ranking, predictions, metrics, diagnostics, prep.genes_)
