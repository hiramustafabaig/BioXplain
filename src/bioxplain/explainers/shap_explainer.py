"""SHAP attributions (decision D12c).

Population: ALL TRAINING-FOLD samples, for every model (no different explanation populations between models).
importance = mean |SHAP| over those samples; signed = mean SHAP. Model output explained:
  LR / linear SVM : decision function (log-odds / margin), LinearExplainer, interventional, training-fold background
  Random forest   : probability of the cancer class, TreeExplainer (tree_path_dependent)
  XGBoost         : log-odds (raw margin), TreeExplainer
SHAP values explain the fitted model's output, not causal effects of genes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bioxplain.explainers.rank import rank_features


def shap_explain(model, name: str, Z: np.ndarray) -> tuple[np.ndarray, float]:
    """(SHAP values (n_samples, n_genes) of the cancer-class output, base value) from ONE explainer object.

    The base value must be read from the same explainer AFTER shap_values() has run: for XGBoost a freshly built
    TreeExplainer reports a different expected_value (found 2026-09-22; values themselves were unaffected).
    """
    import shap

    Z = np.asarray(Z, dtype=float)
    if name in ("logreg", "svm"):
        explainer = shap.LinearExplainer(model, Z)
        vals = explainer.shap_values(Z)
    elif name in ("rf", "xgb"):
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Z, check_additivity=True)
    else:
        raise ValueError(f"no SHAP explainer for model {name!r}")
    vals = np.asarray(vals) if not isinstance(vals, list) else np.stack(vals, axis=-1)
    ev = np.ravel(explainer.expected_value)
    if vals.ndim == 3:                                     # (n, p, n_classes) -> cancer class
        vals, base = vals[:, :, 1], float(ev[-1])
    else:
        base = float(ev[0])
    if vals.shape != Z.shape:
        raise ValueError(f"SHAP output {vals.shape} does not match input {Z.shape}")
    return vals, base


def shap_values(model, name: str, Z: np.ndarray) -> np.ndarray:
    return shap_explain(model, name, Z)[0]


def shap_attribution(model, name: str, Z: np.ndarray, genes) -> pd.DataFrame:
    v = shap_values(model, name, Z)
    return rank_features(genes, np.abs(v).mean(axis=0), v.mean(axis=0))
