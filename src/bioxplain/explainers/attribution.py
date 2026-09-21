"""Explanations/attributions -> deterministic feature rankings.

Different explainers measure different quantities (docs/methodology_decisions.md D12, D12b, D12c); they are not
interchangeable 'feature importance':
  coef   the fitted linear weight on z-scored genes (LR, linear SVM); conditional on all other genes
  shap   mean |additive attribution| of the model output over training samples; depends on model and correlations
  perm   loss increase when one gene is permuted in the training fold; diluted by correlated genes
  ttest  model-free univariate Welch statistic (baseline)
Each returns ``gene, importance, signed, rank`` (rank 1 = largest importance; deterministic ties). None is causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bioxplain.explainers.rank import rank_features

__all__ = ["rank_features", "coefficient_attribution", "explain", "VALID_EXPLAINERS"]

# which explainers are defined for which models
VALID_EXPLAINERS = {
    "coef": ("logreg", "svm"),
    "shap": ("logreg", "svm", "rf", "xgb"),
    "perm": ("logreg", "svm", "rf", "xgb"),
    "ttest": (None,),                 # model-free
}


def coefficient_attribution(model, genes) -> pd.DataFrame:
    """Linear-model coefficients on z-scored genes: importance = |coef|, signed = coef (positive = towards cancer)."""
    coef = np.asarray(model.coef_, dtype=float).ravel()
    if len(coef) != len(genes):
        raise ValueError("number of coefficients does not match number of genes")
    return rank_features(genes, np.abs(coef), coef)


EXPLAINERS = {"coef": coefficient_attribution}      # kept for backward compatibility


def explain(explainer: str, model, model_name: str | None, Z, y, genes, seed_seq: np.random.SeedSequence | None = None, n_perm: int = 10) -> pd.DataFrame:
    """Dispatch to an explainer. ``Z`` is the TRAINING-fold matrix the model was fitted on (same columns as ``genes``)."""
    if explainer not in VALID_EXPLAINERS:
        raise ValueError(f"unknown explainer {explainer!r}")
    if explainer != "ttest" and model_name not in VALID_EXPLAINERS[explainer]:
        raise ValueError(f"explainer {explainer!r} is not defined for model {model_name!r}")
    if explainer == "coef":
        return coefficient_attribution(model, genes)
    if explainer == "shap":
        from bioxplain.explainers.shap_explainer import shap_attribution
        return shap_attribution(model, model_name, Z, genes)
    if explainer == "perm":
        from bioxplain.explainers.permutation import permutation_attribution
        return permutation_attribution(model, model_name, Z, y, genes, seed_seq or np.random.SeedSequence(0), n_perm)
    from bioxplain.explainers.ttest import ttest_attribution
    return ttest_attribution(Z, y, genes)
