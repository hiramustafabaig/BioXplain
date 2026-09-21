"""Model factory. Configurations are fixed and pre-specified (docs/methodology_decisions.md D10, D11b); nothing is tuned.

* ``resolve_spec``  merges a config entry with the pre-specified defaults and rejects unknown keys (no silent typos);
                    the resolved spec is what is recorded in run manifests.
* ``make_model``    builds an unfitted estimator (XGBoost's class-balance weight is computed from the TRAINING labels).
* ``model_scores``  probability of the cancer class (LR, RF, XGBoost) or the margin (linear SVM) used for ROC / PR.
* ``model_support`` positions of the features the fitted model actually uses (non-zero coefficients / split features).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

IMPLEMENTED = ("logreg", "svm", "rf", "xgb")

# D11b (pre-specified 2026-09-22, before any SVM/RF/XGBoost result)
DEFAULTS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
    "svm": {"C": 1.0, "loss": "squared_hinge", "class_weight": "balanced", "max_iter": 20000},
    "rf": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced_subsample", "n_jobs": 4},
    "xgb": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.5,
            "tree_method": "hist", "n_jobs": 4},
}
# z-scored input for the regularised linear models; trees are scale-invariant and see log2 values (D8, D11b)
SCALED = {"logreg": True, "svm": True, "rf": False, "xgb": False}
PROBABILITY_MODELS = ("logreg", "rf", "xgb")


def resolve_spec(spec: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    name = spec.pop("name", None)
    if name not in IMPLEMENTED:
        raise NotImplementedError(f"model {name!r} is not implemented (implemented: {IMPLEMENTED})")
    unknown = set(spec) - set(DEFAULTS[name])
    if unknown:
        raise ValueError(f"unknown parameters for model {name!r}: {sorted(unknown)}")
    return {"name": name, **DEFAULTS[name], **spec}


def make_model(spec: dict[str, Any], seed: int, y_train=None):
    """Build an unfitted estimator from a (possibly partial) spec. ``y_train`` is needed only for XGBoost."""
    p = resolve_spec(spec)
    name = p.pop("name")
    if name == "logreg":
        # sklearn's default penalty is L2; `penalty` is deliberately not passed (deprecated in recent scikit-learn).
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"], solver="lbfgs", max_iter=p["max_iter"], random_state=seed)
    if name == "svm":
        return LinearSVC(C=p["C"], loss=p["loss"], penalty="l2", dual=True, class_weight=p["class_weight"], max_iter=p["max_iter"],
                         random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                      class_weight=p["class_weight"], n_jobs=p["n_jobs"], random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier

        if y_train is None:
            raise ValueError("XGBoost needs y_train to compute scale_pos_weight (n_normal / n_cancer, training fold only)")
        y_train = np.asarray(y_train)
        spw = float((y_train == 0).sum() / (y_train == 1).sum())
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                             subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], tree_method=p["tree_method"],
                             n_jobs=p["n_jobs"], scale_pos_weight=spw, random_state=seed, verbosity=0)
    raise AssertionError(name)


def model_scores(model, name: str, Z: np.ndarray) -> np.ndarray:
    """Score for the cancer class: probability (LR, RF, XGBoost) or signed margin (linear SVM)."""
    if name in PROBABILITY_MODELS:
        return model.predict_proba(Z)[:, 1]
    return model.decision_function(Z)


def model_support(model, name: str, n_features: int) -> np.ndarray:
    """Sorted positions of the features the fitted model uses (features outside this set have exactly zero attribution)."""
    if name in ("logreg", "svm"):
        return np.flatnonzero(np.asarray(model.coef_).ravel() != 0)
    if name == "rf":
        used = [e.tree_.feature[e.tree_.feature >= 0] for e in model.estimators_]
        return np.unique(np.concatenate(used)) if used else np.array([], dtype=int)
    if name == "xgb":
        scores = model.get_booster().get_score(importance_type="weight")
        return np.array(sorted(int(k[1:]) for k in scores), dtype=int)
    raise ValueError(name)
