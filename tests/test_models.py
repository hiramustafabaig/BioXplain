"""Model factory and per-model contract tests (toy data; the real-data benchmark is scripts/benchmark_models.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from toy import make_toy_cohort

import bioxplain.validation.cv as cv
from bioxplain.models.factory import DEFAULTS, IMPLEMENTED, PROBABILITY_MODELS, SCALED, make_model, model_scores, model_support, resolve_spec
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.validation.splits import repeated_stratified_splits

PREP = {"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean"}
EXPLAINER = {"logreg": "coef", "svm": "coef", "rf": None, "xgb": None}


@pytest.fixture(scope="module")
def cohort():
    return make_toy_cohort(seed=8, n_normal=14, n_cancer=60, n_signal=6, n_noise=150)


def test_defaults_are_the_preregistered_D11b_values():
    """Guards against silent drift of the pre-specified configurations (docs/methodology_decisions.md D11b)."""
    assert DEFAULTS == {
        "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
        "svm": {"C": 1.0, "loss": "squared_hinge", "class_weight": "balanced", "max_iter": 20000},
        "rf": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced_subsample", "n_jobs": 4},
        "xgb": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.5,
                "tree_method": "hist", "n_jobs": 4},
    }
    assert SCALED == {"logreg": True, "svm": True, "rf": False, "xgb": False}
    assert IMPLEMENTED == ("logreg", "svm", "rf", "xgb")


def test_resolve_spec_merges_defaults_and_rejects_bad_input():
    assert resolve_spec({"name": "logreg", "C": 0.1})["C"] == 0.1
    assert resolve_spec({"name": "rf"})["n_estimators"] == 500
    with pytest.raises(ValueError, match="unknown parameters"):
        resolve_spec({"name": "svm", "Cc": 1.0})
    with pytest.raises(NotImplementedError):
        resolve_spec({"name": "knn"})
    with pytest.raises(NotImplementedError):
        resolve_spec({})


def test_estimators_carry_the_specified_parameters():
    y = np.r_[np.zeros(5, int), np.ones(20, int)]
    lr, svm, rf, xgb = (make_model({"name": n}, 3, y) for n in IMPLEMENTED)
    assert lr.get_params()["C"] == 1.0 and lr.get_params()["class_weight"] == "balanced" and lr.get_params()["random_state"] == 3
    p = svm.get_params()
    assert (p["C"], p["loss"], p["penalty"], p["dual"], p["class_weight"], p["max_iter"]) == (1.0, "squared_hinge", "l2", True, "balanced", 20000)
    p = rf.get_params()
    assert (p["n_estimators"], p["max_features"], p["class_weight"], p["n_jobs"], p["random_state"]) == (500, "sqrt", "balanced_subsample", 4, 3)
    p = xgb.get_params()
    assert (p["n_estimators"], p["max_depth"], p["learning_rate"], p["subsample"], p["colsample_bytree"], p["n_jobs"]) == (300, 3, 0.05, 0.8, 0.5, 4)


def test_xgb_class_weight_is_computed_from_training_labels_only():
    y_train = np.r_[np.zeros(5, int), np.ones(20, int)]
    assert make_model({"name": "xgb"}, 0, y_train).get_params()["scale_pos_weight"] == pytest.approx(5 / 20)
    with pytest.raises(ValueError, match="y_train"):
        make_model({"name": "xgb"}, 0)


def _split(y):
    return next(iter(repeated_stratified_splits(y, 5, 1, seed=11)))


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_fold_contract_scores_labels_dimensions_and_signal(cohort, name):
    X, y, m, signal = cohort
    sp = _split(y)
    out = cv.run_fold(X, y, sp.train_idx, sp.test_idx, model_spec={"name": name}, explainer_name=EXPLAINER[name], probe_to_gene=m, prep_cfg=PREP, seed=5)
    pr = out.predictions
    assert len(pr) == len(sp.test_idx) and np.isfinite(pr.score).all() and set(pr.pred) <= {0, 1}
    if name in PROBABILITY_MODELS:
        assert pr.score.between(0, 1).all()
    else:                                                          # SVM: margin; predicted label is its sign
        assert (pr.pred == (pr.score > 0).astype(int)).all() and not pr.score.between(0, 1).all()
    assert out.diagnostics["n_features_used"] == len(out.support_genes) <= out.diagnostics["n_genes"]
    assert set(out.support_genes) <= set(out.eligible_genes)
    assert out.metrics["roc_auc"] > 0.9                            # toy cohort has strong planted signal
    if name in ("rf", "xgb"):
        assert len(set(signal) & set(out.support_genes)) >= 4       # trees split on the informative genes
        assert len(out.ranking) == 0                                # explainers arrive in Phase 4 (documented plan)
    else:
        assert len(out.ranking) == out.diagnostics["n_positive_attribution"] == out.diagnostics["n_genes"]
        assert set(signal) <= set(out.ranking.head(12)["gene"])     # coefficients rank the planted genes near the top


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_scaling_policy_follows_the_model(cohort, name, monkeypatch):
    X, y, m, _ = cohort
    seen = []

    class Spy(FoldPreprocessor):
        def __init__(self, *a, **kw):
            seen.append(kw["scale"])
            super().__init__(*a, **kw)

    monkeypatch.setattr(cv, "FoldPreprocessor", Spy)
    sp = _split(y)
    cv.run_fold(X, y, sp.train_idx, sp.test_idx, model_spec={"name": name}, explainer_name=None, probe_to_gene=m, prep_cfg=PREP, seed=5)
    assert seen == [SCALED[name]]
    seen.clear()
    cv.run_fold(X, y, sp.train_idx, sp.test_idx, model_spec={"name": name}, explainer_name=None, probe_to_gene=m, prep_cfg={**PREP, "scale": not SCALED[name]}, seed=5)
    assert seen == [not SCALED[name]]                              # an explicit config value wins


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_deterministic_at_fixed_thread_count(cohort, name):
    X, y, m, _ = cohort
    sp = _split(y)
    kw = dict(model_spec={"name": name}, explainer_name=EXPLAINER[name], probe_to_gene=m, prep_cfg=PREP, seed=5)
    a, b = (cv.run_fold(X, y, sp.train_idx, sp.test_idx, **kw) for _ in range(2))
    pd.testing.assert_frame_equal(a.predictions, b.predictions, check_exact=True)
    assert a.support_genes == b.support_genes


def test_random_forest_is_thread_count_invariant(cohort):
    X, y, m, _ = cohort
    Z = FoldPreprocessor(m, scale=False).fit_transform(X, y).to_numpy()
    s1 = model_scores(make_model({"name": "rf", "n_jobs": 1}, 7).fit(Z, y), "rf", Z)
    s4 = model_scores(make_model({"name": "rf", "n_jobs": 4}, 7).fit(Z, y), "rf", Z)
    assert np.array_equal(s1, s4)


def test_xgboost_thread_count_is_part_of_the_pinned_configuration():
    """Empirically (2026-09-22) XGBoost scores differ by ~1e-2 between n_jobs=1 and 4, so n_jobs is fixed in the spec and recorded."""
    assert DEFAULTS["xgb"]["n_jobs"] == 4 and resolve_spec({"name": "xgb"})["n_jobs"] == 4


@pytest.mark.parametrize("name,extra", [("rf", {"n_estimators": 30}), ("xgb", {})])
def test_features_outside_model_support_have_exactly_zero_permutation_effect(name, extra):
    """Premise of the exact permutation-importance shortcut (D12b): an unused feature cannot influence any prediction."""
    X, y, m, _ = make_toy_cohort(seed=9, n_normal=14, n_cancer=60, n_noise=600)
    Z = FoldPreprocessor(m, scale=False).fit_transform(X, y).to_numpy()
    model = make_model({"name": name, **extra}, 7, y).fit(Z, y)
    base = model_scores(model, name, Z)
    support = set(model_support(model, name, Z.shape[1]).tolist())
    unused = [j for j in range(Z.shape[1]) if j not in support]
    assert len(unused) > 50 and len(support) > 0
    rng = np.random.default_rng(0)
    for j in rng.choice(unused, 40, replace=False):
        Zp = Z.copy(); Zp[:, j] = rng.permutation(Zp[:, j])
        assert np.array_equal(model_scores(model, name, Zp), base)
    changed = 0
    for j in sorted(support)[:30]:
        Zp = Z.copy(); Zp[:, j] = rng.permutation(Zp[:, j]); changed += int(not np.array_equal(model_scores(model, name, Zp), base))
    assert changed > 0                                              # and the used ones do matter (the test can fail)


def test_model_support_for_linear_models_is_non_zero_coefficients():
    X, y, m, _ = make_toy_cohort(seed=1)
    Z = FoldPreprocessor(m, scale=True).fit_transform(X, y).to_numpy()
    for name in ("logreg", "svm"):
        model = make_model({"name": name}, 0).fit(Z, y)
        assert np.array_equal(model_support(model, name, Z.shape[1]), np.flatnonzero(model.coef_.ravel() != 0))


def test_all_models_beat_chance_on_toy_and_stay_at_chance_on_noise_labels(cohort):
    X, y, m, _ = cohort
    y_noise = np.random.default_rng(3).permutation(y)
    for name in IMPLEMENTED:
        scores, scores_noise = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for sp in repeated_stratified_splits(y, 4, 1, seed=2):
            kw = dict(model_spec={"name": name}, explainer_name=None, probe_to_gene=m, prep_cfg=PREP, seed=5)
            scores[sp.test_idx] = cv.run_fold(X, y, sp.train_idx, sp.test_idx, **kw).predictions.score.to_numpy()
        for sp in repeated_stratified_splits(y_noise, 4, 1, seed=2):
            scores_noise[sp.test_idx] = cv.run_fold(X, y_noise, sp.train_idx, sp.test_idx, **kw).predictions.score.to_numpy()
        assert roc_auc_score(y, scores) > 0.95, name
        assert 0.2 < roc_auc_score(y_noise, scores_noise) < 0.8, name
