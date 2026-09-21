"""Explainers: SHAP, exact permutation importance (vs brute force), Welch-t baseline, dispatch."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from toy import make_toy_cohort

from bioxplain.explainers.attribution import VALID_EXPLAINERS, explain
from bioxplain.explainers.permutation import (TOLERANCE, balanced_brier, make_permutations, permutation_attribution, permutation_importance,
                                              permutation_importance_bruteforce, probability_function)
from bioxplain.explainers.shap_explainer import shap_explain
from bioxplain.explainers.ttest import ttest_attribution, welch_t
from bioxplain.models.factory import SCALED, make_model, model_scores, model_support
from bioxplain.preprocessing.transformers import FoldPreprocessor

MODELS = ("logreg", "svm", "rf", "xgb")
SMALL = {"rf": {"n_estimators": 40}, "xgb": {"n_estimators": 60}}


@pytest.fixture(scope="module")
def toy():
    X, y, m, signal = make_toy_cohort(seed=12, n_normal=14, n_cancer=50, n_signal=4, n_noise=30, n_lowexp=5)
    prep = FoldPreprocessor(m).fit(X, y)
    return prep, X, y, signal


def _fit(toy, name):
    prep, X, y, _ = toy
    Z = prep.transform(X, scale=SCALED[name])
    model = make_model({"name": name, **SMALL.get(name, {})}, 3, y).fit(Z.to_numpy(), y)
    return model, Z.to_numpy(), list(Z.columns), y


# ---------------------------------------------------------------- permutation importance
def test_balanced_brier_hand_calculation():
    y = np.array([1, 1, 0, 0, 0]); p = np.array([.9, .7, .2, .1, .0])
    assert balanced_brier(p, y) == pytest.approx(0.5 * (np.mean([.01, .09]) + np.mean([.04, .01, 0.0])))
    P = np.stack([p, np.ones(5)], axis=1)                      # matrix form: one column per scenario
    assert balanced_brier(P, y)[0] == pytest.approx(0.5 * (0.05 + 0.05 / 3)) and balanced_brier(P, y)[1] == pytest.approx(0.5 * (0.0 + 1.0))


@pytest.mark.parametrize("name", MODELS)
def test_permutation_shortcut_equals_bruteforce_for_every_feature(toy, name):
    """Every gene is permuted explicitly and the model re-predicted; the optimised implementation must agree (D12b)."""
    model, Z, genes, y = _fit(toy, name)
    perms = make_permutations(len(y), 5, np.random.SeedSequence(7))
    fast = permutation_importance(model, name, Z, y, perms)
    slow = permutation_importance_bruteforce(probability_function(model, name), Z, y, perms)
    assert fast.shape == (len(genes),)
    assert np.abs(fast - slow).max() <= TOLERANCE, np.abs(fast - slow).max()
    if name in ("rf", "xgb"):                                    # genes the model never uses: exactly zero
        unused = np.setdiff1d(np.arange(len(genes)), model_support(model, name, len(genes)))
        assert len(unused) > 0 and (fast[unused] == 0).all()
    assert (fast != 0).sum() > 0                                 # and the test can fail: some genes matter


@pytest.mark.parametrize("name", MODELS)
def test_permutation_deterministic_and_seed_dependent(toy, name):
    model, Z, genes, y = _fit(toy, name)
    a = permutation_attribution(model, name, Z, y, genes, np.random.SeedSequence([1, 0, 0]))
    b = permutation_attribution(model, name, Z, y, genes, np.random.SeedSequence([1, 0, 0]))
    c = permutation_attribution(model, name, Z, y, genes, np.random.SeedSequence([1, 0, 1]))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert not np.array_equal(a.set_index("gene").loc[genes, "importance"].to_numpy(), c.set_index("gene").loc[genes, "importance"].to_numpy())


@pytest.mark.parametrize("name", MODELS)
def test_identity_permutation_gives_exactly_zero_importance(toy, name):
    model, Z, genes, y = _fit(toy, name)
    imp = permutation_importance(model, name, Z, y, [np.arange(len(y))])
    assert np.abs(imp).max() <= 1e-15


def test_permutation_does_not_modify_the_input_matrix(toy):
    model, Z, genes, y = _fit(toy, "rf")
    before = Z.copy()
    permutation_importance(model, "rf", Z, y, make_permutations(len(y), 3, np.random.SeedSequence(0)))
    assert np.array_equal(before, Z)


def test_single_permutation_and_constant_gene_edge_cases(toy):
    model, Z, genes, y = _fit(toy, "logreg")
    Zc = Z.copy(); Zc[:, 0] = 1.0                                # constant gene: permuting it changes nothing
    m2 = make_model({"name": "logreg"}, 3, y).fit(Zc, y)
    imp = permutation_importance(m2, "logreg", Zc, y, make_permutations(len(y), 1, np.random.SeedSequence(5)))
    assert abs(imp[0]) <= 1e-15 and imp.shape == (Zc.shape[1],)             # floating-point cancellation, not bitwise 0
    r = permutation_attribution(m2, 'logreg', Zc, y, ['c'] + [f'g{i}' for i in range(Zc.shape[1] - 1)], np.random.SeedSequence(5), n_perm=1)
    assert r.set_index('gene').loc['c', 'importance'] == 0.0                  # numerical floor makes it ineligible (D13b)


def test_feature_order_alignment_planted_gene_is_top_for_every_explainer():
    """Only column 5 carries the label; the top-ranked NAME must be genes[5] for coef, SHAP and permutation importance."""
    rng = np.random.default_rng(0)
    n, p = 120, 12
    y = np.r_[np.zeros(30, int), np.ones(90, int)]
    Z = rng.normal(size=(n, p)); Z[:, 5] += 3 * y
    genes = [f"G{i}" for i in range(p)]
    for name in MODELS:
        model = make_model({"name": name, **SMALL.get(name, {})}, 0, y).fit(Z, y)
        for expl in ("coef", "shap", "perm"):
            if name not in VALID_EXPLAINERS[expl]:
                continue
            r = explain(expl, model, name, Z, y, genes, np.random.SeedSequence(1))
            assert r.iloc[0]["gene"] == "G5", (name, expl)
            assert r["rank"].tolist() == list(range(1, p + 1))


# ---------------------------------------------------------------- SHAP
def test_linear_shap_known_answer():
    """Interventional LinearSHAP: phi_ij = w_j * (x_ij - mean_j)  =>  importance_j = |w_j| * mean|x - mean|."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(80, 4)) * np.array([1, 2, 0.5, 3])
    y = (Z[:, 0] + Z[:, 1] > 0).astype(int)
    model = make_model({"name": "logreg"}, 0, y).fit(Z, y)
    vals, base = shap_explain(model, "logreg", Z)
    expected = model.coef_.ravel() * (Z - Z.mean(0))
    np.testing.assert_allclose(vals, expected, atol=1e-10)
    np.testing.assert_allclose(vals.sum(1) + base, model.decision_function(Z), atol=1e-10)


@pytest.mark.parametrize("name", MODELS)
def test_shap_dimensions_additivity_and_zero_outside_support(toy, name):
    model, Z, genes, y = _fit(toy, name)
    vals, base = shap_explain(model, name, Z)
    assert vals.shape == Z.shape
    out = {"logreg": lambda: model.decision_function(Z), "svm": lambda: model.decision_function(Z),
           "rf": lambda: model.predict_proba(Z)[:, 1], "xgb": lambda: model.predict(Z, output_margin=True)}[name]()
    assert np.abs(vals.sum(1) + base - out).max() < 1e-4        # float32 tree arithmetic for XGBoost
    if name in ("rf", "xgb"):
        unused = np.setdiff1d(np.arange(Z.shape[1]), model_support(model, name, Z.shape[1]))
        assert (vals[:, unused] == 0).all()
    v2, _ = shap_explain(model, name, Z)
    assert np.array_equal(vals, v2)                              # deterministic


def test_xgboost_base_value_regression():
    """Regression (2026-09-22): a fresh TreeExplainer's expected_value differs from the one after shap_values();
    shap_explain must return the value consistent with additivity."""
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(100, 40)); y = (Z[:, 2] + 0.3 * rng.normal(size=100) > 0).astype(int)
    model = make_model({"name": "xgb", "n_estimators": 80}, 0, y).fit(Z, y)
    vals, base = shap_explain(model, "xgb", Z)
    assert np.abs(vals.sum(1) + base - model.predict(Z, output_margin=True)).max() < 1e-4


# ---------------------------------------------------------------- t-test baseline
def test_welch_t_matches_scipy_and_sign_convention():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(12, int), np.ones(30, int)]
    Z = rng.normal(size=(42, 6)); Z[y == 1, 0] += 2.0; Z[y == 0, 1] += 2.0
    t = welch_t(Z, y)
    ref = stats.ttest_ind(Z[y == 1], Z[y == 0], equal_var=False).statistic
    np.testing.assert_allclose(t, ref, atol=1e-12)
    assert t[0] > 0 > t[1]                                       # positive = higher in cancer


def test_ttest_ranking_ties_zero_variance_alignment_and_class_handling():
    y = np.r_[np.zeros(5, int), np.ones(5, int)]
    base = np.r_[np.zeros(5), np.ones(5)]
    Z = np.stack([base, base, np.full(10, 3.0), base * 0 + np.arange(10) % 2], axis=1)     # cols 0,1 tie exactly; col 2 constant
    r = ttest_attribution(Z, y, ["b", "a", "const", "noise"])
    # cols 'b' and 'a' are perfectly separated with zero within-class variance: t = +-inf -> capped, ranked first, tie broken by name
    assert r["gene"].tolist()[:2] == ["a", "b"] and r.iloc[0]["importance"] == 1e9
    assert r.set_index("gene").loc["const", "importance"] == 0.0                            # equal means, zero variance -> t = 0
    with pytest.raises(ValueError, match="at least two"):
        welch_t(Z, np.r_[np.zeros(9, int), 1])


# ---------------------------------------------------------------- dispatch
def test_dispatch_rejects_undefined_combinations(toy):
    model, Z, genes, y = _fit(toy, "rf")
    with pytest.raises(ValueError, match="not defined"):
        explain("coef", model, "rf", Z, y, genes)
    with pytest.raises(ValueError, match="unknown explainer"):
        explain("lime", model, "rf", Z, y, genes)
    assert explain("ttest", None, None, Z, y, genes).shape[0] == len(genes)
