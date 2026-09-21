"""The vectorised bootstrap must be mathematically equivalent to the reference implementation."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from bioxplain.evaluation.bootstrap import (
    bootstrap_intervals,
    stratified_bootstrap_weights,
    weighted_average_precision,
    weighted_balanced_accuracy,
    weighted_roc_auc,
)
from bioxplain.evaluation.metrics import bootstrap_ci

TOL = 1e-12


def _data(seed=0, n0=17, n1=104, ties=True):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    s = np.r_[rng.normal(0, 1, n0), rng.normal(1.2, 1, n1)]
    if ties:
        s = np.round(1 / (1 + np.exp(-s)), 2)          # coarse scores -> many exact ties, like saturated probabilities
    return y, s, (s >= 0.5).astype(int)


def test_weights_have_the_stratified_structure():
    y, _, _ = _data()
    w = stratified_bootstrap_weights(y, 50, seed=3)
    assert w.shape == (50, len(y)) and (w >= 0).all()
    assert (w[:, y == 0].sum(1) == 17).all() and (w[:, y == 1].sum(1) == 104).all()     # class sizes preserved
    assert np.array_equal(w, stratified_bootstrap_weights(y, 50, seed=3))              # reproducible
    assert not np.array_equal(w, stratified_bootstrap_weights(y, 50, seed=4))


def test_weights_use_the_same_resamples_as_the_reference_implementation():
    y, _, _ = _data()
    w = stratified_bootstrap_weights(y, 5, seed=11)
    rng = np.random.default_rng(11)                                                     # reference draw order
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    for b in range(5):
        idx = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))])
        assert np.array_equal(np.bincount(idx, minlength=len(y)), w[b])


@pytest.mark.parametrize("ties", [True, False])
def test_metrics_equal_scikit_learn_on_explicit_resamples(ties):
    y, s, pred = _data(seed=2, ties=ties)
    w = stratified_bootstrap_weights(y, 40, seed=5)
    auc = weighted_roc_auc(y, s, w)
    ap = weighted_average_precision(y == 0, 1 - s, w)
    ba = weighted_balanced_accuracy(y, pred, w)
    for b in range(40):
        idx = np.repeat(np.arange(len(y)), w[b])
        yb, sb, pb = y[idx], s[idx], pred[idx]
        assert auc[b] == pytest.approx(roc_auc_score(yb, sb), abs=TOL)
        assert ap[b] == pytest.approx(average_precision_score(1 - yb, 1 - sb), abs=TOL)
        assert ba[b] == pytest.approx(balanced_accuracy_score(yb, pb), abs=TOL)


def test_intervals_equal_the_reference_bootstrap_ci_for_every_metric():
    y, s, pred = _data(seed=7)
    new = bootstrap_intervals(y, s, pred, n_boot=300, seed=20260921)
    ref = {
        "roc_auc": bootstrap_ci(y, s, roc_auc_score, 300, 20260921),
        "ap_normal": bootstrap_ci(y, s, lambda a, b: average_precision_score(1 - a, 1 - b), 300, 20260921),
        "balanced_accuracy": bootstrap_ci(y, s, lambda a, b: balanced_accuracy_score(a, (b >= 0.5).astype(int)), 300, 20260921),
    }
    for k in ref:
        assert new[k] == pytest.approx(ref[k], abs=TOL), k


def test_known_values():
    y = np.r_[np.zeros(4, int), np.ones(6, int)]
    w = np.ones((1, 10), dtype=np.int64)
    assert weighted_roc_auc(y, np.r_[np.zeros(4), np.ones(6)], w)[0] == 1.0             # perfect separation
    assert weighted_roc_auc(y, np.full(10, .5), w)[0] == 0.5                             # all tied
    assert weighted_average_precision(y == 0, 1 - np.r_[np.zeros(4), np.ones(6)], w)[0] == 1.0
    assert weighted_balanced_accuracy(y, np.ones(10, int), w)[0] == 0.5                  # always-cancer predictor


def test_a_resample_missing_high_scoring_samples_is_handled():
    """Zero-weight thresholds (0/0 precision) must not create NaNs."""
    y, s, _ = _data(seed=1, n0=5, n1=8)
    w = np.zeros((1, len(y)), dtype=np.int64)
    w[0, [0, 1, 6]] = [3, 2, 8]                                                          # only 2 distinct samples per class kept
    ap = weighted_average_precision(y == 0, 1 - s, w)
    idx = np.repeat(np.arange(len(y)), w[0])
    assert ap[0] == pytest.approx(average_precision_score(1 - y[idx], 1 - s[idx]), abs=TOL)


def test_single_class_is_rejected():
    with pytest.raises(ValueError):
        stratified_bootstrap_weights(np.ones(5, int), 10, 0)
