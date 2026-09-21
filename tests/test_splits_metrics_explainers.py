import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bioxplain.evaluation.metrics import bootstrap_ci, classification_metrics
from bioxplain.explainers.attribution import coefficient_attribution, rank_features
from bioxplain.stability.summary import stability_summary, top_k_sets
from bioxplain.validation.splits import repeated_stratified_splits

Y = np.r_[np.zeros(17, int), np.ones(104, int)]        # class-blocked, exactly like GSE42568


def test_splits_partition_the_data_and_are_stratified():
    splits = list(repeated_stratified_splits(Y, 5, 3, seed=7))
    assert len(splits) == 15
    for rep in range(3):
        tests = [s.test_idx for s in splits if s.repeat == rep]
        allidx = np.concatenate(tests)
        assert sorted(allidx) == list(range(len(Y)))                       # every sample tested exactly once per repeat
    for s in splits:
        assert np.intersect1d(s.train_idx, s.test_idx).size == 0
        assert len(s.train_idx) + len(s.test_idx) == len(Y)
        assert 3 <= (Y[s.test_idx] == 0).sum() <= 4                        # 17 normals over 5 folds
        assert (Y[s.train_idx] == 0).sum() >= 13


def test_splits_are_shuffled_not_contiguous_blocks():
    s = next(iter(repeated_stratified_splits(Y, 5, 1, seed=0)))
    assert not np.array_equal(np.sort(s.test_idx), np.arange(len(s.test_idx)))
    assert (s.test_idx < 17).sum() == (Y[s.test_idx] == 0).sum() and (Y[s.test_idx] == 0).sum() > 0


def test_splits_are_reproducible_and_differ_between_repeats_and_seeds():
    a = list(repeated_stratified_splits(Y, 5, 2, seed=1))
    b = list(repeated_stratified_splits(Y, 5, 2, seed=1))
    c = list(repeated_stratified_splits(Y, 5, 2, seed=2))
    assert all(np.array_equal(x.test_idx, y.test_idx) for x, y in zip(a, b))
    assert not np.array_equal(a[0].test_idx, a[5].test_idx)                # repeat 0 vs repeat 1
    assert not np.array_equal(a[0].test_idx, c[0].test_idx)
    assert [s.seed for s in a] == [1] * 5 + [2] * 5                        # seed = master + repeat (D22)


def test_split_input_order_does_not_change_group_structure():
    perm = np.random.default_rng(0).permutation(len(Y))
    for s in repeated_stratified_splits(Y[perm], 5, 1, seed=3):
        assert 3 <= (Y[perm][s.test_idx] == 0).sum() <= 4


def test_splits_reject_impossible_designs():
    with pytest.raises(ValueError, match="fewer than"):
        list(repeated_stratified_splits(np.r_[np.zeros(3, int), np.ones(30, int)], 5, 1, 0))
    with pytest.raises(ValueError, match="binary"):
        list(repeated_stratified_splits(np.arange(10), 2, 1, 0))


def test_classification_metrics_toy():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    score = np.array([.1, .2, .3, .6, .4, .7, .8, .9, .9, .95])
    pred = (score >= .5).astype(int)
    m = classification_metrics(y, score, pred)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 5)
    assert m["sensitivity"] == pytest.approx(5 / 6) and m["specificity"] == pytest.approx(3 / 4)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 3 / 4) / 2)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, score))
    assert 0 <= m["ap_normal"] <= 1


def test_metrics_penalise_always_cancer_prediction():
    y = np.r_[np.zeros(17, int), np.ones(104, int)]
    m = classification_metrics(y, np.full(121, 0.9), np.ones(121, int))
    assert m["balanced_accuracy"] == pytest.approx(0.5) and m["specificity"] == 0.0
    assert (m["tp"] + m["tn"]) / 121 > 0.85           # plain accuracy would look excellent


def test_bootstrap_ci_reproducible_and_contains_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(15, int), np.ones(45, int)]
    s = np.r_[rng.normal(0, 1, 15), rng.normal(1.5, 1, 45)]
    ci1, ci2 = (bootstrap_ci(y, s, roc_auc_score, 300, seed=5) for _ in range(2))
    assert ci1 == ci2 and ci1[0] < roc_auc_score(y, s) < ci1[1]
    with pytest.raises(ValueError):
        bootstrap_ci(np.ones(5, int), np.arange(5.0), roc_auc_score, 10)


def test_rank_features_is_deterministic_with_ties():
    r = rank_features(["b", "a", "c", "d"], [1.0, 1.0, 3.0, 0.5], [-1.0, 1.0, 3.0, 0.5])
    assert r["gene"].tolist() == ["c", "a", "b", "d"]                    # tie a/b broken alphabetically
    assert r["rank"].tolist() == [1, 2, 3, 4]
    assert r.set_index("gene").loc["b", "signed"] == -1.0
    with pytest.raises(ValueError):
        rank_features(["a"], [np.nan])


def test_coefficient_attribution_ranks_by_absolute_coefficient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 2] * 3 - X[:, 4] * 1.5 + rng.normal(0, .3, 200) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    r = coefficient_attribution(model, list("abcde"))
    assert r["gene"].tolist()[:2] == ["c", "e"]
    assert r.set_index("gene").loc["c", "signed"] > 0 > r.set_index("gene").loc["e", "signed"]
    with pytest.raises(ValueError):
        coefficient_attribution(model, list("abc"))


def test_default_logistic_penalty_is_l2():
    """factory.py relies on sklearn's default being L2 (it does not pass `penalty`)."""
    m = LogisticRegression()
    # the fitted model must shrink rather than zero coefficients: no exact zeros on informative noise-free data
    X = np.random.default_rng(0).normal(size=(60, 20)); y = (X[:, 0] > 0).astype(int)
    assert (m.fit(X, y).coef_ == 0).sum() == 0


def test_top_k_sets_and_summary():
    rows = []
    for rep in range(2):
        for fold in range(3):
            for r in range(1, 6):
                rows.append({"repeat": rep, "fold": fold, "rank": r, "gene": f"g{r}"})   # identical rankings
    rk = pd.DataFrame(rows)
    assert all(s == {"g1", "g2"} for s in top_k_sets(rk, 2))
    s = stability_summary(rk, [2, 3], n_features=50, n_features_alt=20)
    assert s.loc[s.scope == "all_runs", "nogueira"].tolist() == [pytest.approx(1.0)] * 2
    assert set(s.scope) == {"all_runs", "repeat_0", "repeat_1"} and (s.n_features_alt == 20).all()
    with pytest.raises(ValueError, match="store_top"):
        top_k_sets(rk, 10)
