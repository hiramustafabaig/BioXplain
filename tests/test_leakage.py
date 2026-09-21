"""Leakage tests.

Contract (validation/cv.py): everything learned is computed from training-fold samples only. Three
independent checks + one demonstration of why it matters:

1. SPY: record the sample IDs every ``fit`` receives and assert none belongs to the validation fold.
2. INVARIANCE: arbitrarily corrupt the validation-fold rows; every learned artefact (kept probes, chosen
   probe per gene, scaler statistics, coefficients, ranking) must be bit-identical. Any learned step that
   accidentally saw the validation rows would change.
3. DETECTION: a deliberately LEAKY fold runner (preprocessing fitted on all samples) is run through the same
   invariance check and MUST fail it, proving the check has teeth.
4. WHY IT MATTERS: supervised feature selection on all samples before CV turns pure-noise labels into
   "high AUC"; the leakage-safe protocol stays at chance.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from toy import make_toy_cohort

import bioxplain.validation.cv as cv
from bioxplain.explainers.attribution import coefficient_attribution
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.validation.splits import repeated_stratified_splits

CFG = dict(model_spec={"name": "logreg", "C": 1.0, "class_weight": "balanced", "max_iter": 2000},
           explainer_name="coef", prep_cfg={"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean", "scale": True},
           seed=11, store_top=40)


@pytest.fixture(scope="module")
def cohort():
    X, y, m, signal = make_toy_cohort(seed=3)
    return X, y, m, signal


def _first_split(y):
    return next(iter(repeated_stratified_splits(y, 5, 1, seed=11)))


def _learned_artifacts(out):
    return out.ranking[["gene", "probe", "importance", "signed", "rank"]].reset_index(drop=True), out.diagnostics


def test_fit_never_sees_validation_samples(cohort, monkeypatch):
    X, y, m, _ = cohort
    seen = []

    class Spy(FoldPreprocessor):
        def fit(self, X_train, y_train):
            seen.append(set(X_train.index))
            return super().fit(X_train, y_train)

    monkeypatch.setattr(cv, "FoldPreprocessor", Spy)
    sp = _first_split(y)
    cv.run_fold(X, y, sp.train_idx, sp.test_idx, probe_to_gene=m, **CFG)
    test_ids = set(X.index[sp.test_idx])
    assert len(seen) == 1 and seen[0].isdisjoint(test_ids) and seen[0] == set(X.index[sp.train_idx])


def test_overlapping_train_and_test_indices_are_rejected(cohort):
    X, y, m, _ = cohort
    with pytest.raises(ValueError, match="overlap"):
        cv.run_fold(X, y, np.arange(30), np.arange(25, 40), probe_to_gene=m, **CFG)


def _corrupt(X, idx):
    Xc = X.copy()
    Xc.iloc[idx, :] = 15.0 + np.random.default_rng(0).normal(size=(len(idx), X.shape[1]))
    return Xc


def test_learned_artifacts_are_invariant_to_validation_fold_contents(cohort):
    X, y, m, _ = cohort
    sp = _first_split(y)
    base = cv.run_fold(X, y, sp.train_idx, sp.test_idx, probe_to_gene=m, **CFG)
    pert = cv.run_fold(_corrupt(X, sp.test_idx), y, sp.train_idx, sp.test_idx, probe_to_gene=m, **CFG)
    r0, d0 = _learned_artifacts(base)
    r1, d1 = _learned_artifacts(pert)
    pd.testing.assert_frame_equal(r0, r1, check_exact=True)
    assert d0 == d1
    assert list(base.eligible_genes) == list(pert.eligible_genes)
    assert not np.allclose(base.predictions["score"], pert.predictions["score"])   # the corruption did reach predictions


def leaky_run_fold(X, y, train_idx, test_idx, *, probe_to_gene, prep_cfg, seed, **_):
    """DELIBERATELY WRONG: preprocessing is fitted on ALL samples (including validation rows)."""
    prep = FoldPreprocessor(probe_to_gene, prep_cfg["min_detect_frac_of_minority"], prep_cfg["collapse_rule"], True)
    prep.fit(X, y)                                                # <-- the leak
    Z = prep.transform(X)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(Z.iloc[train_idx].to_numpy(), y[train_idx])
    ranking = coefficient_attribution(model, list(Z.columns)).head(40)
    return ranking[["gene", "importance", "signed", "rank"]].reset_index(drop=True)


def test_invariance_check_catches_a_deliberately_leaky_pipeline(cohort):
    """If a future change fits any learned transform on all samples, THIS mechanism detects it."""
    X, y, m, _ = cohort
    sp = _first_split(y)
    kw = dict(probe_to_gene=m, prep_cfg=CFG["prep_cfg"], seed=CFG["seed"])
    clean = leaky_run_fold(X, y, sp.train_idx, sp.test_idx, **kw)
    corrupted = leaky_run_fold(_corrupt(X, sp.test_idx), y, sp.train_idx, sp.test_idx, **kw)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(clean, corrupted, check_exact=True)


def test_leaky_probe_collapse_alone_is_detected(cohort):
    """Even the mild, label-free leak (choosing the probe per gene on all data) changes the chosen probes."""
    X, y, m, _ = cohort
    sp = _first_split(y)
    from bioxplain.preprocessing.transformers import ProbeGeneCollapser
    safe = ProbeGeneCollapser(m).fit(X.iloc[sp.train_idx])
    leaky_all = ProbeGeneCollapser(m).fit(_corrupt(X, sp.test_idx))     # sees corrupted validation rows
    assert not safe.selected_.equals(leaky_all.selected_)
    safe_again = ProbeGeneCollapser(m).fit(_corrupt(X, sp.test_idx).iloc[sp.train_idx])
    assert safe.selected_.equals(safe_again.selected_)


def test_supervised_selection_before_cv_inflates_auc_on_pure_noise_but_safe_protocol_does_not():
    rng = np.random.default_rng(0)
    n, p, k = 40, 3000, 20
    X = rng.normal(size=(n, p))
    y = np.r_[np.zeros(20, int), np.ones(20, int)]                # labels carry NO information
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def top_k_by_ttest(Xs, ys):
        a, b = Xs[ys == 1], Xs[ys == 0]
        t = (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        return np.argsort(-np.abs(t))[:k]

    leaky_sel = top_k_by_ttest(X, y)                              # selection on ALL samples  (LEAK)
    leaky_scores = np.zeros(n); safe_scores = np.zeros(n)
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=1000).fit(X[tr][:, leaky_sel], y[tr])
        leaky_scores[te] = m.predict_proba(X[te][:, leaky_sel])[:, 1]
        sel = top_k_by_ttest(X[tr], y[tr])                        # selection inside the training fold (SAFE)
        m = LogisticRegression(max_iter=1000).fit(X[tr][:, sel], y[tr])
        safe_scores[te] = m.predict_proba(X[te][:, sel])[:, 1]
    leaky_auc, safe_auc = roc_auc_score(y, leaky_scores), roc_auc_score(y, safe_scores)
    assert leaky_auc > 0.85, leaky_auc          # spurious "discovery" from noise
    assert safe_auc < 0.70, safe_auc            # honest chance-level performance


def test_our_pipeline_is_at_chance_when_labels_are_random(cohort):
    X, _, m, _ = cohort
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(12, int), np.ones(48, int)]
    y = rng.permutation(y)
    scores = np.full(len(y), np.nan)
    for sp in repeated_stratified_splits(y, 4, 1, seed=1):
        out = cv.run_fold(X, y, sp.train_idx, sp.test_idx, probe_to_gene=m, **CFG)
        scores[sp.test_idx] = out.predictions["score"].to_numpy()
    assert not np.isnan(scores).any()
    assert 0.25 < roc_auc_score(y, scores) < 0.75
