"""Vectorised stratified bootstrap for ROC-AUC, average precision and balanced accuracy.

Statistical definition (unchanged from ``metrics.bootstrap_ci``): resample WITHIN each class with replacement,
keeping the class sizes, evaluate the metric on every resample, report the percentile interval.

Speed-up: instead of materialising each resample and calling scikit-learn, a resample is represented by its
multiplicity vector w (w_i = number of times sample i was drawn). All three metrics are functions of the
multiset, so they are evaluated for all resamples at once with matrix products. The resamples themselves are
drawn with the SAME generator calls in the SAME order as ``bootstrap_ci`` (class 0 draw, then class 1 draw, per
resample), so for a given seed both implementations use identical resamples and agree to floating-point rounding
(verified in tests/test_bootstrap.py and scripts/benchmark_metrics.py).

Tie handling matches scikit-learn: AUC counts tied (positive, negative) pairs as 1/2; average precision is
computed over the distinct score thresholds (step-wise, no interpolation).
"""
from __future__ import annotations

import numpy as np


def stratified_bootstrap_weights(y, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) integer multiplicities of within-class bootstrap resamples (same RNG order as bootstrap_ci)."""
    y = np.asarray(y)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("both classes are required for a bootstrap CI")
    rng = np.random.default_rng(seed)
    n = len(y)
    w = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        d0 = rng.choice(idx0, len(idx0))
        d1 = rng.choice(idx1, len(idx1))
        w[b] = np.bincount(d0, minlength=n) + np.bincount(d1, minlength=n)
    return w


def weighted_roc_auc(y, score, w: np.ndarray) -> np.ndarray:
    """ROC-AUC of every resample in ``w``; ties count 1/2 (Mann-Whitney)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    pos, neg = y == 1, y == 0
    sp, sn = score[pos][:, None], score[neg][None, :]
    conc = (sp > sn).astype(float) + 0.5 * (sp == sn)                      # (P, N) pair concordance
    wp, wn = w[:, pos].astype(float), w[:, neg].astype(float)
    return np.einsum("bp,pn,bn->b", wp, conc, wn) / (wp.sum(1) * wn.sum(1))


def weighted_average_precision(is_positive, score, w: np.ndarray) -> np.ndarray:
    """Average precision (scikit-learn definition) of every resample; ``is_positive`` marks the positive class."""
    is_positive = np.asarray(is_positive, dtype=bool)
    score = np.asarray(score, dtype=float)
    uniq = np.unique(score)[::-1]                                          # distinct thresholds, descending
    group = np.searchsorted(-uniq, -score)                                 # threshold index of each sample
    onehot = np.zeros((len(score), len(uniq)))
    onehot[np.arange(len(score)), group] = 1.0
    wf = w.astype(float)
    tp_step = (wf * is_positive) @ onehot                                  # positives added at each threshold
    fp_step = (wf * ~is_positive) @ onehot
    tp, fp = np.cumsum(tp_step, axis=1), np.cumsum(fp_step, axis=1)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    n_pos = (wf * is_positive).sum(1, keepdims=True)
    return ((tp_step / n_pos) * precision).sum(1)                          # sum_t (R_t - R_{t-1}) * P_t


def weighted_balanced_accuracy(y, pred, w: np.ndarray) -> np.ndarray:
    """Balanced accuracy of every resample for FIXED predicted labels ``pred`` (mean of per-class recall)."""
    y, pred = np.asarray(y), np.asarray(pred)
    wf = w.astype(float)
    tp = wf @ ((y == 1) & (pred == 1)).astype(float)
    tn = wf @ ((y == 0) & (pred == 0)).astype(float)
    return 0.5 * (tp / wf[:, y == 1].sum(1) + tn / wf[:, y == 0].sum(1))


def bootstrap_intervals(y, score, pred, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CIs for roc_auc, ap_normal (normal = positive, score -> 1 - score) and balanced_accuracy."""
    y, score, pred = np.asarray(y), np.asarray(score, dtype=float), np.asarray(pred)
    w = stratified_bootstrap_weights(y, n_boot, seed)
    draws = {
        "roc_auc": weighted_roc_auc(y, score, w),
        "ap_normal": weighted_average_precision(y == 0, 1 - score, w),
        "balanced_accuracy": weighted_balanced_accuracy(y, pred, w),
    }
    q = [alpha / 2, 1 - alpha / 2]
    return {k: (float(np.quantile(v, q[0])), float(np.quantile(v, q[1]))) for k, v in draws.items()}
