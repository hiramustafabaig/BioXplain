"""Scale-free external replication statistics (Addendum A4). Everything here is invariant to monotone per-gene rescaling
of a cohort, so cohorts on different numeric scales are comparable.

Gene effect e = AUC - 0.5 (Mann-Whitney AUC, cancer > normal; average ranks for ties). Aligned external effect
r_g = sign(e_disc) * e_ext. C1 = Spearman(s, r); C2 = incremental association of stability s with r beyond |e_disc|
(rank regression coefficient and partial Spearman). Uncertainty comes from bootstrapping the EXTERNAL samples
(stratified); genes are correlated, so gene-level p-values are not used as evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bioxplain.evaluation.bootstrap import bootstrap_intervals
from bioxplain.evaluation.metrics import classification_metrics


def auc_effects(X: np.ndarray, y) -> np.ndarray:
    """Per-gene Mann-Whitney AUC (P[cancer > normal] + 0.5 P[tie]) for X (samples x genes)."""
    y = np.asarray(y).astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("both classes are required")
    ranks = stats.rankdata(np.asarray(X, dtype=float), axis=0)
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _rank01(v: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    return (r - 1) / (len(r) - 1) if len(r) > 1 else r


def c1_c2(s: np.ndarray, e_disc: np.ndarray, auc_ext: np.ndarray) -> dict:
    """C1: Spearman(s, r). C1_pos: same among genes with s > 0. C2: coefficient of rank(s) in rank(r) ~ rank(s) + rank(|e_disc|)
    (all ranks scaled to [0, 1]) and the partial Spearman of (s, r) given |e_disc|."""
    r = np.sign(e_disc) * (auc_ext - 0.5)
    c1 = stats.spearmanr(s, r).statistic
    pos = s > 0
    c1_pos = stats.spearmanr(s[pos], r[pos]).statistic if pos.sum() > 3 else np.nan
    Rs, Rr, Ra = _rank01(s), _rank01(r), _rank01(np.abs(e_disc))
    A = np.column_stack([np.ones_like(Rs), Rs, Ra])
    beta = np.linalg.lstsq(A, Rr, rcond=None)[0]
    res = lambda v: v - np.column_stack([np.ones_like(Ra), Ra]) @ np.linalg.lstsq(np.column_stack([np.ones_like(Ra), Ra]), v, rcond=None)[0]
    partial = float(np.corrcoef(res(Rs), res(Rr))[0, 1])
    return {"c1_spearman": float(c1), "c1_spearman_selected_only": float(c1_pos), "c2_rank_coef_stability": float(beta[1]),
            "c2_rank_coef_discovery_effect": float(beta[2]), "c2_partial_spearman": partial, "n_genes": int(len(s)), "n_selected": int(pos.sum())}


def bootstrap_c1_c2(s, e_disc, X_ext: np.ndarray, y_ext, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Stratified bootstrap over EXTERNAL samples; discovery quantities (s, e_disc) fixed."""
    y_ext = np.asarray(y_ext).astype(int)
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y_ext == 0), np.flatnonzero(y_ext == 1)
    rows = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        rows.append(c1_c2(s, e_disc, auc_effects(X_ext[idx], y_ext[idx])))
    return pd.DataFrame(rows)


def percentile_ci(values, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def direction_consistency(e_disc: np.ndarray, e_ext: np.ndarray) -> dict:
    """Fraction of genes whose external effect has the discovery direction, with an exact (Clopper-Pearson) interval.
    Genes are correlated, so the interval is descriptive. Genes with exactly zero external effect count as inconsistent."""
    agree = int((np.sign(e_disc) == np.sign(e_ext)).sum())
    n = len(e_disc)
    ci = stats.binomtest(agree, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"n": n, "n_consistent": agree, "fraction": agree / n, "ci_lo": float(ci.low), "ci_hi": float(ci.high)}


def hedges_g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-gene standardised mean difference (x = cancer, y = normal) with small-sample correction; within-cohort, scale-free."""
    n1, n0 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(axis=0, ddof=1) + (n0 - 1) * y.var(axis=0, ddof=1)) / (n1 + n0 - 2))
    d = np.divide(x.mean(axis=0) - y.mean(axis=0), sp, out=np.zeros(x.shape[1]), where=sp > 0)
    return d * (1 - 3 / (4 * (n1 + n0) - 9))


def within_cohort_z(X: np.ndarray) -> np.ndarray:
    """z-score each gene with the cohort's own (unlabeled) mean and SD: the transductive, label-free standardisation of A4."""
    sd = X.std(axis=0, ddof=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def signature_score(Z: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """mean z of frozen up-genes minus mean z of frozen down-genes (either group may be empty)."""
    score = np.zeros(Z.shape[0])
    if len(up):
        score += Z[:, up].mean(axis=1)
    if len(down):
        score -= Z[:, down].mean(axis=1)
    return score


def bootstrap_auc_ci(y, score, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    return bootstrap_intervals(y, score, (np.asarray(score) > np.median(score)).astype(int), n_boot, seed)["roc_auc"]


def frozen_lr_transfer(Xd: np.ndarray, yd, Xe: np.ndarray, ye, n_boot: int = 1000, seed: int = 0) -> dict:
    """LR (C=1, balanced) fitted on the FULL discovery cohort restricted to the frozen genes (discovery z-scoring), applied to the
    external samples after within-cohort z-scoring. Returns metrics with bootstrap intervals."""
    yd, ye = np.asarray(yd).astype(int), np.asarray(ye).astype(int)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed).fit(within_cohort_z(Xd), yd)
    Ze = within_cohort_z(Xe)
    score, pred = lr.predict_proba(Ze)[:, 1], lr.predict(Ze)
    ci = bootstrap_intervals(ye, score, pred, n_boot, seed)
    return {**classification_metrics(ye, score, pred), **{f"{k}_ci": v for k, v in ci.items()}, "n": int(len(ye)), "n_normal": int((ye == 0).sum())}


def residual_auc(y, score, covariate) -> float:
    """AUC of the part of `score` not linearly explained by `covariate` (used for the adipose-panel adjustment)."""
    y, score, cov = np.asarray(y), np.asarray(score, float), np.asarray(covariate, float)
    A = np.column_stack([np.ones_like(cov), cov])
    resid = score - A @ np.linalg.lstsq(A, score, rcond=None)[0]
    return float(roc_auc_score(y, resid))
