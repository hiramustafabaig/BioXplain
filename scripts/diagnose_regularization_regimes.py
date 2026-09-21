"""Exploratory, POST-HOC diagnostics for the regularisation sensitivity result (discovery cohort only).

Run after `run_regularization_sensitivity.py` found that C=1 is less stable than both smaller and larger C. Reports,
for the first folds of the sensitivity design: (1) solver convergence, (2) coefficient-profile shape, (3) similarity
of |coef| to a univariate mean-difference ranking (training data only), (4) pairwise rank / top-25 agreement between
all C values. Nothing here changes any pre-specified decision.

    python scripts/diagnose_regularization_regimes.py > results/sensitivity/<id>/regime_diagnostics.txt
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from bioxplain.data.discovery import load_discovery
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.validation.splits import repeated_stratified_splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 20260921

d = load_discovery(ROOT)
y = d.y.to_numpy()
conv_rows, rho, jac = [], {(a, b): [] for a in CS for b in CS}, {(a, b): [] for a in CS for b in CS}
for i, sp in enumerate(repeated_stratified_splits(y, 5, 2, SEED)):
    if i >= 6:
        break
    Xtr, ytr = d.X.iloc[sp.train_idx], y[sp.train_idx]
    Z = FoldPreprocessor(d.probe_to_gene).fit(Xtr, ytr).transform(Xtr).to_numpy()
    mean_diff = np.abs(Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0))
    coefs = {}
    for C in CS:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=1).fit(Z, ytr)
        a = np.abs(m.coef_.ravel())
        coefs[C] = a
        f, sgn = m.decision_function(Z), np.where(ytr == 1, 1, -1)
        conv_rows.append({"C": C, "n_iter": int(m.n_iter_[0]), "convergence_warning": any("converge" in str(x.message).lower() for x in w),
                          "coef_norm": float(np.linalg.norm(m.coef_)), "min_train_margin": float((sgn * f).min()),
                          "top25_over_median_abs_coef": float(np.sort(a)[-25] / np.median(a)),
                          "spearman_with_univariate_mean_difference": float(spearmanr(a, mean_diff).statistic)})
    top = {C: set(np.argsort(-coefs[C])[:25]) for C in CS}
    for a in CS:
        for b in CS:
            rho[(a, b)].append(spearmanr(coefs[a], coefs[b]).statistic)
            jac[(a, b)].append(len(top[a] & top[b]) / len(top[a] | top[b]))

pd.set_option("display.width", 250)
print("== solver / coefficient regime (mean over 6 training folds; L2 logistic regression, balanced, max_iter=5000)")
print(pd.DataFrame(conv_rows).groupby("C").agg(n_iter=("n_iter", "mean"), warnings=("convergence_warning", "sum"), coef_norm=("coef_norm", "mean"),
      min_train_margin=("min_train_margin", "mean"), top25_over_median=("top25_over_median_abs_coef", "mean"),
      rho_vs_mean_difference=("spearman_with_univariate_mean_difference", "mean")).round(3).to_string())
mat = lambda D: pd.DataFrame([[np.mean(D[(a, b)]) for b in CS] for a in CS], index=CS, columns=CS).round(2)
print("\n== mean Spearman correlation of |coef| between C settings (same training data)")
print(mat(rho).to_string())
print("\n== mean top-25 Jaccard between C settings")
print(mat(jac).to_string())
