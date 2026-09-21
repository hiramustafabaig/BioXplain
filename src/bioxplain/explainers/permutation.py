"""Exact permutation importance with exactness-preserving shortcuts (decision D12b).

Definition (fixed before any result):
  data       = the TRAINING fold (model reliance on the data it was fitted to)
  loss       = class-balanced Brier score  L = 0.5 * [ mean_{y=1}(1-p)^2 + mean_{y=0} p^2 ],
               p = P(cancer) (LR, RF, XGBoost) or sigmoid(margin) (linear SVM)
  importance_j = mean over R permutations of  L(gene j permuted) - L(original)
  the same R permutation vectors of the training rows are used for every gene of a fold.
Shortcuts (all exact, verified against ``permutation_importance_bruteforce`` in tests):
  * linear models  : permuting gene j only changes the margin by w_j * (x_perm - x)  -> vectorised over all genes
  * tree ensembles : a gene the fitted model never splits on has exactly zero importance -> only used genes are evaluated
  * random forest  : only the trees that split on gene j are re-evaluated
No gene is dropped for being 'probably unimportant'; there is no top-N pre-filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from bioxplain.explainers.rank import rank_features
from bioxplain.models.factory import model_support

TOLERANCE = 1e-12          # documented equivalence tolerance between shortcuts and brute force (absolute, on importance)


def balanced_brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class-balanced Brier score along axis 0 (works for a vector p or a matrix with one column per scenario)."""
    y = np.asarray(y)
    pos = y == 1
    return 0.5 * (((1.0 - p[pos]) ** 2).mean(axis=0) + (p[~pos] ** 2).mean(axis=0))


def make_permutations(n: int, n_perm: int, seed_seq: np.random.SeedSequence) -> list[np.ndarray]:
    rng = np.random.default_rng(seed_seq)
    return [rng.permutation(n) for _ in range(n_perm)]


def _linear_importance(coef: np.ndarray, intercept: float, Z: np.ndarray, y: np.ndarray, perms) -> np.ndarray:
    f = Z @ coef + intercept
    base = balanced_brier(expit(f), y)
    acc = np.zeros(Z.shape[1])
    for perm in perms:
        F = f[:, None] + (Z[perm] - Z) * coef[None, :]           # margin after permuting each single gene
        acc += balanced_brier(expit(F), y)
    return acc / len(perms) - base


def _xgb_importance(model, Z, y, perms) -> np.ndarray:
    """Only genes the boosted trees split on can matter. Predictions are made from a SPARSE matrix that stores just those genes
    (unused columns are never read by any tree), which is bit-identical to predicting on the full dense matrix (verified in tests and
    on real folds) and 6-12x faster than densely re-predicting a 20k-wide matrix for every gene and permutation."""
    import scipy.sparse as sp

    n, p = Z.shape
    support = model_support(model, "xgb", p)
    imp = np.zeros(p)
    if len(support) == 0:
        return imp
    booster = model.get_booster()
    R = np.ascontiguousarray(Z[:, support], dtype=np.float32)               # XGBoost works in float32 internally
    indices = np.tile(support, n).astype(np.int32)
    indptr = np.arange(0, n * len(support) + 1, len(support))

    def prob(Rm: np.ndarray) -> np.ndarray:
        return booster.inplace_predict(sp.csr_matrix((Rm.ravel(), indices, indptr), shape=(n, p)))

    base = balanced_brier(prob(R), y)
    for c, j in enumerate(support):
        col = R[:, c].copy()
        acc = 0.0
        for perm in perms:
            R[:, c] = col[perm]
            acc += balanced_brier(prob(R), y)
        R[:, c] = col
        imp[j] = acc / len(perms) - base
    return imp


def _rf_importance(model, Z, y, perms) -> np.ndarray:
    Z32 = np.ascontiguousarray(Z, dtype=np.float32)               # scikit-learn trees see float32
    trees = model.estimators_
    nt = len(trees)
    P = np.stack([t.predict_proba(Z32, check_input=False)[:, 1] for t in trees])      # (trees, n) class-1 probability
    total = P.sum(axis=0)
    base = balanced_brier(total / nt, y)
    by_gene: dict[int, list[int]] = {}
    for t, tree in enumerate(trees):
        for j in np.unique(tree.tree_.feature[tree.tree_.feature >= 0]):
            by_gene.setdefault(int(j), []).append(t)
    imp = np.zeros(Z.shape[1])
    for j, tl in by_gene.items():
        col = Z32[:, j].copy()
        old = P[tl].sum(axis=0)
        acc = 0.0
        for perm in perms:
            Z32[:, j] = col[perm]
            new = np.sum([trees[t].predict_proba(Z32, check_input=False)[:, 1] for t in tl], axis=0)
            acc += balanced_brier((total - old + new) / nt, y)
        Z32[:, j] = col
        imp[j] = acc / len(perms) - base
    return imp


def permutation_importance(model, name: str, Z: np.ndarray, y, perms) -> np.ndarray:
    """Importance of every gene (column of Z), exact; zeros for genes the model cannot use."""
    Z = np.array(Z, dtype=float, order="C")                       # private copy (in-place column edits are restored)
    y = np.asarray(y).astype(int)
    if name in ("logreg", "svm"):
        return _linear_importance(np.asarray(model.coef_, float).ravel(), float(np.ravel(model.intercept_)[0]), Z, y, perms)
    if name == "xgb":
        return _xgb_importance(model, Z, y, perms)
    if name == "rf":
        return _rf_importance(model, Z, y, perms)
    raise ValueError(f"unknown model {name!r}")


def permutation_importance_bruteforce(predict_p, Z: np.ndarray, y, perms, features=None) -> np.ndarray:
    """Reference implementation: permute one column at a time and re-predict with the model. Slow; used in tests."""
    Z = np.array(Z, dtype=float)
    y = np.asarray(y).astype(int)
    base = balanced_brier(np.asarray(predict_p(Z)), y)
    features = range(Z.shape[1]) if features is None else features
    imp = np.zeros(Z.shape[1])
    for j in features:
        acc = 0.0
        for perm in perms:
            Zp = Z.copy()
            Zp[:, j] = Z[perm, j]
            acc += balanced_brier(np.asarray(predict_p(Zp)), y)
        imp[j] = acc / len(perms) - base
    return imp


def probability_function(model, name: str):
    """p(cancer) used by the loss: probability for LR/RF/XGBoost, sigmoid(margin) for the SVM."""
    if name == "svm":
        return lambda Z: expit(model.decision_function(Z))
    return lambda Z: model.predict_proba(Z)[:, 1]


def permutation_attribution(model, name: str, Z, y, genes, seed_seq: np.random.SeedSequence, n_perm: int = 10) -> pd.DataFrame:
    Z = np.asarray(Z, dtype=float)
    perms = make_permutations(Z.shape[0], n_perm, seed_seq)
    raw = permutation_importance(model, name, Z, y, perms)
    # numerical floor: differences of two nearly equal losses carry ~1e-17 noise; anything <= TOLERANCE counts as no importance,
    # so it can never become an 'eligible positive' gene (D13b). The raw signed value is kept in `signed`.
    return rank_features(genes, np.where(raw > TOLERANCE, raw, 0.0), raw)
