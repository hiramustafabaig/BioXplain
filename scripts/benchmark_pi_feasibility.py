"""Feasibility benchmark for exact permutation importance (decision D12b). Timing/structure only, no outcomes.

Fits RF / XGBoost / LinearSVC on ONE training fold of GSE42568 (discovery cohort only) and reports how many features
each model uses, the cost of one re-prediction, and the projected cost of naive exact permutation importance.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from bioxplain.data.discovery import load_discovery
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.validation.splits import repeated_stratified_splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = 10   # permutations per feature assumed for the projection

d = load_discovery(ROOT)
y = d.y.to_numpy()
sp = next(iter(repeated_stratified_splits(y, 5, 1, 20260921)))
Xtr, ytr = d.X.iloc[sp.train_idx], y[sp.train_idx]
Z = FoldPreprocessor(d.probe_to_gene, scale=False).fit(Xtr, ytr).transform(Xtr).to_numpy()
p = Z.shape[1]
print(f"training matrix {Z.shape}; normals {(ytr == 0).sum()}")


def timeit(fn, reps=20):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


t = time.perf_counter()
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced_subsample", random_state=1, n_jobs=4).fit(Z, ytr)
fit_rf = time.perf_counter() - t
used_rf = len(np.unique(np.concatenate([e.tree_.feature[e.tree_.feature >= 0] for e in rf.estimators_])))
pred_rf = timeit(lambda: rf.predict_proba(Z))
print(f"RF  : fit {fit_rf:.1f}s, features used {used_rf}/{p} ({used_rf / p:.1%}), predict {pred_rf * 1e3:.0f} ms, "
      f"naive exact PI R={R}: used-only {used_rf * R * pred_rf:,.0f}s / all {p * R * pred_rf:,.0f}s per fold")

spw = (ytr == 0).sum() / (ytr == 1).sum()
t = time.perf_counter()
xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                    scale_pos_weight=spw, random_state=1, n_jobs=4, verbosity=0).fit(Z, ytr)
fit_x = time.perf_counter() - t
used_x = len(xgb.get_booster().get_score(importance_type="weight"))
pred_x = timeit(lambda: xgb.predict_proba(Z))
print(f"XGB : fit {fit_x:.1f}s, features used {used_x}/{p}, predict {pred_x * 1e3:.0f} ms, "
      f"naive exact PI R={R}: used-only {used_x * R * pred_x:,.0f}s / all {p * R * pred_x:,.0f}s per fold")

Zs = FoldPreprocessor(d.probe_to_gene, scale=True).fit(Xtr, ytr).transform(Xtr).to_numpy()
t = time.perf_counter()
svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, dual=True, random_state=1).fit(Zs, ytr)
print(f"SVM : fit {time.perf_counter() - t:.2f}s, nonzero coefficients {(svm.coef_ != 0).sum()}/{p} (dense linear model)")
