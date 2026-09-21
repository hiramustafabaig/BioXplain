"""Classification metrics for a heavily imbalanced tumour-vs-normal task (positive class = cancer, 1).

Why several metrics (decision D21 / user requirement G):
* ROC-AUC: threshold-free ranking quality; insensitive to prevalence.
* PR-AUC (average precision) for cancer AND for normal: with 86% cancer, cancer-AP is near 1 even for weak
  models, so the normal-class AP is the informative one.
* Balanced accuracy / sensitivity (cancer recall) / specificity (normal recall): accuracy alone would reward
  predicting "cancer" always.
* F1 per class and the raw confusion matrix.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_score, y_pred) -> dict[str, float]:
    y_true, y_score, y_pred = (np.asarray(a) for a in (y_true, y_score, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap_cancer": float(average_precision_score(y_true, y_score)),
        "ap_normal": float(average_precision_score(1 - y_true, 1 - y_score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_cancer": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(
    y_true, y_score, metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling within each class so both classes are always present."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y_true == 0), np.flatnonzero(y_true == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("both classes are required for a bootstrap CI")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))])
        vals[b] = metric(y_true[idx], y_score[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
