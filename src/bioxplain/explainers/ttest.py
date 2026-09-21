"""Univariate baseline: Welch t-statistic (cancer vs normal) computed on the TRAINING fold only (decision D16b).

Ranking statistic: |t|. Raw statistics are used only to rank; no p-value is interpreted as evidence, no multiple-testing
claim is made, and a top-k list from this baseline is not a validated discovery. The statistic is invariant to per-gene
scaling, so scaled and unscaled training matrices give identical rankings. Zero pooled variance: if the class means also
agree the gene gets t = 0 (never selectable); if they differ (perfect separation, t = +-infinity) the gene gets the finite cap
+-T_CAP so that it ranks first and ties are broken by gene name (cannot occur with continuous GC-RMA data; handled for correctness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bioxplain.explainers.rank import rank_features


T_CAP = 1e9


def welch_t(Z: np.ndarray, y) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y).astype(int)
    a, b = Z[y == 1], Z[y == 0]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each class needs at least two training samples for a Welch t-statistic")
    num = a.mean(axis=0) - b.mean(axis=0)
    den = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    t = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    separated = (den == 0) & (num != 0)
    return np.where(separated, np.sign(num) * T_CAP, t)


def ttest_attribution(Z, y, genes) -> pd.DataFrame:
    t = welch_t(Z, y)
    return rank_features(genes, np.abs(t), t)          # signed = t (positive = higher in cancer)
