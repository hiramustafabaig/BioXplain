"""Repeated stratified cross-validation splits.

GEO series order is class-blocked (GSE42568: 17 normals first), so splits are always shuffled and
stratified; the ordering of the input can therefore not influence the folds (tested).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def repeated_stratified_splits(y, n_splits: int, n_repeats: int, seed: int) -> Iterator[Split]:
    """Yield ``n_repeats x n_splits`` splits; repeat r uses ``random_state = seed + r`` (decision D22)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"expected a binary target, found classes {classes.tolist()}")
    if counts.min() < n_splits:
        raise ValueError(f"smallest class has {counts.min()} samples, fewer than n_splits={n_splits}")
    for r in range(n_repeats):
        rep_seed = seed + r
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep_seed)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            yield Split(repeat=r, fold=f, seed=rep_seed, train_idx=tr, test_idx=te)
