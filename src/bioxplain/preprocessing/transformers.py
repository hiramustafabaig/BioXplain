"""Fold-safe preprocessing.

Every class here LEARNS something in ``fit`` (a floor, a set of kept probes, a probe per gene, scaling
statistics). ``fit`` must only ever be called with training-fold samples; the validation/test fold is
passed to ``transform`` only. See docs/methodology_decisions.md D6-D8 and tests/test_leakage.py.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd


class FloorDetectionFilter:
    """Drop probes that are (almost) never detected above the array floor (decision D7).

    Learned in ``fit``: the floor (minimum training value, accepted only when at least
    ``min_floor_fraction`` of training values sit on it) and the set of probes detected in at least
    ``ceil(min_detect_frac_of_minority * n_minority)`` training samples. Only the *count* of the minority
    class is used, never any expression-label relationship.
    """

    def __init__(self, min_detect_frac_of_minority: float = 0.5, min_floor_fraction: float = 0.01, tol: float = 1e-9):
        self.min_detect_frac_of_minority = min_detect_frac_of_minority
        self.min_floor_fraction = min_floor_fraction
        self.tol = tol

    def fit(self, X: pd.DataFrame, y) -> "FloorDetectionFilter":
        y = np.asarray(y)
        values = X.to_numpy(dtype=float)
        floor = float(values.min())
        self.floor_fraction_ = float((values <= floor + self.tol).mean())
        self.n_input_ = X.shape[1]
        if self.floor_fraction_ < self.min_floor_fraction:
            warnings.warn(
                f"no array floor detected (only {self.floor_fraction_:.4f} of values at the minimum); "
                "expression filter not applied",
                stacklevel=2,
            )
            self.floor_ = None
            self.threshold_ = 0
            self.kept_ = X.columns
        else:
            n_minority = int(np.bincount(y.astype(int)).min())
            self.floor_ = floor
            self.threshold_ = math.ceil(self.min_detect_frac_of_minority * n_minority)
            detected = (values > floor + self.tol).sum(axis=0)
            self.kept_ = X.columns[detected >= self.threshold_]
        self.n_kept_ = len(self.kept_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.kept_]


class ProbeGeneCollapser:
    """Represent each gene by ONE probe chosen from the training data (decision D6).

    ``rule='max_mean'``: highest training mean (primary). ``rule='max_variance'``: highest training variance
    (sensitivity alternative). Ties are broken by probe ID so the choice is deterministic. Probes without a
    mapping are dropped.
    """

    RULES = ("max_mean", "max_variance")

    def __init__(self, probe_to_gene: pd.Series, rule: str = "max_mean"):
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}, got {rule!r}")
        self.probe_to_gene = probe_to_gene
        self.rule = rule

    def fit(self, X: pd.DataFrame, y=None) -> "ProbeGeneCollapser":
        probes = [p for p in X.columns if p in self.probe_to_gene.index]
        sub = X.loc[:, probes]
        stat = sub.mean(axis=0) if self.rule == "max_mean" else sub.var(axis=0, ddof=0)
        table = pd.DataFrame(
            {"probe": probes, "gene": self.probe_to_gene.loc[probes].to_numpy(), "stat": stat.to_numpy()}
        )
        table = table.sort_values(["gene", "stat", "probe"], ascending=[True, False, True], kind="mergesort")
        self.selected_ = table.drop_duplicates("gene").set_index("gene")["probe"]  # gene -> probe
        self.n_probes_input_ = len(probes)
        return self

    @property
    def genes_(self) -> pd.Index:
        return self.selected_.index

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.loc[:, self.selected_.to_numpy()].copy()
        out.columns = self.selected_.index
        return out


class FoldPreprocessor:
    """floor-detection filter -> probe-to-gene collapse -> optional z-scoring, all fitted on training data."""

    def __init__(
        self,
        probe_to_gene: pd.Series,
        min_detect_frac_of_minority: float = 0.5,
        collapse_rule: str = "max_mean",
        scale: bool = True,
    ):
        self.probe_to_gene = probe_to_gene
        self.min_detect_frac_of_minority = min_detect_frac_of_minority
        self.collapse_rule = collapse_rule
        self.scale = scale

    def fit(self, X_train: pd.DataFrame, y_train) -> "FoldPreprocessor":
        self.filter_ = FloorDetectionFilter(self.min_detect_frac_of_minority).fit(X_train, y_train)
        Xf = self.filter_.transform(X_train)
        self.collapser_ = ProbeGeneCollapser(self.probe_to_gene, self.collapse_rule).fit(Xf)
        Xg = self.collapser_.transform(Xf)                       # scaling statistics are ALWAYS learned (training rows only)
        self.mean_ = Xg.mean(axis=0)
        std = Xg.std(axis=0, ddof=0)
        self.std_ = std.where(std > 0, 1.0)
        self.genes_ = self.collapser_.genes_
        return self

    def transform(self, X: pd.DataFrame, scale: bool | None = None) -> pd.DataFrame:
        """Filter -> collapse -> optional z-scoring with TRAINING statistics; ``scale=None`` uses the constructor setting."""
        Xg = self.collapser_.transform(self.filter_.transform(X))
        if self.scale if scale is None else scale:
            Xg = (Xg - self.mean_) / self.std_
        return Xg

    def fit_transform(self, X_train: pd.DataFrame, y_train) -> pd.DataFrame:
        return self.fit(X_train, y_train).transform(X_train)

    def diagnostics(self) -> dict:
        return {
            "floor": self.filter_.floor_,
            "floor_fraction": self.filter_.floor_fraction_,
            "detect_threshold_samples": self.filter_.threshold_,
            "n_probes_input": self.filter_.n_input_,
            "n_probes_after_filter": self.filter_.n_kept_,
            "n_genes": len(self.genes_),
        }
