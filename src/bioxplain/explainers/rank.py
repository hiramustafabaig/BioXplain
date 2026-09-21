"""Deterministic ranking of attributions. Rank 1 = largest ``importance``; ties broken by gene name.

Genes with importance <= 0 are ranked (so nothing is lost) but are never eligible for a top-k set (D13b); the
alphabetical tie-break therefore only orders genes whose selection is decided by the strictly-positive rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_features(genes, importance, signed=None) -> pd.DataFrame:
    genes = np.asarray(genes, dtype=object)
    importance = np.asarray(importance, dtype=float)
    if genes.shape != importance.shape:
        raise ValueError("genes and importance must have the same length")
    if not np.all(np.isfinite(importance)):
        raise ValueError("importance contains non-finite values")
    signed = importance if signed is None else np.asarray(signed, dtype=float)
    df = pd.DataFrame({"gene": genes, "importance": importance, "signed": signed})
    df = df.sort_values(["importance", "gene"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
