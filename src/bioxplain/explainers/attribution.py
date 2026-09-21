"""Explanations/attributions -> deterministic feature rankings.

Different explainers measure different quantities (see docs/methodology_decisions.md D12). Each returns a
DataFrame with ``gene, importance, signed, rank`` where rank 1 = largest ``importance``; ties are broken by
gene name so rankings are deterministic.
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


def coefficient_attribution(model, genes) -> pd.DataFrame:
    """Logistic-regression coefficients on z-scored genes.

    ``importance`` = |coef| (size of the model's linear weight given all other genes, on a common scale);
    ``signed`` = coef (positive = towards the cancer class). This is a property of the fitted model, not a
    causal or univariate effect.
    """
    coef = np.asarray(model.coef_, dtype=float).ravel()
    if len(coef) != len(genes):
        raise ValueError("number of coefficients does not match number of genes")
    return rank_features(genes, np.abs(coef), coef)


EXPLAINERS = {"coef": coefficient_attribution}
