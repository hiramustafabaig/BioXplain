"""Tissue-composition hypothesis (Addendum A6): adipocyte marker panel checks. A hypothesis, not an assumption."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_PANEL = ("ADIPOQ", "PLIN1", "FABP4")
EXTENDED_PANEL = ("LEP", "LPL", "CFD", "ADH1B", "CIDEC", "PLIN4", "CD36", "GPD1")


def panel_membership(frozen_lists: dict[str, list[str]], s_g: pd.Series, panel=PRIMARY_PANEL + EXTENDED_PANEL) -> pd.DataFrame:
    """For each panel gene: in universe?, selection frequency, percentile among universe genes, and membership of each frozen list."""
    pct = s_g.rank(pct=True, method="average")
    rows = []
    for g in panel:
        rows.append({"gene": g, "panel": "primary" if g in PRIMARY_PANEL else "extended", "in_universe": g in s_g.index,
                     "selection_frequency": float(s_g.get(g, np.nan)), "percentile_in_universe": float(pct.get(g, np.nan)),
                     **{f"in_{k}": g in v for k, v in frozen_lists.items()}})
    return pd.DataFrame(rows)


def panel_score(Z: np.ndarray, genes: list[str], panel) -> np.ndarray:
    """Mean within-cohort z of the panel genes present (Z is a within-cohort z-scored matrix; genes label its columns)."""
    idx = [genes.index(g) for g in panel if g in genes]
    if not idx:
        raise ValueError("no panel gene present")
    return Z[:, idx].mean(axis=1)
