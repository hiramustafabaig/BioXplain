"""Turn per-run rankings into feature sets and stability tables."""
from __future__ import annotations

import pandas as pd

from bioxplain.stability.metrics import jaccard_stability, kuncheva_index, nogueira_stability


def top_k_sets(rankings: pd.DataFrame, k: int, run_cols=("repeat", "fold")) -> list[frozenset]:
    """One frozenset of the top-k genes per run, in deterministic run order."""
    if rankings.groupby(list(run_cols))["rank"].max().min() < k:
        raise ValueError(f"stored rankings hold fewer than k={k} genes for some run; increase store_top")
    top = rankings[rankings["rank"] <= k]
    return [frozenset(g["gene"]) for _, g in sorted(top.groupby(list(run_cols)), key=lambda kv: kv[0])]


def stability_summary(
    rankings: pd.DataFrame, ks, n_features: int, n_features_alt: int | None = None,
    run_cols=("repeat", "fold"), extra: dict | None = None,
) -> pd.DataFrame:
    """Nogueira / Kuncheva / Jaccard at each k over all runs, plus per-repeat values.

    ``n_features_alt`` (optional) is a second universe size used only to show that the chance correction is
    insensitive to how the universe is defined.
    """
    rows = []
    scopes = [("all_runs", rankings)] + [
        (f"repeat_{r}", g) for r, g in sorted(rankings.groupby("repeat"), key=lambda kv: kv[0])
    ]
    for scope, sub in scopes:
        for k in ks:
            sets = top_k_sets(sub, k, run_cols)
            row = {"scope": scope, "k": k, "n_sets": len(sets), "n_features": n_features,
                   "nogueira": nogueira_stability(sets, n_features),
                   "kuncheva": kuncheva_index(sets, n_features),
                   "jaccard": jaccard_stability(sets),
                   "n_distinct_genes": len(frozenset().union(*sets))}
            if n_features_alt:
                row["nogueira_alt_universe"] = nogueira_stability(sets, n_features_alt)
                row["n_features_alt"] = n_features_alt
            rows.append({**(extra or {}), **row})
    return pd.DataFrame(rows)
