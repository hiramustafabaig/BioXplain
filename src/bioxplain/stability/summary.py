"""Turn per-run rankings into feature sets and stability tables."""
from __future__ import annotations

import pandas as pd

from bioxplain.stability.metrics import jaccard_stability, kuncheva_index, nogueira_stability


def top_k_sets(rankings: pd.DataFrame, k: int, run_cols=("repeat", "fold")) -> list[frozenset]:
    """One frozenset per run: the k highest-ranked genes among those with strictly positive importance (D13b).

    Runs are returned in deterministic order. A run with fewer than k positive-attribution genes (sparse models)
    yields a smaller set; zero-importance genes are never used as fillers. ``rankings`` must contain ``importance``.
    """
    if "importance" not in rankings.columns:
        raise ValueError("rankings need an 'importance' column to apply the zero-attribution rule")
    top = rankings[(rankings["rank"] <= k) & (rankings["importance"] > 0)]
    keys = sorted(rankings.groupby(list(run_cols)).groups.keys())
    groups = {key: frozenset(g["gene"]) for key, g in top.groupby(list(run_cols))}
    sets = [groups.get(key, frozenset()) for key in keys]
    if any(len(x) == 0 for x in sets):
        raise ValueError("a run has no positive-attribution gene; stability is undefined")
    return sets


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
                   "kuncheva": kuncheva_index(sets, n_features) if len({len(x) for x in sets}) == 1 else float("nan"),
                   "jaccard": jaccard_stability(sets),
                   "n_distinct_genes": len(frozenset().union(*sets)),
                   "mean_set_size": sum(len(x) for x in sets) / len(sets)}
            if n_features_alt:
                row["nogueira_alt_universe"] = nogueira_stability(sets, n_features_alt)
                row["n_features_alt"] = n_features_alt
            rows.append({**(extra or {}), **row})
    return pd.DataFrame(rows)
