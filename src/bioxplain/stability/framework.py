"""Stability dimensions on top of the verified metric functions (Nogueira Phi, Kuncheva, Jaccard). No composite score.

Dimensions (never averaged into one number):
  resampling  : ONE pipeline (model x explainer) across the folds of a repeat (M = n_splits sets of different training data)
  model       : fixed explainer, different models, WITHIN each fold (same training data; isolates the model effect)
  explainer   : fixed model, different explainers, within each fold
  combined    : all primary pipelines within each fold
  baseline    : each primary pipeline vs the t-test baseline within each fold (M = 2)
  sensitivity : logistic-regression coefficient sets across the analytic-sensitivity C settings within each fold
Unit of comparison = one CV repeat: resampling uses that repeat's folds as the M sets; within-fold dimensions are averaged
over the folds of the repeat. Observed and permutation-null values are computed by the SAME function.
Set = genes with strictly positive attribution in the top-k (D13b); sizes may be < k for sparse models, so Phi
(variable size) is the comparable statistic; Kuncheva is NaN unless all sets have equal size; Jaccard is always given.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from bioxplain.stability.metrics import jaccard_stability, kuncheva_index, nogueira_stability


def pipeline_sets(rankings: pd.DataFrame, k: int) -> dict:
    """{(pipeline, repeat, fold): frozenset of top-k genes}; only strictly positive attributions are stored upstream."""
    top = rankings[rankings["rank"] <= k]
    return {key: frozenset(g["gene"]) for key, g in top.groupby(["pipeline", "repeat", "fold"])}


def _stats(sets: list[frozenset], d: int) -> dict:
    sets = [s for s in sets if len(s)]
    if len(sets) < 2:
        return {"phi": np.nan, "jaccard": np.nan, "kuncheva": np.nan, "mean_set_size": np.nan}
    sizes = {len(s) for s in sets}
    return {"phi": nogueira_stability(sets, d), "jaccard": jaccard_stability(sets),
            "kuncheva": kuncheva_index(sets, d) if len(sizes) == 1 else np.nan, "mean_set_size": float(np.mean([len(s) for s in sets]))}


def group_definitions(pipelines: pd.DataFrame) -> list[tuple[str, str, list[str]]]:
    """(dimension, group label, member pipelines) for every within-fold comparison."""
    prim = pipelines[pipelines.role == "primary"]
    groups = []
    for e, g in prim.groupby("explainer"):
        if len(g) >= 2:
            groups.append(("model", f"explainer={e}", g.pipeline.tolist()))
    for m, g in prim.groupby("model"):
        if len(g) >= 2:
            groups.append(("explainer", f"model={m}", g.pipeline.tolist()))
    if len(prim) >= 2:
        groups.append(("combined", "all_primary", prim.pipeline.tolist()))
    if (pipelines.role == "baseline").any():
        t = pipelines[pipelines.role == "baseline"].pipeline.iloc[0]
        groups += [("baseline", f"{p}_vs_ttest", [p, t]) for p in prim.pipeline]
    sens = pipelines[(pipelines.model == "logreg") & (pipelines.explainer == "coef")]
    if len(sens) >= 2:
        groups.append(("sensitivity", "logreg_coef_across_C", sens.pipeline.tolist()))
    return groups


def repeat_level_stability(rankings: pd.DataFrame, pipelines: pd.DataFrame, ks, d: int) -> pd.DataFrame:
    """Rows: dimension, group, k, repeat, phi, jaccard, kuncheva, mean_set_size, n_units (folds or sets used)."""
    rows = []
    for k in ks:
        sets = pipeline_sets(rankings, k)
        by_rep: dict = {}
        for (p, rep, fold), s in sets.items():
            by_rep.setdefault((p, rep), []).append(((fold), s))
        for (p, rep), lst in sorted(by_rep.items()):
            rows.append({"dimension": "resampling", "group": p, "k": k, "repeat": rep, **_stats([s for _, s in sorted(lst, key=lambda x: x[0])], d), "n_units": len(lst)})
        folds = sorted({(rep, fold) for (_, rep, fold) in sets})
        for dim, label, members in group_definitions(pipelines):
            per_rep: dict = {}
            for rep, fold in folds:
                mem = [sets[(m, rep, fold)] for m in members if (m, rep, fold) in sets]
                st = _stats(mem, d)
                per_rep.setdefault(rep, []).append(st)
            for rep, lst in per_rep.items():
                df = pd.DataFrame(lst)
                rows.append({"dimension": dim, "group": label, "k": k, "repeat": rep, "phi": df.phi.mean(), "jaccard": df.jaccard.mean(),
                             "kuncheva": df.kuncheva.mean(), "mean_set_size": df.mean_set_size.mean(), "n_units": int(df.phi.notna().sum())})
    return pd.DataFrame(rows)


def summarize_repeats(repeat_level: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / min / max / sd over repeats. Repeats re-use the same samples, so these describe spread, not a CI."""
    g = repeat_level.groupby(["dimension", "group", "k"], sort=False)
    out = g.agg(n_repeats=("repeat", "nunique"), phi_mean=("phi", "mean"), phi_median=("phi", "median"), phi_min=("phi", "min"), phi_max=("phi", "max"),
                phi_sd=("phi", "std"), jaccard_mean=("jaccard", "mean"), jaccard_median=("jaccard", "median"), kuncheva_mean=("kuncheva", "mean"),
                mean_set_size=("mean_set_size", "mean")).reset_index()
    return out


def pooled_resampling(rankings: pd.DataFrame, pipelines: pd.DataFrame, ks, d: int) -> pd.DataFrame:
    """Per pipeline and k: Phi/Kuncheva/Jaccard over ALL folds of ALL repeats (M = n_folds_total), distinct genes, frequency counts."""
    rows = []
    for k in ks:
        sets = pipeline_sets(rankings, k)
        for p in pipelines.pipeline:
            sl = [s for (pp, _, _), s in sorted(sets.items()) if pp == p]
            if len(sl) < 2:
                continue
            freq = pd.Series([g for s in sl for g in s]).value_counts() / len(sl)
            rows.append({"pipeline": p, "k": k, "n_sets": len(sl), **_stats(sl, d), "n_distinct_genes": len(freq),
                         "n_genes_freq_ge_0.5": int((freq >= 0.5).sum()), "n_genes_freq_ge_0.8": int((freq >= 0.8).sum()), "max_selection_frequency": float(freq.max())})
    return pd.DataFrame(rows)


def pairwise_jaccard_matrix(rankings: pd.DataFrame, pipelines: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mean within-fold Jaccard between every pair of pipelines (primary + baseline), for the heatmap."""
    sets = pipeline_sets(rankings, k)
    names = pipelines.pipeline.tolist()
    folds = sorted({(rep, fold) for (_, rep, fold) in sets})
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for a, b in itertools.combinations_with_replacement(names, 2):
        vals = [len(sets[(a, r, f)] & sets[(b, r, f)]) / len(sets[(a, r, f)] | sets[(b, r, f)]) for r, f in folds if (a, r, f) in sets and (b, r, f) in sets]
        M.loc[a, b] = M.loc[b, a] = float(np.mean(vals)) if vals else np.nan
    return M
