"""Transparent consensus / feature summary (decision D16). No weighted 'BioXplain score'.

For every gene that appears in any primary top-k set the following quantities are recorded (k = headline 25, also 10 and 50):
  selection_frequency_<pipeline>  fraction of folds in which the gene is in that pipeline's top-k
  mean_selection_frequency        mean of those frequencies over the primary pipelines (0 where never selected)
  n_pipelines_ge50 / n_models_ge50 / n_explainers_ge50
                                  number of primary pipelines (and distinct models / explainer families among them) whose own
                                  selection frequency is >= 0.5
  median_rank / best_rank         over all stored (pipeline, fold) rows in which the gene was ranked <= k
  frac_folds_signed_positive      share of signed rows (coef, SHAP, t-test; not permutation importance) with a positive sign
  ttest_frequency                 selection frequency of the t-test baseline
Pre-specified consensus rule: n_pipelines_ge50 >= ceil(n_primary_pipelines / 2) at k = 25.
Pre-specified frozen lists: S_cons (consensus rule), S_top25 (25 highest mean_selection_frequency at k=25),
S_top50 (50 highest at k=50); ties by mean rank then gene name. Separate discovery-stability criteria are never merged into a score.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HEADLINE_K = 25


def gene_table(rankings: pd.DataFrame, pipelines: pd.DataFrame, k: int = HEADLINE_K) -> pd.DataFrame:
    prim = pipelines[pipelines.role == "primary"]
    n_folds = rankings.groupby(["repeat", "fold"]).ngroups
    top = rankings[rankings["rank"] <= k]
    counts = top.groupby(["pipeline", "gene"]).size().unstack(0).reindex(columns=pipelines.pipeline).fillna(0) / n_folds
    freq = counts.copy()
    prim_freq = freq[prim.pipeline]
    out = pd.DataFrame(index=freq.index)
    out["mean_selection_frequency"] = prim_freq.mean(axis=1)
    ge50 = prim_freq >= 0.5
    out["n_pipelines_ge50"] = ge50.sum(axis=1)
    model_of = prim.set_index("pipeline")["model"]; expl_of = prim.set_index("pipeline")["explainer"]
    out["n_models_ge50"] = ge50.apply(lambda r: model_of[r[r].index].nunique() if r.any() else 0, axis=1)
    out["n_explainers_ge50"] = ge50.apply(lambda r: expl_of[r[r].index].nunique() if r.any() else 0, axis=1)
    prim_top = top[top.pipeline.isin(prim.pipeline)]
    out["median_rank"] = prim_top.groupby("gene")["rank"].median()
    out["best_rank"] = prim_top.groupby("gene")["rank"].min()
    out["mean_rank"] = prim_top.groupby("gene")["rank"].mean()
    signed = top[top.explainer.isin(["coef", "shap", "ttest"])]
    out["frac_signed_positive"] = signed.groupby("gene")["signed"].apply(lambda s: float((s > 0).mean()))
    out["ttest_frequency"] = freq["ttest|ttest"] if "ttest|ttest" in freq else np.nan
    for p in pipelines.pipeline:
        out[f"freq:{p}"] = freq[p]
    out["consensus"] = out["n_pipelines_ge50"] >= math.ceil(len(prim) / 2)
    out["k"] = k
    return out.reset_index(names="gene").sort_values(["mean_selection_frequency", "mean_rank", "gene"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)


def frozen_lists(rankings: pd.DataFrame, pipelines: pd.DataFrame) -> dict[str, list[str]]:
    g25, g50 = gene_table(rankings, pipelines, 25), gene_table(rankings, pipelines, 50)
    return {"S_cons": g25.loc[g25.consensus, "gene"].tolist(), "S_top25": g25["gene"].head(25).tolist(), "S_top50": g50["gene"].head(50).tolist()}


def universe_frequency(rankings: pd.DataFrame, pipelines: pd.DataFrame, universe: pd.Index, k: int = HEADLINE_K) -> pd.Series:
    """Mean selection frequency at k over primary pipelines for EVERY gene of the eligible universe (0 for never selected):
    the continuous internal-stability quantity used by the external-replication analysis (D17 C1/C2)."""
    t = gene_table(rankings, pipelines, k).set_index("gene")["mean_selection_frequency"]
    return t.reindex(universe).fillna(0.0)
