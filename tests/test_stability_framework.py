import math

import numpy as np
import pandas as pd
import pytest

from bioxplain.consensus import frozen_lists, gene_table, universe_frequency
from bioxplain.stability.framework import (group_definitions, pairwise_jaccard_matrix, pooled_resampling, repeat_level_stability, summarize_repeats)
from bioxplain.stability.metrics import jaccard_stability, nogueira_stability

PIPES = pd.DataFrame([
    {"pipeline": "a|coef", "config_id": "a", "model": "a", "explainer": "coef", "role": "primary"},
    {"pipeline": "a|shap", "config_id": "a", "model": "a", "explainer": "shap", "role": "primary"},
    {"pipeline": "b|shap", "config_id": "b", "model": "b", "explainer": "shap", "role": "primary"},
    {"pipeline": "ttest|ttest", "config_id": "ttest", "model": "none", "explainer": "ttest", "role": "baseline"},
])
D = 100


def close(series, value):
    return bool(np.allclose(np.asarray(series, dtype=float), value))


def make_rankings(gene_sets: dict, repeats=2, folds=3):
    """gene_sets[pipeline] = callable(repeat, fold) -> ordered gene list."""
    rows = []
    for p, fn in gene_sets.items():
        for r in range(repeats):
            for f in range(folds):
                for rank, g in enumerate(fn(r, f), start=1):
                    rows.append({"pipeline": p, "config_id": p.split("|")[0], "explainer": p.split("|")[1], "repeat": r, "fold": f, "rank": rank,
                                 "gene": g, "importance": 1.0 / rank, "signed": 1.0 if g.startswith("u") else -1.0})
    return pd.DataFrame(rows)


def test_identical_rankings_give_perfect_stability_in_every_dimension():
    same = lambda r, f: ["g1", "g2", "g3", "g4", "g5"]
    rk = make_rankings({p: same for p in PIPES.pipeline})
    rl = repeat_level_stability(rk, PIPES, [3], D)
    assert close(rl.phi.dropna(), 1.0) and close(rl.jaccard.dropna(), 1.0)
    assert set(rl.dimension) == {"resampling", "model", "explainer", "combined", "baseline"}


def test_dimensions_separate_model_from_resampling_effects():
    """Every pipeline is perfectly stable across folds, but the two models disagree: only the model dimension drops."""
    rk = make_rankings({"a|coef": lambda r, f: ["g1", "g2", "g3"], "a|shap": lambda r, f: ["g1", "g2", "g3"],
                        "b|shap": lambda r, f: ["h1", "h2", "h3"], "ttest|ttest": lambda r, f: ["g1", "g2", "g3"]})
    rl = repeat_level_stability(rk, PIPES, [3], D)
    res = rl[rl.dimension == "resampling"]
    assert close(res.phi, 1.0)                                             # A: stable across resamples
    assert close(rl[(rl.dimension == "explainer") & (rl.group == "model=a")].phi, 1.0)   # C: same model, explainers agree
    m = rl[(rl.dimension == "model") & (rl.group == "explainer=shap")]                       # B: a vs b disagree completely
    assert (m.jaccard == 0.0).all() and (m.phi < 0).all()
    assert (rl[rl.dimension == "combined"].jaccard < 1).all()


def test_within_fold_values_equal_direct_metric_computation():
    fn = {"a|coef": lambda r, f: [f"g{i}" for i in range(1 + f, 6 + f)], "a|shap": lambda r, f: [f"g{i}" for i in range(2, 7)],
          "b|shap": lambda r, f: [f"g{i}" for i in range(3, 8)], "ttest|ttest": lambda r, f: [f"g{i}" for i in range(1, 6)]}
    rk = make_rankings(fn, repeats=1, folds=3)
    rl = repeat_level_stability(rk, PIPES, [5], D)
    row = rl[(rl.dimension == "model") & (rl.group == "explainer=shap")].iloc[0]
    expect = np.mean([nogueira_stability([set(fn["a|shap"](0, f)), set(fn["b|shap"](0, f))], D) for f in range(3)])
    assert row.phi == pytest.approx(expect)
    res = rl[(rl.dimension == "resampling") & (rl.group == "a|coef")].iloc[0]
    assert res.phi == pytest.approx(nogueira_stability([set(fn["a|coef"](0, f)) for f in range(3)], D))
    assert res.jaccard == pytest.approx(jaccard_stability([set(fn["a|coef"](0, f)) for f in range(3)]))


def test_variable_set_sizes_give_phi_but_no_kuncheva():
    rk = make_rankings({"a|coef": lambda r, f: ["g1", "g2", "g3", "g4"][: 2 + f], "a|shap": lambda r, f: ["g1", "g2"], "b|shap": lambda r, f: ["g1", "g2"],
                        "ttest|ttest": lambda r, f: ["g1", "g2"]})
    rl = repeat_level_stability(rk, PIPES, [4], D)
    r = rl[(rl.dimension == "resampling") & (rl.group == "a|coef")]
    assert r.kuncheva.isna().all() and r.phi.notna().all() and close(r.mean_set_size, 3.0)


def test_summaries_pooled_and_pairwise():
    rk = make_rankings({p: (lambda r, f: ["g1", "g2", "g3"]) for p in PIPES.pipeline})
    rl = repeat_level_stability(rk, PIPES, [3], D)
    s = summarize_repeats(rl)
    assert {"phi_mean", "phi_median", "phi_min", "phi_max", "n_repeats"} <= set(s.columns) and (s.n_repeats == 2).all()
    pooled = pooled_resampling(rk, PIPES, [3], D)
    assert (pooled.n_sets == 6).all() and (pooled.n_distinct_genes == 3).all() and (pooled["n_genes_freq_ge_0.8"] == 3).all()
    M = pairwise_jaccard_matrix(rk, PIPES, 3)
    assert (M.to_numpy() == 1.0).all()


def test_group_definitions_only_use_valid_groups():
    labels = [(d, g) for d, g, _ in group_definitions(PIPES)]
    assert ("model", "explainer=shap") in labels and ("explainer", "model=a") in labels and ("combined", "all_primary") in labels
    assert ("model", "explainer=coef") not in labels                      # only one primary coef pipeline: no comparison possible
    assert ("baseline", "a|coef_vs_ttest") in labels


def test_consensus_table_rule_and_lists():
    # primary pipelines: a|coef, a|shap, b|shap -> consensus needs >= ceil(3/2) = 2 pipelines with own frequency >= 0.5
    fn = {"a|coef": lambda r, f: ["u1", "u2", "x3"], "a|shap": lambda r, f: ["u1", "u2", "y4"], "b|shap": lambda r, f: ["u1", f"z{f}", "z9"],
          "ttest|ttest": lambda r, f: ["u1", "u2", "w"]}
    rk = make_rankings(fn, repeats=2, folds=3)
    t = gene_table(rk, PIPES, k=3).set_index("gene")
    assert t.loc["u1", "mean_selection_frequency"] == pytest.approx(1.0) and t.loc["u1", "n_pipelines_ge50"] == 3 and t.loc["u1", "n_models_ge50"] == 2
    assert t.loc["u1", "n_explainers_ge50"] == 2 and t.loc["u2", "n_pipelines_ge50"] == 2 and t.loc["u2", "consensus"]
    assert not t.loc["x3", "consensus"] and t.loc["x3", "n_pipelines_ge50"] == 1
    assert t.loc["z0", "freq:b|shap"] == pytest.approx(1 / 3) and t.loc["u1", "ttest_frequency"] == 1.0
    assert t.loc["u1", "frac_signed_positive"] == 1.0 and t.loc["x3", "median_rank"] == 3
    fl = frozen_lists(rk, PIPES)
    assert fl["S_cons"] == ["u1", "u2"] and fl["S_top25"][:2] == ["u1", "u2"]
    f = universe_frequency(rk, PIPES, pd.Index(["u1", "nope", "x3"]), k=3)
    assert f["nope"] == 0.0 and f["u1"] == pytest.approx(1.0) and 0 < f["x3"] < 1
