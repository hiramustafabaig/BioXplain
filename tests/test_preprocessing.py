import numpy as np
import pandas as pd
import pytest
from toy import FLOOR, make_toy_cohort

from bioxplain.preprocessing.transformers import FloorDetectionFilter, FoldPreprocessor, ProbeGeneCollapser


def _floor_matrix():
    """20 samples: 4 minority (y=0) + 16 majority (y=1); floor = 2.0 (plenty of values on it)."""
    y = np.r_[np.zeros(4, int), np.ones(16, int)]
    n = len(y)
    cols = {
        "always_floor": np.full(n, 2.0),
        "rare_noise": np.where(np.arange(n) == 10, 5.0, 2.0),             # detected in 1 sample
        "minority_only": np.where(y == 0, 6.0, 2.0),                       # detected only in the minority class
        "common": np.full(n, 7.0),
    }
    return pd.DataFrame(cols, index=[f"s{i}" for i in range(n)]), y


def test_filter_drops_undetected_but_keeps_minority_specific_gene():
    X, y = _floor_matrix()
    f = FloorDetectionFilter(min_detect_frac_of_minority=0.5).fit(X, y)
    assert f.floor_ == 2.0
    assert f.threshold_ == 2                                                # ceil(0.5 * 4)
    assert set(f.kept_) == {"minority_only", "common"}
    assert list(f.transform(X).columns) == list(f.kept_)


def test_filter_threshold_scales_with_minority_size():
    X, y = _floor_matrix()
    assert FloorDetectionFilter(1.0).fit(X, y).threshold_ == 4
    assert FloorDetectionFilter(0.25).fit(X, y).threshold_ == 1


def test_a_fixed_fraction_of_all_samples_would_remove_minority_specific_genes():
    """Justifies D7: a 20%-of-all-samples rule (4 of 20) is stricter than needed, the minority-based rule is not."""
    X, y = _floor_matrix()
    detected = (X > 2.0 + 1e-9).sum()
    assert detected["minority_only"] == 4
    assert detected["minority_only"] >= FloorDetectionFilter(0.5).fit(X, y).threshold_


def test_filter_warns_and_passes_everything_when_no_floor():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 1, (20, 8)), columns=list("abcdefgh"))
    y = np.r_[np.zeros(5, int), np.ones(15, int)]
    with pytest.warns(UserWarning, match="no array floor"):
        f = FloorDetectionFilter().fit(X, y)
    assert f.floor_ is None and list(f.transform(X).columns) == list(X.columns)


def test_collapser_max_mean_variance_and_tiebreak():
    X = pd.DataFrame({"pA": [1.0, 1.0, 1.0, 1.0], "pB": [5.0, 9.0, 5.0, 9.0], "pC": [6.0, 6.0, 6.0, 6.0],
                      "pD": [6.0, 6.0, 6.0, 6.0], "pX": [3.0, 3.0, 3.0, 3.0]})
    m = pd.Series({"pA": "G1", "pB": "G1", "pC": "G2", "pD": "G2"})          # pX unmapped
    c = ProbeGeneCollapser(m, "max_mean").fit(X)
    assert c.selected_.to_dict() == {"G1": "pB", "G2": "pC"}                # pC/pD tie -> alphabetical
    assert list(c.transform(X).columns) == ["G1", "G2"]
    assert ProbeGeneCollapser(m, "max_variance").fit(X).selected_["G1"] == "pB"
    with pytest.raises(ValueError):
        ProbeGeneCollapser(m, "median")


def test_collapser_choice_is_learned_from_training_rows_only():
    """The chosen probe depends on which rows fit() sees, so fitting on all rows would leak validation information."""
    X = pd.DataFrame({"pA": [9.0, 9.0, 0.0, 0.0], "pB": [5.0, 5.0, 5.0, 5.0]})   # pA mean: 9 on train rows, 4.5 on all
    m = pd.Series({"pA": "G", "pB": "G"})
    train = ProbeGeneCollapser(m).fit(X.iloc[:2])
    everything = ProbeGeneCollapser(m).fit(X)
    assert train.selected_["G"] == "pA" and everything.selected_["G"] == "pB"


def test_fold_preprocessor_uses_training_statistics_for_test_data():
    X, y, m, _ = make_toy_cohort(seed=1)
    tr, te = np.arange(0, 40), np.arange(40, 60)
    prep = FoldPreprocessor(m).fit(X.iloc[tr], y[tr])
    Ztr, Zte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
    assert list(Ztr.columns) == list(Zte.columns)
    np.testing.assert_allclose(Ztr.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(Ztr.std(ddof=0), 1, atol=1e-9)
    assert abs(Zte.to_numpy().mean()) > 1e-6            # test rows are NOT re-centred with their own statistics
    d = prep.diagnostics()
    assert d["n_probes_after_filter"] < d["n_probes_input"] and d["floor"] == pytest.approx(FLOOR)


def test_fold_preprocessor_drops_low_expression_genes():
    X, y, m, _ = make_toy_cohort(seed=2)
    prep = FoldPreprocessor(m).fit(X, y)
    assert not any(g.startswith("L") for g in prep.genes_)      # mostly-floor genes removed
    assert all(g in prep.genes_ for g in ["S0", "S1", "N0"])
