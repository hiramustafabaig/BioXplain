import gzip
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from bioxplain.biology.composition import PRIMARY_PANEL, panel_membership, panel_score
from bioxplain.biology.enrichment import benjamini_hochberg, enrich
from bioxplain.data.geo import read_series_matrix
from bioxplain.external import build_units, load_external, verify_freeze
from bioxplain.freeze import FREEZE_JSON, FreezeError
from bioxplain.replication import (auc_effects, bootstrap_c1_c2, c1_c2, direction_consistency, frozen_lr_transfer, hedges_g, percentile_ci,
                                   residual_auc, signature_score, within_cohort_z)


# ---------------------------------------------------------------- replication statistics
def test_auc_effects_match_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(9, int), np.ones(21, int)]
    X = np.round(rng.normal(size=(30, 6)) + y[:, None] * np.array([0, 1, -1, 2, 0.5, 0]), 1)      # rounding -> ties
    ours = auc_effects(X, y)
    ref = [roc_auc_score(y, X[:, j]) for j in range(6)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    with pytest.raises(ValueError):
        auc_effects(X, np.ones(30, int))


def test_auc_effects_are_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(8, int), np.ones(12, int)]
    X = rng.normal(size=(20, 5)) + y[:, None]
    np.testing.assert_allclose(auc_effects(X, y), auc_effects(np.exp(3 * X) + 10, y), atol=1e-12)     # scale-free by construction


def test_c1_c2_known_answers():
    rng = np.random.default_rng(2)
    p = 2000
    e_disc = rng.normal(0, 0.2, p)
    s = np.clip(np.abs(e_disc) + rng.normal(0, 0.05, p), 0, None)               # stability tracks discovery effect
    auc_ext = 0.5 + 0.5 * e_disc + rng.normal(0, 0.02, p)                       # external replicates discovery
    out = c1_c2(s, e_disc, auc_ext)
    assert out["c1_spearman"] > 0.3 and out["n_genes"] == p
    # stability adds nothing beyond |e_disc| here (s is a noisy function of |e_disc|): partial association near 0
    assert abs(out["c2_partial_spearman"]) < 0.1
    # now stability carries information beyond discovery effect
    hidden = rng.normal(size=p)
    auc2 = 0.5 + 0.5 * e_disc * (1 + 0.5 * (hidden > 0)) + 0.1 * hidden * 0.2
    s2 = 0.5 * np.abs(e_disc) + 0.5 * (hidden > 0)
    assert c1_c2(s2, e_disc, auc2)["c2_partial_spearman"] > 0.3


def test_bootstrap_is_reproducible_and_interval_contains_estimate():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(11, int), np.ones(40, int)]
    e = rng.normal(0, 0.2, 60); s = np.abs(e) + rng.random(60) * 0.1
    X = rng.normal(size=(51, 60)) + y[:, None] * e[None, :] * 4
    a = bootstrap_c1_c2(s, e, X, y, 40, seed=5); b = bootstrap_c1_c2(s, e, X, y, 40, seed=5)
    pd.testing.assert_frame_equal(a, b)
    est = c1_c2(s, e, auc_effects(X, y))["c1_spearman"]
    lo, hi = percentile_ci(a["c1_spearman"])
    assert lo <= est <= hi


def test_direction_consistency_uses_exact_interval():
    r = direction_consistency(np.array([1, 1, 1, -1, -1.0] * 4), np.array([1, 1, -1, -1, 1.0] * 4))
    assert r["n"] == 20 and r["n_consistent"] == 12 and r["fraction"] == pytest.approx(0.6)
    ref = stats.binomtest(12, 20).proportion_ci(method="exact")
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((ref.low, ref.high))


def test_hedges_g_and_within_cohort_z_and_signature_score():
    rng = np.random.default_rng(4)
    x, y = rng.normal(1, 1, (30, 3)), rng.normal(0, 1, (12, 3))
    g = hedges_g(x, y)
    assert g.shape == (3,) and (g > 0).all()
    Z = within_cohort_z(np.vstack([x, y]))
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12); np.testing.assert_allclose(Z.std(0), 1, atol=1e-12)
    sc = signature_score(Z, up=np.array([0, 1]), down=np.array([2]))
    np.testing.assert_allclose(sc, Z[:, [0, 1]].mean(1) - Z[:, 2])
    assert np.allclose(signature_score(Z, np.array([], int), np.array([], int)), 0)


def test_frozen_lr_transfer_and_residual_auc():
    rng = np.random.default_rng(5)
    yd = np.r_[np.zeros(15, int), np.ones(60, int)]; ye = np.r_[np.zeros(11, int), np.ones(40, int)]
    Xd = rng.normal(size=(75, 8)) + yd[:, None] * 2
    Xe = 3.0 * (rng.normal(size=(51, 8)) + ye[:, None] * 2) + 7.0                 # different scale, same direction
    r = frozen_lr_transfer(Xd, yd, Xe, ye, n_boot=30, seed=1)
    assert r["roc_auc"] > 0.95 and r["n"] == 51 and len(r["roc_auc_ci"]) == 2
    cov = rng.normal(size=51); score = ye * 1.0 + cov
    assert residual_auc(ye, score, cov) > residual_auc(ye, cov, cov) + 0.2


# ---------------------------------------------------------------- enrichment
def test_benjamini_hochberg_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg(np.array([0.9, 0.95])).max() <= 1.0


def test_enrichment_hypergeometric_background_and_size_filters():
    lib = {"A": [f"g{i}" for i in range(10)], "B": [f"g{i}" for i in range(10, 20)], "tiny": ["g0", "g1"], "huge": [f"g{i}" for i in range(0, 60)]}
    background = [f"g{i}" for i in range(50)] + ["notinlib1", "notinlib2"]
    fg = [f"g{i}" for i in range(8)] + ["notinlib1"]                           # genes outside the library are dropped from both sides
    df = enrich(fg, background, lib, min_size=5, max_size=15)
    assert set(df.term) == {"A", "B"}                                          # tiny (<5) and huge (>15 within background) excluded
    a = df.set_index("term").loc["A"]
    assert a.foreground_size == 8 and a.background_size == 50 and a.overlap == 8 and a.term_size_in_background == 10
    assert a.p_value == pytest.approx(stats.hypergeom.sf(7, 50, 10, 8))
    assert a.adj_p_bh < 0.05 and df.set_index("term").loc["B"].overlap == 0


# ---------------------------------------------------------------- composition
def test_panel_membership_and_score():
    s = pd.Series({"ADIPOQ": 0.9, "PLIN1": 0.0, "X": 0.1, "Y": 0.0})
    m = panel_membership({"S_top25": ["ADIPOQ"], "S_cons": []}, s)
    row = m.set_index("gene").loc["ADIPOQ"]
    assert row.in_S_top25 and not row.in_S_cons and row.percentile_in_universe == 1.0 and row.panel == "primary"
    assert not m.set_index("gene").loc["FABP4", "in_universe"]
    Z = np.array([[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]])
    np.testing.assert_allclose(panel_score(Z, ["ADIPOQ", "PLIN1", "Q"], PRIMARY_PANEL), [2.0, 3.0])
    with pytest.raises(ValueError):
        panel_score(Z, ["Q", "R", "S"], PRIMARY_PANEL)


# ---------------------------------------------------------------- external access control and unit rules
def _toy_gse65194(tmp_path):
    titles = ["TNBC, TUM001_repA", "TNBC, TUM001_repB", "Luminal A, TUM002", "Healthy, H1", "Healthy, H2", "CellLine, MCF-12A"]
    groups = ["TNBC", "TNBC", "Luminal A", "Healthy", "Healthy", "CellLine"]
    gsm = [f"GSM{i}" for i in range(6)]
    q = lambda v: "\t".join(f'"{x}"' for x in v)
    lines = ['!Series_title\t"toy"', "!Sample_title\t" + q(titles), "!Sample_geo_accession\t" + q(gsm), "!Sample_source_name_ch1\t" + q(["s"] * 6),
             "!Sample_characteristics_ch1\t" + q([f"sample_group: {g}" for g in groups]), "!Sample_description\t" + q(["d"] * 6),
             "!series_matrix_table_begin", '"ID_REF"\t' + q(gsm)]
    vals = np.array([[1.0, 3.0, 5.0, 7.0, 8.0, 100.0], [2.0, 2.0, 2.0, 1.0, 1.0, 50.0]])
    for name, row in zip(["p1", "p2"], vals):
        lines.append(f'"{name}"\t' + "\t".join(map(str, row)))
    lines.append("!series_matrix_table_end")
    f = tmp_path / "toy65194.txt.gz"
    with gzip.open(f, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return f


def test_external_unit_rules_average_duplicates_and_exclude_cell_lines(tmp_path):
    header, expr = read_series_matrix(_toy_gse65194(tmp_path))
    d = build_units(header, expr)
    assert list(d.X.index) == ["GSM3", "GSM4", "TUM001", "TUM002"] and d.y.tolist() == [0, 0, 1, 1]
    assert d.X.loc["TUM001", "p1"] == pytest.approx(2.0) and d.units.loc["TUM001", "n_arrays"] == 2      # mean of 1.0 and 3.0
    assert "GSM5" not in d.X.index and d.n_arrays_used == 5
    a = build_units(header, expr, "repA")
    assert a.X.loc["TUM001", "p1"] == 1.0 and len(a.X) == 4
    with pytest.raises(ValueError):
        build_units(header, expr, "other")


def test_external_loader_refuses_without_a_valid_freeze(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "--allow-empty", "-q", "-m", "c0"], cwd=tmp_path, check=True)
    with pytest.raises(FreezeError, match="not found"):
        load_external(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    (tmp_path / "docs/freeze").mkdir(parents=True); (tmp_path / "results/freeze").mkdir(parents=True)
    genes = tmp_path / "results/freeze/frozen_genes.csv"; genes.write_text("gene\nA\n")
    from bioxplain.utils.provenance import sha256_file
    fz = {"git": {"commit": head}, "frozen_genes_file": "results/freeze/frozen_genes.csv", "frozen_genes_sha256": sha256_file(genes)}
    (tmp_path / FREEZE_JSON).write_text(json.dumps(fz))
    assert verify_freeze(tmp_path)["git"]["commit"] == head                    # valid freeze passes
    genes.write_text("gene\nA\nB\n")                                           # modified after freeze
    with pytest.raises(FreezeError, match="modified"):
        verify_freeze(tmp_path)
    genes.write_text("gene\nA\n")
    fz["git"]["commit"] = "0" * 40; (tmp_path / FREEZE_JSON).write_text(json.dumps(fz))
    with pytest.raises(FreezeError, match="ancestor"):
        verify_freeze(tmp_path)
