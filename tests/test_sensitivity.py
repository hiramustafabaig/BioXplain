import json

import pandas as pd
import pytest
import yaml
from toy import make_toy_cohort

import bioxplain.sensitivity as sens
from bioxplain.data.discovery import DiscoveryData
from bioxplain.sensitivity import evaluate_criterion, phi_band, run_regularization_sensitivity

CFG = {
    "name": "toy_sens", "dataset": "GSE42568", "seed": 3, "cv": {"n_splits": 4, "n_repeats": 2},
    "preprocessing": {"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean", "scale": True},
    "model": {"name": "logreg", "class_weight": "balanced", "max_iter": 2000},
    "grid": {"C": [0.01, 1.0, 100.0], "reference_C": 1.0},
    "explainer": "coef", "top_k": [3, 6], "store_top": 12, "bootstrap": {"n_boot": 50},
    "criterion": {"k": 6, "robust_auc_min": 0.98},
}


def test_phi_bands_follow_the_paper_scale_at_the_boundaries():
    assert phi_band(0.399) == "poor" and phi_band(0.40) == "intermediate_to_good"
    assert phi_band(0.749) == "intermediate_to_good" and phi_band(0.75) == "excellent"
    assert phi_band(-0.2) == "poor"


def test_criterion_robust_only_if_band_and_auc_unchanged():
    ok = evaluate_criterion({0.1: 0.30, 1.0: 0.32, 10.0: 0.35}, {0.1: 0.99, 1.0: 0.99, 10.0: 0.99}, 1.0, 0.98)
    assert ok["verdict"] == "robust"
    band_change = evaluate_criterion({0.1: 0.55, 1.0: 0.32, 10.0: 0.35}, {0.1: 0.99, 1.0: 0.99, 10.0: 0.99}, 1.0, 0.98)
    assert band_change["verdict"] == "materially_changed" and not band_change["per_C"]["0.1"]["same_band_as_reference"]
    auc_drop = evaluate_criterion({0.1: 0.30, 1.0: 0.32}, {0.1: 0.97, 1.0: 0.99}, 1.0, 0.98)
    assert auc_drop["verdict"] == "materially_changed" and not auc_drop["per_C"]["0.1"]["auc_ok"]


@pytest.fixture()
def project(tmp_path, monkeypatch):
    X, y, m, signal = make_toy_cohort(seed=6)
    data = DiscoveryData(X=X, y=pd.Series(y, index=X.index, name="cancer"), samples=pd.DataFrame(index=X.index),
                         probe_to_gene=m, entrez=pd.Series(dtype=str))
    monkeypatch.setattr(sens, "load_discovery", lambda root: data)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/data_manifest.json").write_text(json.dumps({"files": {"GSE42568": {"sha256": "toy"}}}))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    return tmp_path, cfg


def test_outputs_reference_agreement_and_provenance(project):
    root, cfg = project
    out = run_regularization_sensitivity(cfg, root, out_root=root / "out")
    for f in ["stability_by_C.csv", "pooled_metrics_by_C.csv", "rank_agreement.csv", "verdict.json", "manifest.json", "config.yaml"]:
        assert (out / f).exists()
    agree = pd.read_csv(out / "rank_agreement.csv")
    ref = agree[agree.C == 1.0]
    assert ref.spearman_abs_coef.between(1 - 1e-9, 1 + 1e-9).all() and (ref.jaccard_top25 == 1.0).all()   # C=1 vs itself
    assert len(agree) == 8 * 3 and set(agree.C) == {0.01, 1.0, 100.0}
    stab = pd.read_csv(out / "stability_by_C.csv")
    assert set(stab.C) == {0.01, 1.0, 100.0} and (stab.n_sets[stab.scope == "all_runs"] == 8).all()
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_splits_total"] == 8 and man["config_sha256"] and man["software"]["scikit-learn"]
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["verdict"] in {"robust", "materially_changed"} and set(verdict["per_C"]) == {"0.01", "1.0", "100.0"}


def test_identical_splits_across_C_and_determinism(project):
    root, cfg = project
    a = run_regularization_sensitivity(cfg, root, out_root=root / "a")
    b = run_regularization_sensitivity(cfg, root, out_root=root / "b")
    for f in ["stability_by_C.csv", "rank_agreement.csv"]:
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f), check_exact=True)
    pm = pd.read_csv(a / "pooled_metrics_by_C.csv")
    assert (pm.groupby("C").n.first() == 60).all()


def test_reference_C_must_be_in_grid(project, tmp_path):
    root, cfg = project
    bad = dict(CFG, grid={"C": [0.1, 10.0], "reference_C": 1.0})
    p = root / "bad.yaml"; p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="reference_C"):
        run_regularization_sensitivity(p, root, out_root=root / "c")


def test_criterion_k_must_be_one_of_top_k(project):
    root, _ = project
    bad = dict(CFG, criterion={"k": 25, "robust_auc_min": 0.98})
    p = root / "badk.yaml"; p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="criterion.k"):
        run_regularization_sensitivity(p, root, out_root=root / "d")
