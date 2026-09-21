import json

import numpy as np
import pandas as pd
import pytest
import yaml
from toy import make_toy_cohort

import bioxplain.experiment as exp
from bioxplain.data.discovery import DiscoveryData
from bioxplain.experiment import load_config, run_experiment
from bioxplain.stability.summary import top_k_sets

CFG = {
    "name": "toy", "dataset": "GSE42568", "seed": 5, "cv": {"n_splits": 4, "n_repeats": 2},
    "preprocessing": {"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean", "scale": True},
    "model": {"name": "logreg", "C": 1.0, "class_weight": "balanced", "max_iter": 2000},
    "explainer": "coef", "top_k": [3, 6], "store_top": 12, "bootstrap": {"n_boot": 50},
}


@pytest.fixture()
def project(tmp_path, monkeypatch):
    X, y, m, signal = make_toy_cohort(seed=4)
    data = DiscoveryData(X=X, y=pd.Series(y, index=X.index, name="cancer"), samples=pd.DataFrame(index=X.index),
                         probe_to_gene=m, entrez=pd.Series(dtype=str))
    monkeypatch.setattr(exp, "load_discovery", lambda root: data)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/data_manifest.json").write_text(json.dumps({"files": {"GSE42568": {"sha256": "toy"}}}))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    return tmp_path, cfg_path, signal


def test_end_to_end_outputs_schema_and_provenance(project):
    root, cfg_path, _ = project
    out = run_experiment(cfg_path, root, out_root=root / "out")
    for name in ["rankings_top.csv.gz", "predictions.csv", "runs.csv", "pooled_metrics.csv", "stability.csv", "config.yaml", "manifest.json"]:
        assert (out / name).exists(), name
    rk = pd.read_csv(out / "rankings_top.csv.gz")
    assert {"experiment_id", "dataset", "model", "explainer", "repeat", "fold", "seed", "rank", "gene", "probe", "importance", "signed", "n_detected_train"} <= set(rk.columns)
    assert rk.groupby(["repeat", "fold"]).ngroups == 8 and rk["rank"].max() == 12
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 8 and (runs.n_train + runs.n_test == 60).all()
    man = json.loads((out / "manifest.json").read_text())
    assert man["config_sha256"] and man["software"]["scikit-learn"] and man["n_normal"] == 12 and "cv_loop_s" in man["timings_seconds"]
    pooled = pd.read_csv(out / "pooled_metrics.csv")
    assert len(pooled) == 2 and (pooled["n"] == 60).all()


def test_planted_signal_genes_are_recovered_and_stable(project):
    root, cfg_path, signal = project
    out = run_experiment(cfg_path, root, out_root=root / "out")
    rk = pd.read_csv(out / "rankings_top.csv.gz")
    stab = pd.read_csv(out / "stability.csv")
    top6 = set().union(*top_k_sets(rk, 6))
    assert set(signal) <= top6                                             # the 6 planted genes are always found
    row = stab[(stab.scope == "all_runs") & (stab.k == 6)].iloc[0]
    assert row.nogueira > 0.8 and row.n_distinct_genes < 12


def test_deterministic_given_config(project):
    root, cfg_path, _ = project
    a = run_experiment(cfg_path, root, out_root=root / "out")
    b = run_experiment(cfg_path, root, out_root=root / "out2")
    cols = ["repeat", "fold", "rank", "gene", "probe", "importance", "signed"]
    pd.testing.assert_frame_equal(pd.read_csv(a / "rankings_top.csv.gz")[cols], pd.read_csv(b / "rankings_top.csv.gz")[cols])
    pd.testing.assert_frame_equal(pd.read_csv(a / "predictions.csv")[["sample", "score"]], pd.read_csv(b / "predictions.csv")[["sample", "score"]])


def test_noise_labels_give_low_stability(project, monkeypatch):
    root, cfg_path, _ = project
    X, y, m, _ = make_toy_cohort(seed=4, n_signal=0)
    y_perm = np.random.default_rng(1).permutation(y)
    data = DiscoveryData(X=X, y=pd.Series(y_perm, index=X.index, name="cancer"), samples=pd.DataFrame(index=X.index),
                         probe_to_gene=m, entrez=pd.Series(dtype=str))
    monkeypatch.setattr(exp, "load_discovery", lambda r: data)
    stab = pd.read_csv(run_experiment(cfg_path, root, out_root=root / "noise") / "stability.csv")
    assert stab[(stab.scope == "all_runs") & (stab.k == 6)].nogueira.iloc[0] < 0.25


def test_config_validation(tmp_path):
    bad = dict(CFG); bad.pop("seed")
    p = tmp_path / "bad.yaml"; p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(p)
    ext = dict(CFG, dataset="GSE65194"); p.write_text(yaml.safe_dump(ext))
    with pytest.raises(ValueError, match="external"):
        load_config(p)
    small = dict(CFG, store_top=2); p.write_text(yaml.safe_dump(small))
    with pytest.raises(ValueError, match="store_top"):
        load_config(p)
