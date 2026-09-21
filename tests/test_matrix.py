import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml
from toy import make_toy_cohort

import bioxplain.matrix as mx
from bioxplain.data.discovery import DiscoveryData, processing_month, restrict_subset
from bioxplain.validation.splits import repeated_stratified_splits

CFG = {
    "name": "toy_matrix", "dataset": "GSE42568", "subset": "full", "seed": 4, "cv": {"n_splits": 4, "n_repeats": 2},
    "preprocessing": {"min_detect_frac_of_minority": 0.5, "collapse_rule": "max_mean"}, "top_k": [3, 6], "store_top": 20, "n_perm": 3,
    "bootstrap": {"n_boot": 30}, "baseline": {"ttest": True, "ttest_lr_k": 5},
    "configs": [
        {"id": "logreg", "model": {"name": "logreg"}, "explainers": ["coef", "shap", "perm"]},
        {"id": "svm", "model": {"name": "svm"}, "explainers": ["coef", "shap", "perm"]},
        {"id": "rf", "model": {"name": "rf", "n_estimators": 30}, "explainers": ["shap", "perm"]},
        {"id": "xgb", "model": {"name": "xgb", "n_estimators": 40}, "explainers": ["shap", "perm"]},
        {"id": "logreg_C100", "model": {"name": "logreg", "C": 100.0}, "explainers": ["coef"], "role": "sensitivity"},
    ],
}


@pytest.fixture()
def project(tmp_path, monkeypatch):
    X, y, m, signal = make_toy_cohort(seed=21, n_normal=12, n_cancer=48, n_signal=4, n_noise=40, n_lowexp=5)
    titles = [f"Normal breast, N{i}_1_12_04" for i in range(12)] + [f"Breast cancer, T{i}_2_{'12' if i < 40 else '01'}_{'04' if i < 40 else '05'}" for i in range(48)]
    samples = pd.DataFrame({"title": titles}, index=X.index)
    data = DiscoveryData(X=X, y=pd.Series(y, index=X.index, name="cancer"), samples=samples, probe_to_gene=m, entrez=pd.Series(dtype=str))
    monkeypatch.setattr(mx, "load_discovery", lambda root: data)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/data_manifest.json").write_text(json.dumps({"files": {"GSE42568": {"sha256": "toy"}}}))
    cfg = tmp_path / "cfg.yaml"; cfg.write_text(yaml.safe_dump(CFG))
    return tmp_path, cfg, data, signal


def test_config_validation(tmp_path):
    def write(c):
        p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(c)); return p
    bad = {**CFG, "configs": [{"id": "rf", "model": {"name": "rf"}, "explainers": ["coef"]}]}
    with pytest.raises(ValueError, match="not defined"):
        mx.load_matrix_config(write(bad))
    with pytest.raises(ValueError, match="only use GSE42568"):
        mx.load_matrix_config(write({**CFG, "dataset": "GSE65194"}))
    with pytest.raises(ValueError, match="store_top"):
        mx.load_matrix_config(write({**CFG, "store_top": 2}))
    with pytest.raises(ValueError, match="unique"):
        mx.load_matrix_config(write({**CFG, "configs": CFG["configs"][:1] * 2}))
    ok = mx.load_matrix_config(write(CFG))
    assert ok["configs"][-1]["role"] == "sensitivity" and ok["configs"][0]["role"] == "primary"


def test_end_to_end_schema_positive_rule_and_planted_signal(project):
    root, cfg, data, signal = project
    out = mx.run_matrix(cfg, root, out_root=root / "out")
    rk = pd.read_csv(out / "rankings.csv.gz")
    assert {"pipeline", "config_id", "explainer", "repeat", "fold", "rank", "gene", "importance", "signed", "n_positive"} <= set(rk.columns)
    assert set(rk.pipeline) == {"logreg|coef", "logreg|shap", "logreg|perm", "svm|coef", "svm|shap", "svm|perm", "rf|shap", "rf|perm",
                                "xgb|shap", "xgb|perm", "logreg_C100|coef", "ttest|ttest"}
    assert (rk.importance > 0).all()                                     # D13b: no zero-attribution row is ever stored
    assert rk.groupby(["pipeline", "repeat", "fold"]).ngroups == 12 * 8
    assert set(signal) <= set(rk[(rk.pipeline == "ttest|ttest") & (rk["rank"] <= 6)].gene)
    xgb = rk[rk.pipeline == "xgb|shap"].groupby(["repeat", "fold"]).size()
    assert xgb.max() < 20                                                # sparse model: fewer positive genes than store_top
    met = pd.read_csv(out / "fold_metrics.csv")
    assert set(met.config_id) == {"logreg", "svm", "rf", "xgb", "logreg_C100", "ttest_lr"} and (met.roc_auc > 0.8).all()
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_folds"] == 8 and man["resolved_models"]["rf"]["n_jobs"] == 4 and man["sample_definition"]["n_normal"] == 12
    assert (out / "pooled_metrics.csv").exists() and (out / "pipelines.csv").exists()


def test_deterministic_across_runs(project):
    root, cfg, *_ = project
    a = mx.run_matrix(cfg, root, out_root=root / "a"); b = mx.run_matrix(cfg, root, out_root=root / "b")
    for f in ("rankings.csv.gz", "predictions.csv.gz"):
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f), check_exact=True)


def test_resume_recomputes_only_the_missing_fold(project):
    root, cfg, *_ = project
    out = mx.run_matrix(cfg, root, out_root=root / "r")
    ref = pd.read_csv(out / "rankings.csv.gz")
    folds = sorted((out / "folds").iterdir())
    victim, kept = folds[3], folds[0]
    mtime = os.path.getmtime(kept); victim.unlink()
    mx.run_matrix(cfg, root, resume_dir=out)
    assert victim.exists() and os.path.getmtime(kept) == mtime
    pd.testing.assert_frame_equal(pd.read_csv(out / "rankings.csv.gz"), ref, check_exact=True)


def test_every_pipeline_is_invariant_to_validation_fold_contents(project):
    """Leakage contract for the whole matrix (coef, SHAP, permutation importance, t-test, t-test classifier)."""
    root, cfg_path, data, _ = project
    cfg = mx.load_matrix_config(cfg_path); cfg["_probe_to_gene"] = data.probe_to_gene
    y = data.y.to_numpy(); sp = next(iter(repeated_stratified_splits(y, 4, 1, 4)))
    Xc = data.X.copy(); Xc.iloc[sp.test_idx, :] = 15.0 + np.random.default_rng(0).normal(size=(len(sp.test_idx), Xc.shape[1]))
    a = mx.run_matrix_fold(data.X, y, sp.train_idx, sp.test_idx, cfg, [0, 0])
    b = mx.run_matrix_fold(Xc, y, sp.train_idx, sp.test_idx, cfg, [0, 0])
    pd.testing.assert_frame_equal(a.rankings.reset_index(drop=True), b.rankings.reset_index(drop=True), check_exact=True)
    assert not np.allclose(a.predictions.score, b.predictions.score)     # the corruption did reach the predictions


def test_matrix_fold_rejects_overlap(project):
    root, cfg_path, data, _ = project
    cfg = mx.load_matrix_config(cfg_path); cfg["_probe_to_gene"] = data.probe_to_gene
    with pytest.raises(ValueError, match="overlap"):
        mx.run_matrix_fold(data.X, data.y.to_numpy(), np.arange(30), np.arange(25, 40), cfg, [0, 0])


def test_permuted_label_replicate_preserves_pipeline_and_class_counts(project):
    root, cfg_path, data, _ = project
    cfg = {**CFG, "labels": "permuted", "null": {"replicate": 3}, "configs": CFG["configs"][:1], "baseline": {"ttest": True, "ttest_lr_k": 5}}
    p = root / "null.yaml"; p.write_text(yaml.safe_dump(cfg))
    c = mx.load_matrix_config(p)
    d, y_perm, info = mx.prepare_data(c, root)
    assert int(y_perm.sum()) == int(data.y.sum()) and not np.array_equal(y_perm, data.y.to_numpy()) and info["null_replicate"] == 3
    d2, y2, _ = mx.prepare_data(c, root)
    assert np.array_equal(y_perm, y2)                                     # replicate index fully determines the permutation
    c4 = dict(c, null={"replicate": 4}); assert not np.array_equal(y_perm, mx.prepare_data(c4, root)[1])
    out = mx.run_matrix(p, root, out_root=root / "n")
    met = pd.read_csv(out / "fold_metrics.csv")
    assert met[met.config_id == "logreg"].roc_auc.mean() < 0.85          # signal destroyed -> near chance on toy data
    assert not (out / "pooled_metrics.csv").exists()


def test_processing_month_and_subsets(project):
    root, cfg, data, _ = project
    month = processing_month(data.samples["title"])
    assert month.iloc[0] == "04-12" and month.iloc[-1] == "05-01"
    dec = restrict_subset(data, "dec2004")
    assert (dec.y == 0).sum() == 12 and (dec.y == 1).sum() == 40
    ctrl = restrict_subset(data, "random_control", seed=1)
    assert (ctrl.y == 0).sum() == 12 and (ctrl.y == 1).sum() == 40
    assert not set(dec.X.index[dec.y == 1]) == set(ctrl.X.index[ctrl.y == 1])
    with pytest.raises(ValueError):
        restrict_subset(data, "nope")


def test_parallel_folds_give_identical_results(project):
    root, cfg_path, *_ = project
    par = {**CFG, "parallel_folds": 2}
    p = root / "par.yaml"; p.write_text(yaml.safe_dump(par))
    a = mx.run_matrix(cfg_path, root, out_root=root / "seq"); b = mx.run_matrix(p, root, out_root=root / "par")
    for f in ("rankings.csv.gz", "predictions.csv.gz"):
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f), check_exact=True)


def test_completed_folds_are_cached_even_if_a_later_fold_fails(project, monkeypatch):
    """Resumability under failure: folds finished before a crash stay on disk (written per fold, atomically)."""
    root, cfg_path, *_ = project
    real = mx.run_matrix_fold
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("simulated crash")
        return real(*a, **kw)

    monkeypatch.setattr(mx, "run_matrix_fold", flaky)
    out = root / "crash"
    with pytest.raises(RuntimeError, match="simulated crash"):
        mx.run_matrix(cfg_path, root, resume_dir=out)
    assert len(list((out / "folds").glob("*.pkl"))) == 3 and not list((out / "folds").glob("*.tmp"))
    monkeypatch.setattr(mx, "run_matrix_fold", real)
    mx.run_matrix(cfg_path, root, resume_dir=out)             # resumes and finishes
    assert len(list((out / "folds").glob("*.pkl"))) == 8


def test_empty_config_file_gives_a_clear_error(tmp_path):
    """Regression: an empty YAML file used to fail with an unhelpful TypeError."""
    p = tmp_path / "empty.yaml"; p.write_text("")
    with pytest.raises(ValueError, match="empty or not a YAML mapping"):
        mx.load_matrix_config(p)
