"""Model x explainer experiment on the discovery cohort (GSE42568 only), leakage-safe, resumable.

Per training fold: ONE preprocessing fit (filter -> probe collapse -> z-scoring statistics), each model fitted once,
every explainer computed from that fitted model and the SAME training-fold matrix (no different explanation
populations between models), plus the model-free Welch-t baseline and a t-test-top-k logistic regression classifier.
Rankings keep only strictly-positive attributions (D13b). Each finished fold is cached, so an interrupted run resumes.

Config keys: name, dataset, subset (full|dec2004|random_control), seed, cv{n_splits,n_repeats}, preprocessing, top_k,
store_top, n_perm, bootstrap{n_boot}, configs[{id, model, explainers, role}], baseline{ttest, ttest_lr_k},
labels (real|permuted) and null{replicate}.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from bioxplain.data.discovery import DiscoveryData, load_discovery, restrict_subset
from bioxplain.evaluation.bootstrap import bootstrap_intervals
from bioxplain.evaluation.metrics import classification_metrics
from bioxplain.explainers.attribution import VALID_EXPLAINERS, explain
from bioxplain.explainers.ttest import ttest_attribution
from bioxplain.models.factory import SCALED, make_model, model_scores, model_support, resolve_spec
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.utils.provenance import config_hash, git_state, software_versions, utc_now
from bioxplain.validation.splits import repeated_stratified_splits

REQUIRED = ("name", "dataset", "subset", "seed", "cv", "preprocessing", "top_k", "store_top", "n_perm", "configs")
SEED_OFFSET_NULL = 100_000


def load_matrix_config(path) -> dict:
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config {path} is missing keys: {missing}")
    if cfg["dataset"] != "GSE42568":
        raise ValueError("discovery experiments may only use GSE42568; the external cohort is opened only after the freeze")
    if max(cfg["top_k"]) > cfg["store_top"]:
        raise ValueError("store_top must be >= max(top_k)")
    ids = [c["id"] for c in cfg["configs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("config ids must be unique")
    for c in cfg["configs"]:
        name = resolve_spec(c["model"])["name"]
        for e in c["explainers"]:
            if e not in VALID_EXPLAINERS or (e != "ttest" and name not in VALID_EXPLAINERS[e]):
                raise ValueError(f"explainer {e!r} is not defined for model {name!r}")
        c.setdefault("role", "primary")
    cfg.setdefault("baseline", {"ttest": True, "ttest_lr_k": 25})
    cfg.setdefault("labels", "real")
    return cfg


def pipeline_table(cfg: dict) -> pd.DataFrame:
    """One row per (config, explainer) 'pipeline' plus the t-test baseline."""
    rows = [{"pipeline": f"{c['id']}|{e}", "config_id": c["id"], "model": resolve_spec(c["model"])["name"], "explainer": e, "role": c["role"]}
            for c in cfg["configs"] for e in c["explainers"]]
    if cfg["baseline"].get("ttest"):
        rows.append({"pipeline": "ttest|ttest", "config_id": "ttest", "model": "none", "explainer": "ttest", "role": "baseline"})
    return pd.DataFrame(rows)


@dataclass
class FoldResult:
    rankings: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame        # one row per config (+ ttest_lr baseline)
    timings: dict


def _top_positive(attr: pd.DataFrame, store_top: int) -> pd.DataFrame:
    return attr[attr["importance"] > 0].head(store_top)[["rank", "gene", "importance", "signed"]]


def run_matrix_fold(X: pd.DataFrame, y: np.ndarray, train_idx, test_idx, cfg: dict, seed_key: list[int]) -> FoldResult:
    y = np.asarray(y).astype(int)
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train and test indices overlap")
    seed = cfg["seed"]
    pc = cfg["preprocessing"]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    tm: dict[str, float] = {}
    t0 = time.perf_counter()
    prep = FoldPreprocessor(cfg["_probe_to_gene"],
                            pc.get("min_detect_frac_of_minority", 0.5), pc.get("collapse_rule", "max_mean")).fit(X_train, y_train)
    genes = list(prep.genes_)
    Z_tr = {s: prep.transform(X_train, scale=s).to_numpy() for s in (True, False)}
    Z_te = {s: prep.transform(X.iloc[test_idx], scale=s).to_numpy() for s in (True, False)}
    tm["prep"] = time.perf_counter() - t0
    rank_rows, pred_rows, met_rows = [], [], []
    ss = np.random.SeedSequence([seed, *seed_key])
    for c_i, c in enumerate(cfg["configs"]):
        spec = resolve_spec(c["model"])
        name, scaled = spec["name"], SCALED[spec["name"]]
        t0 = time.perf_counter()
        model = make_model(spec, seed, y_train).fit(Z_tr[scaled], y_train)
        tm[f"fit:{c['id']}"] = time.perf_counter() - t0
        score, pred = model_scores(model, name, Z_te[scaled]), model.predict(Z_te[scaled])
        support = model_support(model, name, len(genes))
        met_rows.append({"config_id": c["id"], **classification_metrics(y[test_idx], score, pred), "n_features_used": int(len(support)),
                         "n_genes": len(genes), "n_train": len(train_idx), "n_test": len(test_idx)})
        pred_rows.append(pd.DataFrame({"config_id": c["id"], "sample": X.index[test_idx], "y_true": y[test_idx], "score": score, "pred": pred.astype(int)}))
        for e in c["explainers"]:
            t0 = time.perf_counter()
            child = np.random.SeedSequence(ss.entropy, spawn_key=(*ss.spawn_key, c_i, 1 if e == "perm" else 0))
            attr = explain(e, model, name, Z_tr[scaled], y_train, genes, child, cfg["n_perm"])
            tm[f"explain:{c['id']}|{e}"] = time.perf_counter() - t0
            rank_rows.append(_top_positive(attr, cfg["store_top"]).assign(config_id=c["id"], explainer=e, n_positive=int((attr["importance"] > 0).sum())))
    if cfg["baseline"].get("ttest"):
        attr = ttest_attribution(Z_tr[False], y_train, genes)          # |t| is scale-invariant; training fold only
        rank_rows.append(_top_positive(attr, cfg["store_top"]).assign(config_id="ttest", explainer="ttest", n_positive=int((attr["importance"] > 0).sum())))
        k = cfg["baseline"].get("ttest_lr_k", 25)
        top = [genes.index(g) for g in attr["gene"].head(k)]          # classification with the top-k t-test genes (training fold only)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed).fit(Z_tr[True][:, top], y_train)
        s = lr.predict_proba(Z_te[True][:, top])[:, 1]
        met_rows.append({"config_id": "ttest_lr", **classification_metrics(y[test_idx], s, lr.predict(Z_te[True][:, top])), "n_features_used": k,
                         "n_genes": len(genes), "n_train": len(train_idx), "n_test": len(test_idx)})
        pred_rows.append(pd.DataFrame({"config_id": "ttest_lr", "sample": X.index[test_idx], "y_true": y[test_idx], "score": s, "pred": lr.predict(Z_te[True][:, top]).astype(int)}))
    return FoldResult(pd.concat(rank_rows, ignore_index=True), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(met_rows), tm)


def prepare_data(cfg: dict, root) -> tuple[DiscoveryData, np.ndarray, dict]:
    data = restrict_subset(load_discovery(root), cfg["subset"], cfg.get("subset_seed", 0))
    if cfg.get("exclude_genes"):                     # composition ablation (A6): remove all probes of the listed genes from the universe
        keep = ~data.probe_to_gene.isin(set(cfg["exclude_genes"]))
        data = DiscoveryData(X=data.X.loc[:, keep[keep].index], y=data.y, samples=data.samples, probe_to_gene=data.probe_to_gene[keep], entrez=data.entrez)
    y = data.y.to_numpy()
    info = {"excluded_genes": cfg.get("exclude_genes", []), "subset": cfg["subset"], "n_samples": int(len(y)), "n_cancer": int(y.sum()), "n_normal": int((y == 0).sum()), "labels": cfg["labels"]}
    if cfg["labels"] == "permuted":
        b = cfg["null"]["replicate"]
        y = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 777, b])).permutation(y)     # class counts preserved
        info["null_replicate"] = b
    return data, y, info


def run_matrix(config_path, root, out_root=None, resume_dir=None) -> pathlib.Path:
    root = pathlib.Path(root)
    cfg = load_matrix_config(config_path)
    started = utc_now()
    run_id = f"{started:%Y%m%dT%H%M%SZ}_{cfg['name']}_{config_hash({k: v for k, v in cfg.items() if not k.startswith('_')})[:8]}"
    out = pathlib.Path(resume_dir) if resume_dir else pathlib.Path(out_root or root / "results/matrix") / run_id
    (out / "folds").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data, y, info = prepare_data(cfg, root)
    cfg["_probe_to_gene"] = data.probe_to_gene
    is_null = cfg["labels"] == "permuted"
    split_seed = cfg["seed"] + (SEED_OFFSET_NULL * (1 + cfg["null"]["replicate"]) if is_null else 0)
    splits = list(repeated_stratified_splits(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], split_seed))
    key_prefix = [10_000 + cfg["null"]["replicate"]] if is_null else []
    cache = {(sp.repeat, sp.fold): out / "folds" / f"fold_r{sp.repeat}_f{sp.fold}.pkl" for sp in splits}   # Split holds arrays: not hashable
    todo = [sp for sp in splits if not cache[(sp.repeat, sp.fold)].exists()]

    def compute(sp):
        """Compute one fold and cache it IMMEDIATELY (atomic rename), so an interrupted run loses at most the folds in flight."""
        tf = time.perf_counter()
        res = run_matrix_fold(data.X, y, sp.train_idx, sp.test_idx, cfg, key_prefix + [sp.repeat, sp.fold])
        res.timings["fold_total"] = time.perf_counter() - tf
        target = cache[(sp.repeat, sp.fold)]
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(res))
        tmp.replace(target)

    workers = int(cfg.get("parallel_folds", 1))
    if workers > 1 and len(todo) > 1:                       # results do not depend on the worker count: every fold is seeded independently
        from joblib import Parallel, delayed
        Parallel(n_jobs=workers)(delayed(compute)(sp) for sp in todo)
    else:
        for sp in todo:
            compute(sp)
    fold_times, results = [], []
    for sp in splits:
        res = pickle.loads(cache[(sp.repeat, sp.fold)].read_bytes())
        results.append(({"repeat": sp.repeat, "fold": sp.fold, "seed": sp.seed}, res))
        fold_times.append(res.timings.get("fold_total", float("nan")))
    rank = pd.concat([r.rankings.assign(**k) for k, r in results], ignore_index=True)
    pred = pd.concat([r.predictions.assign(**k) for k, r in results], ignore_index=True)
    met = pd.concat([r.metrics.assign(**k) for k, r in results], ignore_index=True)
    tim = pd.DataFrame([{**k, **{a: b for a, b in r.timings.items()}} for k, r in results])
    rank["pipeline"] = rank["config_id"] + "|" + rank["explainer"]
    rank.to_csv(out / "rankings.csv.gz", index=False)
    pred.to_csv(out / "predictions.csv.gz", index=False)
    met.to_csv(out / "fold_metrics.csv", index=False)
    tim.to_csv(out / "timings.csv", index=False)
    pipeline_table(cfg).to_csv(out / "pipelines.csv", index=False)
    if not is_null:
        pooled = []
        for cid, g in pred.groupby("config_id", sort=False):
            for rep, gg in g.groupby("repeat"):
                yy, ss_, pp = gg.y_true.to_numpy(), gg.score.to_numpy(), gg.pred.to_numpy()
                ci = bootstrap_intervals(yy, ss_, pp, cfg.get("bootstrap", {}).get("n_boot", 2000), cfg["seed"])
                pooled.append({"config_id": cid, "repeat": rep, "n": len(gg), **classification_metrics(yy, ss_, pp),
                               **{f"{a}_ci_{s}": ci[a][i] for a in ci for i, s in enumerate(("lo", "hi"))}})
        pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "config.yaml").write_text(yaml.safe_dump(clean, sort_keys=True))
    manifest = {
        "run_id": run_id, "started_utc": started.isoformat(), "config_sha256": config_hash(clean), "git": git_state(root), "software": software_versions(),
        "data": json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE42568"], "sample_definition": info,
        "feature_universe": {"n_probes": int(len(data.probe_to_gene)), "n_genes": int(data.probe_to_gene.nunique())},
        "resolved_models": {c["id"]: resolve_spec(c["model"]) for c in cfg["configs"]}, "seed": cfg["seed"], "split_seed": split_seed,
        "n_folds": len(results), "runtime_seconds": {"total": round(time.perf_counter() - t0, 1), "fold_mean": round(float(np.nanmean(fold_times)), 2)},
        "outputs": sorted(p.name for p in out.iterdir() if p.is_file()) + ["manifest.json"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
