"""Regularisation-strength sensitivity of logistic-regression coefficient rankings (decision D11c).

Robustness check, NOT hyper-parameter optimisation: the primary configuration (C = reference_C) never changes.
For every C in the pre-specified grid the identical splits, preprocessing and explainer are used; only C differs.
Discovery cohort only.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from bioxplain.data.discovery import load_discovery
from bioxplain.experiment import _pooled_metrics, load_config
from bioxplain.stability.summary import stability_summary, top_k_sets
from bioxplain.utils.provenance import config_hash, git_state, software_versions, utc_now
from bioxplain.validation.cv import run_fold
from bioxplain.validation.splits import repeated_stratified_splits

# Nogueira et al. (2018), Table 3 (after Fleiss et al. 2004): <0.40 poor, 0.40-0.75 intermediate to good, >0.75 excellent
BANDS = ((0.40, "poor"), (0.75, "intermediate_to_good"))


def phi_band(phi: float) -> str:
    for upper, name in BANDS:
        if phi < upper:
            return name
    return "excellent"


def evaluate_criterion(stab_k25: dict[float, float], min_auc: dict[float, float], reference_c: float, auc_floor: float) -> dict:
    """Pre-specified verdict: 'robust' iff every C has the same Phi band (k=25) as the reference and min AUC >= floor."""
    ref_band = phi_band(stab_k25[reference_c])
    detail = {c: {"phi_k25": stab_k25[c], "band": phi_band(stab_k25[c]), "min_pooled_auc": min_auc[c],
                  "same_band_as_reference": phi_band(stab_k25[c]) == ref_band, "auc_ok": min_auc[c] >= auc_floor}
              for c in stab_k25}
    robust = all(v["same_band_as_reference"] and v["auc_ok"] for v in detail.values())
    return {"verdict": "robust" if robust else "materially_changed", "reference_C": reference_c, "reference_band": ref_band,
            "auc_floor": auc_floor, "per_C": {str(c): v for c, v in detail.items()}}


def run_regularization_sensitivity(config_path, root, out_root=None) -> pathlib.Path:
    root = pathlib.Path(root)
    cfg = load_config(config_path)
    grid, ref_c = list(cfg["grid"]["C"]), cfg["grid"]["reference_C"]
    if ref_c not in grid:
        raise ValueError("reference_C must be part of the grid")
    t0 = time.perf_counter()
    started = utc_now()
    sens_id = f"{started:%Y%m%dT%H%M%SZ}_{cfg['name']}_{config_hash(cfg)[:8]}"
    out = pathlib.Path(out_root or root / "results/sensitivity") / sens_id
    out.mkdir(parents=True, exist_ok=False)

    data = load_discovery(root)
    y = data.y.to_numpy()
    d_universe = int(data.probe_to_gene.nunique())
    splits = list(repeated_stratified_splits(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"]))

    top_frames = {c: [] for c in grid}
    pred_frames = {c: [] for c in grid}
    imp = {}                               # (repeat, fold, C) -> |coef| Series indexed by gene (all eligible genes)
    top25 = {}                             # (repeat, fold, C) -> frozenset of top-25 genes
    for sp in splits:
        for c in grid:
            spec = {**cfg["model"], "C": c}
            res = run_fold(data.X, y, sp.train_idx, sp.test_idx, model_spec=spec, explainer_name=cfg["explainer"],
                           probe_to_gene=data.probe_to_gene, prep_cfg=cfg["preprocessing"], seed=cfg["seed"], store_top=None)
            key = {"C": c, "repeat": sp.repeat, "fold": sp.fold, "seed": sp.seed, "model": "logreg", "explainer": cfg["explainer"]}
            full = res.ranking
            imp[(sp.repeat, sp.fold, c)] = full.set_index("gene")["importance"].sort_index()
            top25[(sp.repeat, sp.fold, c)] = frozenset(full.loc[full["rank"] <= 25, "gene"])
            top_frames[c].append(full[full["rank"] <= cfg["store_top"]].assign(**key))
            pred_frames[c].append(res.predictions.assign(**key))

    stab_rows, pooled_rows = [], []
    for c in grid:
        rk = pd.concat(top_frames[c], ignore_index=True)
        stab_rows.append(stability_summary(rk, cfg["top_k"], d_universe, extra={"C": c}))
        pooled_rows.append(_pooled_metrics(pd.concat(pred_frames[c], ignore_index=True), cfg["bootstrap"]["n_boot"], cfg["seed"]).assign(C=c))
    stab = pd.concat(stab_rows, ignore_index=True)
    pooled = pd.concat(pooled_rows, ignore_index=True)

    agree = []
    for sp in splits:
        ref = imp[(sp.repeat, sp.fold, ref_c)]
        for c in grid:
            other = imp[(sp.repeat, sp.fold, c)]
            assert ref.index.equals(other.index), "gene universe must be identical across C within a fold"
            rho = spearmanr(ref.to_numpy(), other.to_numpy()).statistic
            a, b = top25[(sp.repeat, sp.fold, ref_c)], top25[(sp.repeat, sp.fold, c)]
            agree.append({"C": c, "reference_C": ref_c, "repeat": sp.repeat, "fold": sp.fold, "spearman_abs_coef": float(rho),
                          "jaccard_top25": len(a & b) / len(a | b), "n_genes": len(ref)})
    agree = pd.DataFrame(agree)

    k_crit = cfg["criterion"].get("k", 25)          # pre-specified criterion uses k = 25 (D11c)
    if k_crit not in cfg["top_k"]:
        raise ValueError(f"criterion.k={k_crit} must be one of top_k={cfg['top_k']}")
    k25 = stab[(stab.scope == "all_runs") & (stab.k == k_crit)].set_index("C")["nogueira"].to_dict()
    min_auc = pooled.groupby("C")["roc_auc"].min().to_dict()
    verdict = evaluate_criterion(k25, min_auc, ref_c, cfg["criterion"]["robust_auc_min"])
    verdict["criterion_k"] = k_crit

    stab.to_csv(out / "stability_by_C.csv", index=False)
    pooled.to_csv(out / "pooled_metrics_by_C.csv", index=False)
    agree.to_csv(out / "rank_agreement.csv", index=False)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    manifest = {
        "sensitivity_id": sens_id, "started_utc": started.isoformat(), "config_sha256": config_hash(cfg),
        "git": git_state(root), "software": software_versions(), "n_samples": int(len(y)), "n_normal": int((y == 0).sum()),
        "data": json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE42568"],
        "feature_universe": {"n_genes": d_universe}, "n_splits_total": len(splits), "grid_C": grid,
        "runtime_seconds": round(time.perf_counter() - t0, 2), "outputs": sorted(p.name for p in out.iterdir()) + ["manifest.json"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out
