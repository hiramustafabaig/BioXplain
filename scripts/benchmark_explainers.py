"""Benchmark SHAP and exact permutation importance on ONE real training fold (GSE42568 only), all four models.

Reports runtime, peak memory, output dimensions, SHAP additivity, repeatability, share of (near-)zero attributions,
top-k compatibility, and brute-force equivalence of the permutation shortcut on real data (sampled genes).
Output: results/benchmarks/explainers_<id>.json (+ printed table).
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import psutil

from bioxplain.data.discovery import load_discovery
from bioxplain.explainers.permutation import (TOLERANCE, make_permutations, permutation_importance, permutation_importance_bruteforce,
                                              probability_function)
from bioxplain.explainers.shap_explainer import shap_explain, shap_values
from bioxplain.models.factory import SCALED, make_model, model_scores, model_support
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.utils.provenance import git_state, software_versions, utc_now
from bioxplain.validation.splits import repeated_stratified_splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260921


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / 2**20


d = load_discovery(ROOT)
y = d.y.to_numpy()
sp = next(iter(repeated_stratified_splits(y, 5, 1, SEED)))
Xtr, ytr = d.X.iloc[sp.train_idx], y[sp.train_idx]
prep = FoldPreprocessor(d.probe_to_gene).fit(Xtr, ytr)
out = {"fold": {"n_train": len(ytr), "n_normal": int((ytr == 0).sum())}, "models": {}}
rng = np.random.default_rng(1)
for name in ("logreg", "svm", "rf", "xgb"):
    Z = prep.transform(Xtr, scale=SCALED[name]).to_numpy()
    p = Z.shape[1]
    model = make_model({"name": name}, SEED, ytr).fit(Z, ytr)
    support = model_support(model, name, p)
    rec = {"n_genes": p, "support": int(len(support))}
    # ---- SHAP
    t = time.perf_counter(); v = shap_values(model, name, Z); rec["shap_seconds"] = round(time.perf_counter() - t, 2)
    v2 = shap_values(model, name, Z)
    rec["shap_shape"] = list(v.shape); rec["shap_repeatable_bit_identical"] = bool(np.array_equal(v, v2))
    _, base = shap_explain(model, name, Z)
    if name in ("logreg", "svm"):
        out_model = model.decision_function(Z)
    elif name == "xgb":
        out_model = model.predict(Z, output_margin=True)
    else:
        out_model = model.predict_proba(Z)[:, 1]
    rec["shap_additivity_max_abs_error"] = float(np.abs(v.sum(1) + base - out_model).max())
    imp = np.abs(v).mean(0)
    rec["shap_share_exact_zero"] = float((imp == 0).mean()); rec["shap_share_below_1e-6_of_max"] = float((imp < 1e-6 * imp.max()).mean())
    rec["shap_positive_genes"] = int((imp > 0).sum()); rec["shap_top1_over_median_positive"] = float(imp.max() / np.median(imp[imp > 0]))
    if name in ("rf", "xgb"):
        rec["shap_zero_outside_support"] = bool((imp[np.setdiff1d(np.arange(p), support)] == 0).all())
    # ---- permutation importance
    perms = make_permutations(Z.shape[0], 10, np.random.SeedSequence([SEED, 0, 0]))
    t = time.perf_counter(); pi = permutation_importance(model, name, Z, ytr, perms); rec["perm_seconds"] = round(time.perf_counter() - t, 2)
    pi2 = permutation_importance(model, name, Z, ytr, perms)
    rec["perm_shape"] = list(pi.shape); rec["perm_repeatable_bit_identical"] = bool(np.array_equal(pi, pi2))
    rec["perm_positive_genes"] = int((pi > 0).sum()); rec["perm_negative_genes"] = int((pi < 0).sum()); rec["perm_exact_zero"] = int((pi == 0).sum())
    # equivalence with brute force on real data: sample of used and (for linear) arbitrary genes
    pool = support if name in ("rf", "xgb") else np.arange(p)
    sample = rng.choice(pool, size=min(25, len(pool)), replace=False)
    unused = np.setdiff1d(np.arange(p), support)
    sample = np.r_[sample, rng.choice(unused, 5, replace=False)] if len(unused) else sample
    bf = permutation_importance_bruteforce(probability_function(model, name), Z, ytr, perms, features=sample)
    rec["perm_vs_bruteforce_max_abs_diff_on_sampled_genes"] = float(np.abs(bf[sample] - pi[sample]).max())
    rec["perm_bruteforce_within_tolerance"] = bool(rec["perm_vs_bruteforce_max_abs_diff_on_sampled_genes"] <= TOLERANCE)
    rec["peak_rss_mb_so_far"] = round(peak_mb(), 0)
    # top-k compatibility (positive-only rule)
    for label, arr in (("shap", imp), ("perm", pi)):
        order = np.argsort(-arr, kind="stable"); pos = int((arr > 0).sum())
        rec[f"{label}_top50_sizes_after_positive_rule"] = min(50, pos)
    out["models"][name] = rec
    print(name, json.dumps(rec, indent=1))
out["tolerance"] = TOLERANCE
out["provenance"] = {"git": git_state(ROOT), "software": software_versions(), "utc": utc_now().isoformat()}
dest = ROOT / "results/benchmarks" / f"explainers_{utc_now():%Y%m%dT%H%M%SZ}.json"
dest.write_text(json.dumps(out, indent=2, default=str))
print("written", dest)
