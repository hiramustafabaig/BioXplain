"""Discovery freeze (Addendum A3, brief Phase 11).

`create_freeze` computes, from DISCOVERY data only, everything the external analysis is allowed to use: the frozen gene lists,
per-gene stability s_g, the frozen probe per gene, discovery effects and directions. It writes machine-readable and human-readable
freeze documents recording the code commit, configs, hashes, seeds and versions. The working tree must be clean.
After the freeze is committed, `external.verify_freeze` is the only gate to the external cohort.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pandas as pd

from bioxplain.consensus import HEADLINE_K, frozen_lists, gene_table, universe_frequency
from bioxplain.data.discovery import load_discovery
from bioxplain.models.factory import DEFAULTS
from bioxplain.preprocessing.transformers import FoldPreprocessor
from bioxplain.replication import auc_effects, hedges_g
from bioxplain.utils.provenance import git_state, sha256_file, software_versions, utc_now

FREEZE_JSON = "docs/freeze/discovery_freeze.json"


class FreezeError(RuntimeError):
    pass


def _dirty(root) -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip())


def create_freeze(root, run_dir, extra_refs: dict | None = None) -> dict:
    root, run = pathlib.Path(root), pathlib.Path(run_dir)
    if _dirty(root):
        raise FreezeError("working tree has uncommitted tracked changes; commit first so the freeze records an exact commit")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["git"]["dirty"]:
        raise FreezeError("the observed run was produced on a dirty tree")
    rk = pd.read_csv(run / "rankings.csv.gz")
    pipes = pd.read_csv(run / "pipelines.csv")
    data = load_discovery(root)
    y = data.y.to_numpy()
    prep = FoldPreprocessor(data.probe_to_gene).fit(data.X, y)               # frozen probe per gene from the FULL discovery cohort (D6)
    genes = list(prep.genes_)
    probes = prep.collapser_.selected_.loc[genes].to_numpy()
    Xg = data.X.loc[:, probes].to_numpy()
    auc = auc_effects(Xg, y)
    g = hedges_g(Xg[y == 1], Xg[y == 0])
    lists = frozen_lists(rk, pipes)
    s25 = universe_frequency(rk, pipes, pd.Index(genes), 25)
    s10, s50 = universe_frequency(rk, pipes, pd.Index(genes), 10), universe_frequency(rk, pipes, pd.Index(genes), 50)
    t25 = gene_table(rk, pipes, HEADLINE_K).set_index("gene")
    table = pd.DataFrame({"gene": genes, "probe": probes, "entrez_id": data.entrez.reindex(genes).to_numpy(), "e_disc": auc - 0.5,
                          "direction": np.where(auc > 0.5, "up_in_cancer", "down_in_cancer"), "hedges_g_disc": g,
                          "s_g_k25": s25.to_numpy(), "s_g_k10": s10.to_numpy(), "s_g_k50": s50.to_numpy()})
    keep = ["n_pipelines_ge50", "n_models_ge50", "n_explainers_ge50", "median_rank", "best_rank", "mean_rank", "ttest_frequency", "consensus", "frac_signed_positive"]
    table = table.merge(t25[keep].reset_index(names="gene"), on="gene", how="left")
    for name, members in lists.items():
        table[f"in_{name}"] = table["gene"].isin(members)
    out_csv = root / "results/freeze/frozen_genes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    freeze = {
        "frozen_utc": utc_now().isoformat(), "git": git_state(root), "software": software_versions(),
        "discovery_run": {"dir": str(run.relative_to(root)).replace("\\", "/"), "run_id": manifest["run_id"], "config_sha256": manifest["config_sha256"],
                          "code_commit_of_run": manifest["git"]["commit"], "n_folds": manifest["n_folds"], "seed": manifest["seed"],
                          "files_sha256": {f: sha256_file(run / f) for f in ("rankings.csv.gz", "pipelines.csv", "config.yaml", "manifest.json", "predictions.csv.gz")}},
        "data": {"discovery_dataset": "GSE42568", "discovery_file": json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE42568"],
                 "sample_definition": manifest["sample_definition"], "feature_universe_file": "configs/feature_universe.tsv",
                 "feature_universe_sha256": sha256_file(root / "configs/feature_universe.tsv"), "n_universe_probes": manifest["feature_universe"]["n_probes"],
                 "n_universe_genes": manifest["feature_universe"]["n_genes"], "n_eligible_genes_full_cohort": len(genes)},
        "preprocessing": {"expression_filter": "floor detection; detected in >= ceil(0.5 * n_minority_train) training samples (D7)",
                          "probe_collapse": "max training mean, ties by probe id (D6); frozen probe per gene chosen on the full discovery cohort",
                          "scaling": "z-score with training statistics for LR/SVM; log2 values for trees (D8)"},
        "models": manifest["resolved_models"], "model_defaults_pinned": DEFAULTS,
        "explainers": {"coef": "|coef| on z-scored genes (LR, SVM)", "shap": "mean |SHAP| over training-fold samples (Linear/Tree)",
                       "perm": "exact permutation importance, training fold, class-balanced Brier, R=10 (D12b)", "ttest": "|Welch t|, training fold only"},
        "top_k": [10, 25, 50], "headline_k": HEADLINE_K, "positive_only_rule": "D13b",
        "stability": {"primary": "Nogueira Phi (JMLR 2018 Def. 4)", "secondary": ["Kuncheva (constant size only)", "Jaccard"], "d": manifest["feature_universe"]["n_genes"],
                      "unit": "one 5-fold CV repeat", "dimensions": ["resampling", "model", "explainer", "combined", "baseline", "sensitivity"]},
        "consensus": {"rule": "within-pipeline selection frequency >= 0.5 at k=25 in >= ceil(n_primary/2) primary pipelines", "n_primary_pipelines": int((pipes.role == "primary").sum()),
                      "lists": {k: len(v) for k, v in lists.items()}, "frozen_lists": lists},
        "external_endpoints": "docs/methodology_decisions.md Addendum A4 (aligned effect r_g, C1, C2, signature score, frozen LR; bootstrap over external samples)",
        "frozen_genes_file": "results/freeze/frozen_genes.csv", "frozen_genes_sha256": sha256_file(out_csv),
        "references": extra_refs or {},
        "rules_after_freeze": ["no tuning, gene, k, threshold, preprocessing or endpoint change using GSE65194", "external results are reported whether or not favourable"],
    }
    path = root / FREEZE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, indent=2, default=str))
    return freeze
