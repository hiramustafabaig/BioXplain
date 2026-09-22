"""Sensitivity of the discovery results to the probe-collapse rule (D6) and the expression-filter threshold (D7).

Each sensitivity run uses 2 repeats x 5 folds with the same seed and splits as repeats 0-1 of the full run, so the reference is those
two repeats. Reports per-pipeline resampling Phi (k = 25), performance, top-25 overlap and universe-wide selection-frequency correlation.
Discovery cohort only.   python scripts/analyze_preprocessing_sensitivity.py
"""
from __future__ import annotations

import glob
import json
import pathlib

import pandas as pd
from scipy import stats

from bioxplain.consensus import gene_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAB = ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)


def latest(pattern):
    return pathlib.Path(sorted(glob.glob(str(ROOT / pattern)))[-1])


def phi(rl, repeats=None):
    r = rl[(rl.dimension == "resampling") & (rl.k == 25)]
    return (r if repeats is None else r[r.repeat.isin(repeats)]).groupby("group").phi.mean()


def top(g, n=25):
    return set(g["mean_selection_frequency"].sort_values(ascending=False, kind="mergesort").head(n).index)


def main():
    full = latest("results/matrix/*_full_matrix_*")
    pipes = pd.read_csv(full / "pipelines.csv")
    rk = pd.read_csv(full / "rankings.csv.gz"); rk = rk[rk.repeat.isin([0, 1])]
    ref_g = gene_table(rk, pipes, 25).set_index("gene")
    ref_rl = pd.read_csv(full / "analysis/repeat_level_stability.csv")
    pm = pd.read_csv(full / "pooled_metrics.csv"); ref_auc = pm[pm.repeat.isin([0, 1])].groupby("config_id").roc_auc.mean()
    table, summary = pd.DataFrame({"reference_full_run_repeats_0_1": phi(ref_rl, [0, 1])}), {}
    aucs = {"reference": ref_auc}
    for name in ("sens_collapse_var", "sens_filter_025", "sens_filter_100"):
        run = latest(f"results/matrix/*_{name}_*")
        rl = pd.read_csv(run / "analysis/repeat_level_stability.csv"); g = pd.read_csv(run / "analysis/gene_table_k25.csv").set_index("gene")
        table[name] = phi(rl)
        universe = ref_g.index.union(g.index)
        fa, fb = ref_g["mean_selection_frequency"].reindex(universe).fillna(0), g["mean_selection_frequency"].reindex(universe).fillna(0)
        ta, tb = top(ref_g), top(g)
        man = json.loads((run / "manifest.json").read_text())
        summary[name] = {"top25_overlap": len(ta & tb), "top25_jaccard": len(ta & tb) / len(ta | tb), "spearman_universe_selection_frequency": float(stats.spearmanr(fa, fb).statistic),
                         "mean_abs_phi_change_k25": float((table[name] - table["reference_full_run_repeats_0_1"]).abs().mean()), "n_genes": man["feature_universe"]["n_genes"],
                         "consensus_genes": int(g.consensus.sum()), "git_commit": man["git"]["commit"][:8], "dirty": man["git"]["dirty"]}
        aucs[name] = pd.read_csv(run / "pooled_metrics.csv").groupby("config_id").roc_auc.mean()
    table.round(4).to_csv(TAB / "preprocessing_sensitivity_phi_k25.csv"); pd.DataFrame(aucs).round(4).to_csv(TAB / "preprocessing_sensitivity_auc.csv")
    summary["reference_consensus_genes"] = int(ref_g.consensus.sum())
    (ROOT / "results/preprocessing_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=1)); print(table.round(3).to_string()); print(pd.DataFrame(aucs).round(3).to_string())


if __name__ == "__main__":
    main()
