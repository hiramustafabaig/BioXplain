"""Processing-date sensitivity (Addendum A7) and tissue-composition ablation (A6), discovery cohort only.

    python scripts/analyze_sensitivity.py   ->  results/tables/date_sensitivity*.csv, composition_ablation*.csv, results/sensitivity_summary.json
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAB = ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)


def latest(pattern):
    hits = sorted(glob.glob(str(ROOT / pattern)))
    return pathlib.Path(hits[-1])


def load(run):
    man = json.loads((run / "manifest.json").read_text())
    return {"run": run, "man": man, "rl": pd.read_csv(run / "analysis/repeat_level_stability.csv"), "g25": pd.read_csv(run / "analysis/gene_table_k25.csv").set_index("gene"),
            "pm": pd.read_csv(run / "pooled_metrics.csv"), "rk": pd.read_csv(run / "rankings.csv.gz")}


def phi(rl, repeats=None):
    r = rl[(rl.dimension == "resampling") & (rl.k == 25)]
    if repeats is not None:
        r = r[r.repeat.isin(repeats)]
    return r.groupby("group").phi.mean()


def perf(pm, repeats=None):
    p = pm if repeats is None else pm[pm.repeat.isin(repeats)]
    return p.groupby("config_id").agg(roc_auc=("roc_auc", "mean"), ap_normal=("ap_normal", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))


def top(g25, n=25):
    return set(g25["mean_selection_frequency"].sort_values(ascending=False, kind="mergesort").head(n).index)


def compare(a, b, universe):
    fa, fb = a["g25"]["mean_selection_frequency"].reindex(universe).fillna(0), b["g25"]["mean_selection_frequency"].reindex(universe).fillna(0)
    ta, tb = top(a["g25"]), top(b["g25"])
    return {"spearman_universe_selection_frequency": float(stats.spearmanr(fa, fb).statistic), "top25_jaccard": len(ta & tb) / len(ta | tb), "top25_overlap": len(ta & tb),
            "spearman_among_genes_selected_in_either": float(stats.spearmanr(fa[(fa > 0) | (fb > 0)], fb[(fa > 0) | (fb > 0)]).statistic)}


def main():
    full, dec, ctrl, abl = (load(latest(p)) for p in ("results/matrix/*_full_matrix_*", "results/matrix/*_date_dec2004_*", "results/matrix/*_date_control_*", "results/matrix/*_composition_ablation_*"))
    out = {"sample_definitions": {k: v["man"]["sample_definition"] for k, v in (("full", full), ("dec2004", dec), ("control", ctrl), ("ablation", abl))}}
    universe = full["g25"].index.union(dec["g25"].index).union(ctrl["g25"].index)
    # ---- date sensitivity
    t = pd.DataFrame({"full": phi(full["rl"]), "dec2004": phi(dec["rl"]), "size_matched_control": phi(ctrl["rl"])})
    t["dec_minus_full"] = t.dec2004 - t.full; t["control_minus_full"] = t.size_matched_control - t.full; t["dec_minus_control"] = t.dec2004 - t.size_matched_control
    t.round(4).to_csv(TAB / "date_sensitivity_resampling_phi_k25.csv")
    pf = pd.DataFrame({"full": perf(full["pm"]).roc_auc, "dec2004": perf(dec["pm"]).roc_auc, "size_matched_control": perf(ctrl["pm"]).roc_auc}); pf.round(4).to_csv(TAB / "date_sensitivity_auc.csv")
    ov = {f"{a}_vs_{b}": compare(x, y, universe) for (a, x), (b, y) in [(("full", full), ("dec2004", dec)), (("full", full), ("control", ctrl)), (("dec2004", dec), ("control", ctrl))]}
    out["date_overlap"] = ov
    out["date_mean_abs_phi_change"] = {"dec2004_vs_full": float(t.dec_minus_full.abs().mean()), "control_vs_full": float(t.control_minus_full.abs().mean()), "dec2004_vs_control": float(t.dec_minus_control.abs().mean())}
    out["date_consensus_sizes"] = {k: int(v["g25"].consensus.sum()) for k, v in (("full", full), ("dec2004", dec), ("control", ctrl))}
    # ---- composition ablation vs the first two repeats of the full run (identical seed and splits)
    ta = pd.DataFrame({"full_repeats_0_1": phi(full["rl"], [0, 1]), "ablation": phi(abl["rl"])}); ta["change"] = ta.ablation - ta.full_repeats_0_1; ta.round(4).to_csv(TAB / "composition_ablation_phi_k25.csv")
    f01 = full["rk"][full["rk"].repeat.isin([0, 1])]
    pipes = pd.read_csv(full["run"] / "pipelines.csv")
    from bioxplain.consensus import gene_table
    g_full01 = gene_table(f01, pipes, 25).set_index("gene")
    panel = ["ADIPOQ", "PLIN1", "FABP4", "LEP", "LPL", "CFD", "ADH1B", "CIDEC", "PLIN4", "CD36", "GPD1"]
    ta_ = top(g_full01); tb_ = top(abl["g25"])
    out["composition_ablation"] = {"panel_genes_in_full_top25_repeats_0_1": sorted(ta_ & set(panel)), "top25_jaccard_full01_vs_ablation": len(ta_ & tb_) / len(ta_ | tb_), "top25_overlap": len(ta_ & tb_),
                                   "spearman_universe_frequency": float(stats.spearmanr(g_full01["mean_selection_frequency"].reindex(universe).fillna(0), abl["g25"]["mean_selection_frequency"].reindex(universe).fillna(0)).statistic),
                                   "mean_abs_phi_change": float(ta.change.abs().mean()), "auc_full01": perf(full["pm"], [0, 1]).roc_auc.round(4).to_dict(), "auc_ablation": perf(abl["pm"]).roc_auc.round(4).to_dict()}
    out["panel_genes_in_full_run_selection"] = {g: float(full["g25"]["mean_selection_frequency"].get(g, 0.0)) for g in panel}
    (ROOT / "results/sensitivity_summary.json").write_text(json.dumps(out, indent=2, default=str))
    pd.set_option("display.width", 220)
    print(json.dumps(out, indent=1, default=str)); print(t.round(3).to_string()); print(pf.round(3).to_string()); print(ta.round(3).to_string())


if __name__ == "__main__":
    main()
