"""Generate all publication figures from saved result files (no manual editing).

    python scripts/make_figures.py            # figures whose inputs exist are produced; the others are skipped with a message

Outputs PNG (300 dpi) + PDF in results/figures/. Each figure answers one research question (see FIGURES below).
"""
from __future__ import annotations

import glob
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "font.family": "DejaVu Sans"})
COL = {"logreg": "#1b6ca8", "svm": "#e07b00", "rf": "#2a9d55", "xgb": "#a4373a", "ttest": "#555555", "logreg_C0.1": "#7fb3d5", "logreg_C100": "#0b3d63", "null": "#bbbbbb"}


def latest(pattern: str):
    hits = sorted(glob.glob(str(ROOT / pattern)))
    return pathlib.Path(hits[-1]) if hits else None


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def colour(pipeline: str) -> str:
    return COL.get(pipeline.split("|")[0], "#777777")


# ---------------------------------------------------------------- Fig 1: workflow
def fig_workflow():
    fig, ax = plt.subplots(figsize=(8.4, 3.9)); ax.axis("off"); ax.set_xlim(0, 12.2); ax.set_ylim(0, 6)
    W, H = 2.15, 1.25
    top = [("GSE42568 (discovery)\n104 cancer / 17 normal", "#dbe9f6"), ("Leakage-safe folds\nfilter, probe collapse,\nscaling: train only", "#dbe9f6"),
           ("4 models\nLR, SVM, RF, XGBoost", "#dbe9f6"), ("Explainers: coef, SHAP,\npermutation importance\n+ Welch-t baseline", "#dbe9f6"),
           ("Top-k gene sets\nk = 10 / 25 / 50\n(positive attribution)", "#dbe9f6")]
    mid = [("Stability dimensions:\nresampling, model,\nexplainer, combined", "#e8f3e1"), ("Label-permutation null\n30 replicates,\nidentical pipeline", "#e8f3e1"),
           ("Consensus table\n+ DISCOVERY FREEZE", "#fde7c8"), ("GSE65194 (external)\n130 tumours, 11 healthy\nscale-free endpoints", "#f6d5d5")]
    low = [("Processing-date\nsensitivity (Dec-2004)", "#eeeeee"), ("Enrichment\n(eligible-universe background)", "#eeeeee"),
           ("Tissue-composition\nchecks", "#eeeeee"), ("Internal stability vs\nexternal replication (C1, C2)", "#f6d5d5")]

    def draw(row, y, xs):
        for (t, c), x in zip(row, xs):
            ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.03", fc=c, ec="#444", lw=0.8))
            ax.text(x + W / 2, y + H / 2, t, ha="center", va="center", fontsize=6.2)

    xt, xm, xl = [0.1 + i * 2.4 for i in range(5)], [0.1 + i * 3.0 for i in range(4)], [0.1 + i * 3.0 for i in range(4)]
    draw(top, 4.5, xt); draw(mid, 2.3, xm); draw(low, 0.1, xl)
    arrow = dict(arrowstyle="->", lw=0.9)
    for x in xt[:-1]:
        ax.annotate("", xy=(x + 2.4 + 0.02, 5.12), xytext=(x + W + 0.03, 5.12), arrowprops=arrow)
    ax.annotate("", xy=(xm[0] + W / 2, 3.6), xytext=(xt[-1] + W / 2, 4.47), arrowprops=arrow)
    for x in xm[:-1]:
        ax.annotate("", xy=(x + 3.0 + 0.02, 2.92), xytext=(x + W + 0.03, 2.92), arrowprops=arrow)
    ax.annotate("", xy=(xl[3] + W / 2, 1.38), xytext=(xm[3] + W / 2, 2.27), arrowprops=arrow)
    ax.set_title("Figure 1. BioXplain workflow: prediction, explanation, stability, replication, biological specificity", loc="left", fontsize=8.5)
    save(fig, "fig01_workflow")


# ---------------------------------------------------------------- Fig 2: discovery cohort overview
def fig_cohort():
    f = ROOT / "results/metrics/sample_manifest_GSE42568.csv"
    if not f.exists():
        return print("skip fig02 (no manifest)")
    m = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    axes[0].bar(["cancer", "normal"], [(m.label == "cancer").sum(), (m.label == "normal").sum()], color=["#a4373a", "#1b6ca8"]); axes[0].set_title("A  Classes (n = 121)"); axes[0].set_ylabel("samples")
    d = m.assign(month=m.date_ddmmyy.str.split("-").str[1:].str.join("-").where(m.date_ddmmyy.notna(), "undated"))
    d["month"] = d.date_ddmmyy.map(lambda s: "undated" if pd.isna(s) else ("2004-12" if s.split("-")[1:] == ["12", "04"] else "2005-01" if s.split("-")[1:] == ["1", "05"] else "other"))
    ct = pd.crosstab(d.month, d.label).reindex(["2004-12", "2005-01", "undated"]).fillna(0)
    ct.plot.bar(ax=axes[1], color=["#a4373a", "#1b6ca8"], width=0.8, rot=0); axes[1].set_title("B  Processing month vs class"); axes[1].set_xlabel("")
    pf = pd.crosstab(m.title_prefix, m.label); pf.plot.bar(ax=axes[2], color=["#a4373a", "#1b6ca8"], width=0.8, rot=0, legend=False); axes[2].set_title("C  Title prefix vs class"); axes[2].set_xlabel("")
    fig.suptitle("Figure 2. Discovery cohort GSE42568: class balance and known processing confounding", x=0.01, ha="left", fontsize=9, y=1.08)
    fig.tight_layout()
    save(fig, "fig02_discovery_cohort")


# ---------------------------------------------------------------- Fig 3: performance (not a ranking)
def fig_performance(run):
    p = pd.read_csv(run / "pooled_metrics.csv")
    order = ["logreg", "svm", "rf", "xgb", "logreg_C0.1", "logreg_C100", "ttest_lr"]
    g = p.groupby("config_id")[["roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi", "ap_normal", "ap_normal_ci_lo", "ap_normal_ci_hi", "balanced_accuracy", "sensitivity", "specificity"]].mean().reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (m, lo, hi, t) in zip(axes[:2], [("roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi", "ROC-AUC"), ("ap_normal", "ap_normal_ci_lo", "ap_normal_ci_hi", "AP (normal class)")]):
        ax.errorbar(g[m], y, xerr=[g[m] - g[lo], g[hi] - g[m]], fmt="o", color="#333", capsize=2, ms=4); ax.set_title(t); ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0.9, 1.005)
    axes[2].scatter(g["sensitivity"], y, c="#a4373a", label="sensitivity"); axes[2].scatter(g["specificity"], y, c="#1b6ca8", label="specificity"); axes[2].legend(loc="lower left"); axes[2].set_title("Fixed-threshold metrics"); axes[2].set_xlim(0.6, 1.02)
    fig.suptitle("Figure 3. Out-of-fold performance on GSE42568 (mean of 20 repeats; bootstrap CI). Near-ceiling AUC; models are not ranked", x=0.01, ha="left", fontsize=9, y=1.08)
    fig.tight_layout()
    save(fig, "fig03_performance")


# ---------------------------------------------------------------- Fig 4: stability distributions
def fig_stability(run):
    rl = pd.read_csv(run / "analysis/repeat_level_stability.csv")
    r = rl[rl.dimension == "resampling"]
    order = [g for g in ["logreg|coef", "logreg|shap", "logreg|perm", "svm|coef", "svm|shap", "svm|perm", "rf|shap", "rf|perm", "xgb|shap", "xgb|perm", "ttest|ttest", "logreg_C0.1|coef", "logreg_C100|coef"] if g in set(r.group)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    for ax, k in zip(axes, (10, 25, 50)):
        for i, gname in enumerate(order):
            v = r[(r.group == gname) & (r.k == k)].phi.dropna()
            ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, medianprops=dict(color="k"), boxprops=dict(color=colour(gname)), whiskerprops=dict(color=colour(gname)), capprops=dict(color=colour(gname)))
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=6, color=colour(gname), alpha=.7)
        ax.axhline(0.4, ls=":", c="#999", lw=.7); ax.axhline(0.75, ls=":", c="#999", lw=.7); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90); ax.set_title(f"k = {k}")
    axes[0].set_ylabel("Nogueira Φ (5 folds per repeat)"); axes[0].set_ylim(-0.1, 1.0)
    fig.suptitle("Figure 4. Resampling stability of top-k gene sets per pipeline (dotted: Φ = 0.40 poor / 0.75 excellent, Nogueira et al. 2018)", x=0.01, ha="left", fontsize=9)
    save(fig, "fig04_resampling_stability")


# ---------------------------------------------------------------- Fig 5: model x explainer agreement
def fig_heatmap(run):
    M = pd.read_csv(run / "analysis/pairwise_jaccard_k25.csv", index_col=0)
    order = [c for c in ["logreg|coef", "logreg|shap", "logreg|perm", "svm|coef", "svm|shap", "svm|perm", "rf|shap", "rf|perm", "xgb|shap", "xgb|perm", "ttest|ttest"] if c in M.index]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.95})
    im = axes[0].imshow(M.loc[order, order], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels(order, rotation=90); axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            axes[0].text(j, i, f"{M.loc[order[i], order[j]]:.2f}", ha="center", va="center", fontsize=4.6, color="w" if M.loc[order[i], order[j]] < 0.6 else "k")
    fig.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.32, label="mean within-fold Jaccard (k = 25)"); axes[0].set_title("A  Pairwise agreement of top-25 sets")
    s = pd.read_csv(run / "analysis/stability_summary.csv"); s = s[(s.k == 25) & s.dimension.isin(["model", "explainer", "combined"])]
    y = np.arange(len(s))[::-1]; axes[1].barh(y, s.phi_mean, color="#7a7a7a", xerr=[s.phi_mean - s.phi_min, s.phi_max - s.phi_mean], capsize=2)
    axes[1].set_yticks(y); axes[1].set_yticklabels(s.dimension + ": " + s.group, fontsize=6.5); axes[1].set_xlabel("Nogueira Phi within fold (mean; range over repeats)"); axes[1].set_title("B  Model / explainer / combined")
    fig.suptitle("Figure 5. Agreement between analytic choices at the same training data (k = 25)", x=0.01, ha="left", fontsize=9)
    save(fig, "fig05_model_explainer_agreement")


# ---------------------------------------------------------------- Fig 6: observed vs null
def fig_null(run, null_dirs):
    from bioxplain.null import null_repeat_level
    obs = pd.read_csv(run / "analysis/repeat_level_stability.csv")
    d = json.loads((run / "manifest.json").read_text())["feature_universe"]["n_genes"]
    null = null_repeat_level(null_dirs, [10, 25, 50], d)
    null.to_csv(run / "analysis/null_repeat_level_stability.csv", index=False)
    from bioxplain.null import compare_observed_null
    comp = compare_observed_null(obs, null); comp.to_csv(run / "analysis/observed_vs_null.csv", index=False)
    sel = [("resampling", g) for g in ["logreg|coef", "svm|coef", "rf|shap", "xgb|shap", "ttest|ttest"]] + [("model", "explainer=shap"), ("explainer", "model=logreg"), ("combined", "all_primary")]
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 5.4), sharey=True); axes = axes.ravel()
    for ax, (dim, g) in zip(axes, sel):
        n = null[(null.dimension == dim) & (null.group == g) & (null.k == 25)].phi.dropna(); o = obs[(obs.dimension == dim) & (obs.group == g) & (obs.k == 25)].phi.dropna()
        parts = ax.violinplot([n, o], showextrema=False, widths=0.8)
        for pc, c in zip(parts["bodies"], ["#bbbbbb", "#1b6ca8"]):
            pc.set_facecolor(c); pc.set_alpha(0.8)
        ax.scatter(np.full(len(n), 1) + np.random.default_rng(0).uniform(-.1, .1, len(n)), n, s=4, c="#666"); ax.scatter(np.full(len(o), 2) + np.random.default_rng(1).uniform(-.1, .1, len(o)), o, s=4, c="#0b3d63")
        ax.set_xticks([1, 2]); ax.set_xticklabels([f"null\n(n={len(n)})", f"observed\n(n={len(o)})"], fontsize=6.5); ax.set_title(f"{dim}: {g}", fontsize=6.5, pad=4)
    axes[0].set_ylabel("Φ at k = 25"); axes[4].set_ylabel("Φ at k = 25")
    fig.suptitle("Figure 6. Observed stability (real labels) vs permutation-label null; unit = one 5-fold repeat, same pipeline", x=0.01, ha="left", fontsize=9, y=1.03)
    fig.tight_layout(h_pad=3.0)
    save(fig, "fig06_observed_vs_null")


# ---------------------------------------------------------------- Fig 7: top stable genes
def fig_top_genes(run):
    g = pd.read_csv(run / "analysis/gene_table_k25.csv").head(30)
    cols = [c for c in g.columns if c.startswith("freq:") and "C0.1" not in c and "C100" not in c]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(g[cols].to_numpy(), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[5:] for c in cols], rotation=90); ax.set_yticks(range(len(g))); ax.set_yticklabels(g.gene, fontsize=6)
    for i, (mf, n) in enumerate(zip(g.mean_selection_frequency, g.n_pipelines_ge50)):
        ax.text(len(cols) - 0.4, i, f"  mean {mf:.2f} | ≥0.5 in {n}", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.28, label="selection frequency across folds (k = 25)")
    ax.set_title("Figure 7. Thirty highest mean-selection-frequency genes (rows) by pipeline (columns); model-supported signals, not biomarkers", loc="left", fontsize=8)
    save(fig, "fig07_top_stable_genes")


# ---------------------------------------------------------------- Fig 8: external replication
def fig_external(ext):
    s = json.loads((ext / "summary.json").read_text()); pg = pd.read_csv(ext / "per_gene_external_effects.csv"); ll = pd.read_csv(ext / "list_level_replication.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    ax = axes[0]; sel = pg.s_g_k25 > 0
    ax.scatter(pg.e_disc.abs()[~sel], pg.aligned_effect[~sel], s=1, c="#cccccc", label="s = 0"); ax.scatter(pg.e_disc.abs()[sel], pg.aligned_effect[sel], s=5, c=pg.s_g_k25[sel], cmap="viridis", label="selected")
    ax.axhline(0, c="k", lw=.5); ax.set_xlabel("|discovery effect| (AUC − 0.5)"); ax.set_ylabel("aligned external effect"); ax.set_title("A  Gene-level effects")
    g = s["primary_gene_level"]; names = ["C1\nSpearman(s, r)", "C2 partial\nSpearman"]; vals = [g["c1_spearman"], g["c2_partial_spearman"]]
    cis = [g.get("c1_spearman_ci", [np.nan] * 2), g.get("c2_partial_spearman_ci", [np.nan] * 2)]
    axes[1].bar(names, vals, color=["#1b6ca8", "#a4373a"]); axes[1].errorbar(range(2), vals, yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]], fmt="none", c="k", capsize=3); axes[1].axhline(0, c="k", lw=.5)
    axes[1].set_title("B  Internal stability vs external replication")
    l = ll.dropna(subset=["signature_auc_external"]); ci = l["signature_auc_external_ci"].map(lambda x: json.loads(x.replace("(", "[").replace(")", "]")) if isinstance(x, str) else [np.nan] * 2)
    axes[2].errorbar(range(len(l)), l.signature_auc_external, yerr=[l.signature_auc_external - [c[0] for c in ci], [c[1] for c in ci] - l.signature_auc_external], fmt="o", c="k", capsize=3)
    axes[2].set_xticks(range(len(l))); axes[2].set_xticklabels(l["list"]); axes[2].set_ylim(0.4, 1.02); axes[2].axhline(0.5, ls=":", c="#888"); axes[2].set_title("C  Frozen-signature AUC (external)")
    fig.suptitle("Figure 8. External replication on GSE65194 (130 tumours + 11 healthy; scale-free endpoints; same platform)", x=0.01, ha="left", fontsize=9, y=1.08)
    fig.tight_layout()
    save(fig, "fig08_external_replication")


# ---------------------------------------------------------------- Fig 9: date sensitivity
def fig_date(full, dec, ctrl):
    sf, sd, sc = (pd.read_csv(r / "analysis/stability_summary.csv") for r in (full, dec, ctrl))
    f = lambda s: s[(s.dimension == "resampling") & (s.k == 25)].set_index("group").phi_mean
    a, b, c = f(sf), f(sd), f(sc)
    idx = [i for i in a.index if i in b.index and i in c.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); x = np.arange(len(idx))
    ax.scatter(x - .2, a[idx], c="#1b6ca8", label="full cohort (121)"); ax.scatter(x, b[idx], c="#a4373a", label="Dec-2004 only (105)"); ax.scatter(x + .2, c[idx], c="#888", marker="s", label="size-matched random control (105)")
    ax.set_xticks(x); ax.set_xticklabels(idx, rotation=90); ax.set_ylabel("resampling Φ (k = 25)"); ax.legend()
    ax.set_title("Figure 9. Processing-date sensitivity: does the Dec-2004 subset differ more than a size-matched random subset?", loc="left", fontsize=8.5)
    save(fig, "fig09_date_sensitivity")


if __name__ == "__main__":
    fig_workflow(); fig_cohort()
    full = latest("results/matrix/*_full_matrix_*")
    if full and (full / "analysis/stability_summary.csv").exists():
        fig_performance(full); fig_stability(full); fig_heatmap(full); fig_top_genes(full)
        nd = sorted(glob.glob(str(ROOT / "results/null/main/rep_*/manifest.json")))
        if len(nd) >= 5:
            fig_null(full, [pathlib.Path(p).parent for p in nd])
        dec, ctrl = latest("results/matrix/*_date_dec2004_*"), latest("results/matrix/*_date_control_*")
        if dec and ctrl and (dec / "analysis").exists() and (ctrl / "analysis").exists():
            fig_date(full, dec, ctrl)
    ext = latest("results/external/*_gse65194")
    if ext and (ext / "summary.json").exists():
        fig_external(ext)
