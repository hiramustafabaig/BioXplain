"""External replication analysis on GSE65194 (Addendum A4/A6). Runs ONLY after the discovery freeze (gated by `load_external`).

All statistics are scale-free; discovery-side quantities come from the frozen table. Nothing here changes any discovery decision.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from bioxplain.biology.composition import EXTENDED_PANEL, PRIMARY_PANEL, panel_membership
from bioxplain.data.discovery import load_discovery
from bioxplain.external import ExternalData, load_external, verify_freeze
from bioxplain.replication import (auc_effects, bootstrap_auc_ci, bootstrap_c1_c2, c1_c2, direction_consistency, frozen_lr_transfer, hedges_g,
                                   percentile_ci, residual_auc, signature_score, within_cohort_z)
from bioxplain.utils.provenance import git_state, software_versions, utc_now

LISTS = ("S_cons", "S_top25", "S_top50")
PANEL = PRIMARY_PANEL + EXTENDED_PANEL


def _gene_level(frozen: pd.DataFrame, ext: ExternalData, n_boot: int, seed: int, tag: str) -> tuple[dict, pd.DataFrame]:
    Xe = ext.X.loc[:, frozen["probe"]].to_numpy()
    auc_e = auc_effects(Xe, ext.y)
    e_d, s = frozen["e_disc"].to_numpy(), frozen["s_g_k25"].to_numpy()
    point = c1_c2(s, e_d, auc_e)
    res = {"tag": tag, **point, "spearman_e_disc_e_ext_all_genes": float(stats.spearmanr(e_d, auc_e - 0.5).statistic),
           "direction_all_genes": direction_consistency(e_d, auc_e - 0.5), "n_units": int(len(ext.y)), "n_normal": int((ext.y == 0).sum())}
    if n_boot:
        bs = bootstrap_c1_c2(s, e_d, Xe, ext.y, n_boot, seed)
        for k in ("c1_spearman", "c1_spearman_selected_only", "c2_rank_coef_stability", "c2_partial_spearman"):
            res[f"{k}_ci"] = percentile_ci(bs[k])
    per_gene = frozen[["gene", "probe", "e_disc", "s_g_k25"]].assign(auc_ext=auc_e, e_ext=auc_e - 0.5, aligned_effect=np.sign(e_d) * (auc_e - 0.5))
    return res, per_gene


def _list_level(frozen: pd.DataFrame, per_gene: pd.DataFrame, ext: ExternalData, disc_Z: np.ndarray, disc_y: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame:
    Xe = ext.X.loc[:, frozen["probe"]].to_numpy()
    Ze = within_cohort_z(Xe)
    rows = []
    for name in LISTS:
        m = frozen[f"in_{name}"].to_numpy()
        if m.sum() < 2:
            rows.append({"list": name, "n_genes": int(m.sum()), "note": "too few genes for list-level statistics"})
            continue
        e_d, e_e = frozen.loc[m, "e_disc"].to_numpy(), per_gene.loc[m, "e_ext"].to_numpy()
        up, down = np.flatnonzero(e_d[np.newaxis].ravel() > 0), np.flatnonzero(e_d < 0)
        idx = np.flatnonzero(m)
        sc_e = signature_score(Ze[:, idx], up, down); sc_d = signature_score(disc_Z[:, idx], up, down)
        ge = hedges_g(Xe[ext.y == 1][:, idx], Xe[ext.y == 0][:, idx])
        rows.append({"list": name, "n_genes": int(m.sum()), "direction": direction_consistency(e_d, e_e), "mean_aligned_effect": float(np.mean(np.sign(e_d) * e_e)),
                     "spearman_e_disc_e_ext": float(stats.spearmanr(e_d, e_e).statistic) if m.sum() > 2 else np.nan,
                     "mean_aligned_hedges_g_ext": float(np.mean(np.sign(e_d) * ge)),
                     "signature_auc_external": float(roc_auc_score(ext.y, sc_e)), "signature_auc_external_ci": bootstrap_auc_ci(ext.y, sc_e, n_boot, seed),
                     "signature_auc_discovery_in_sample": float(roc_auc_score(disc_y, sc_d)), "n_up": int(len(up)), "n_down": int(len(down))})
    return pd.DataFrame(rows)


def run_external_analysis(root, out_dir=None, n_boot: int = 1000, seed: int = 20260921, variants: bool = True) -> pathlib.Path:
    root = pathlib.Path(root)
    t0 = time.perf_counter()
    fz = verify_freeze(root)
    frozen = pd.read_csv(root / fz["frozen_genes_file"])
    disc = load_discovery(root)
    disc_y = disc.y.to_numpy()
    disc_G = disc.X.loc[:, frozen["probe"]].to_numpy()
    disc_Z = within_cohort_z(disc_G)
    ext = load_external(root, "mean")
    out = pathlib.Path(out_dir or root / "results/external" / f"{utc_now():%Y%m%dT%H%M%SZ}_gse65194")
    out.mkdir(parents=True, exist_ok=True)

    primary, per_gene = _gene_level(frozen, ext, n_boot, seed, "primary_mean_of_duplicates")
    per_gene.to_csv(out / "per_gene_external_effects.csv", index=False)
    lists = _list_level(frozen, per_gene, ext, disc_Z, disc_y, n_boot, seed)
    lists.to_csv(out / "list_level_replication.csv", index=False)

    Ge = ext.X.loc[:, frozen["probe"]].to_numpy()
    top25 = frozen["in_S_top25"].to_numpy()
    sample_level = frozen_lr_transfer(disc_G[:, top25], disc_y, Ge[:, top25], ext.y, n_boot, seed)

    sens_rows, sub_rows = [], []
    if variants:
        repA = load_external(root, "repA")
        r, _ = _gene_level(frozen, repA, min(n_boot, 200), seed, "sensitivity_repA_only")
        sens_rows.append(r)
        groups = {"Luminal (A+B)": ["Luminal A", "Luminal B"], "HER2": ["Her2"], "TNBC": ["TNBC"]}
        for label, sub in groups.items():
            keep = ext.units["subtype"].isin(sub + ["Healthy"]).to_numpy()
            e2 = ExternalData(X=ext.X.loc[keep], y=ext.y[keep], units=ext.units.loc[keep], n_arrays_used=int(ext.units.loc[keep, "n_arrays"].sum()))
            r, _ = _gene_level(frozen, e2, min(n_boot, 200), seed, f"subtype_{label}_vs_healthy")
            sub_rows.append(r)
        pd.DataFrame(sens_rows + sub_rows).to_json(out / "sensitivity_gene_level.json", orient="records", indent=2)

    # ---- tissue composition (A6)
    genes = frozen["gene"].tolist()
    present = [g for g in PANEL if g in genes]
    comp = {"panel_genes_present_in_eligible_universe": present}
    if present:
        idx = [genes.index(g) for g in present]
        Ze = within_cohort_z(Ge)
        pe, pd_ = Ze[:, idx].mean(1), disc_Z[:, idx].mean(1)
        sig_e = signature_score(Ze[:, top25], np.flatnonzero(frozen.loc[top25, "e_disc"].to_numpy() > 0), np.flatnonzero(frozen.loc[top25, "e_disc"].to_numpy() < 0))
        sig_d = signature_score(disc_Z[:, top25], np.flatnonzero(frozen.loc[top25, "e_disc"].to_numpy() > 0), np.flatnonzero(frozen.loc[top25, "e_disc"].to_numpy() < 0))
        for tag, y_, p_, s_ in (("external", ext.y, pe, sig_e), ("discovery", disc_y, pd_, sig_d)):
            comp[tag] = {"adipose_panel_auc_cancer_vs_normal": float(roc_auc_score(y_, p_)),
                         "spearman_signature_vs_panel_within_cancer": float(stats.spearmanr(s_[y_ == 1], p_[y_ == 1]).statistic),
                         "spearman_signature_vs_panel_within_normal": float(stats.spearmanr(s_[y_ == 0], p_[y_ == 0]).statistic),
                         "signature_auc": float(roc_auc_score(y_, s_)), "signature_auc_after_removing_panel_component": residual_auc(y_, s_, p_)}
    s_universe = pd.Series(frozen["s_g_k25"].to_numpy(), index=frozen["gene"])
    panel_membership({k: frozen.loc[frozen[f"in_{k}"], "gene"].tolist() for k in LISTS}, s_universe).assign(
        e_disc=lambda d: d.gene.map(frozen.set_index("gene")["e_disc"]), e_ext=lambda d: d.gene.map(per_gene.set_index("gene")["e_ext"])).to_csv(out / "composition_panel.csv", index=False)
    summary = {"primary_gene_level": primary, "sample_level_frozen_lr_S_top25": sample_level, "composition": comp, "n_boot": n_boot}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "manifest.json").write_text(json.dumps({"freeze_commit": fz["git"]["commit"], "git": git_state(root), "software": software_versions(),
                                                    "external_units": {"n": int(len(ext.y)), "n_normal": int((ext.y == 0).sum()), "n_tumour": int(ext.y.sum()),
                                                                       "n_arrays_used": ext.n_arrays_used, "n_duplicated_tumours": int((ext.units.n_arrays == 2).sum())},
                                                    "runtime_seconds": round(time.perf_counter() - t0, 1), "outputs": sorted(p.name for p in out.iterdir())}, indent=2, default=str))
    return out
