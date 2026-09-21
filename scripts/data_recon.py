"""Data reconnaissance for BioXplain (metadata + structure only; NO modelling, NO gene selection).

Run from the repo root:   .venv\\Scripts\\python scripts\\data_recon.py

Reads the GEO files under data/ and writes
  results/metrics/data_recon.json                 machine-readable summary of every number reported
  results/metrics/sample_manifest_GSE42568.csv    per-sample labels + provenance fields
  results/metrics/sample_manifest_GSE65194.csv    per-sample labels + a-priori inclusion decision

Design rule: for the external cohort (GSE65194) only structural facts are computed (counts, labels,
value scale, ID overlap, replicate similarity). No gene-level statistic, class contrast, model or
performance number is computed on GSE65194. The inclusion rules written to its manifest depend on
metadata only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import re

import numpy as np
import pandas as pd

from bioxplain.data.geo import read_gpl_annotation, read_series_matrix
from bioxplain.data.geo import sample_table as characteristics

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = {
    "GSE42568": ROOT / "data/raw/GSE42568/GSE42568_series_matrix.txt.gz",
    "GSE65194": ROOT / "data/external/GSE65194/GSE65194_series_matrix.txt.gz",
}
GPL = ROOT / "data/raw/GPL570/GPL570.annot.gz"
TUMOUR_GROUPS = ["TNBC", "Her2", "Luminal A", "Luminal B"]


def scale_summary(expr: pd.DataFrame) -> dict:
    v = expr.to_numpy(float)
    mn = float(v.min())
    q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": round(mn, 4), "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4),
        "quantiles_1_5_25_50_75_95_99": [round(float(x), 4) for x in q],
        "fraction_of_values_equal_to_min": round(float(np.mean(v == mn)), 4),
        "fraction_below_2": round(float(np.mean(v < 2)), 4),
        "n_negative": int((v < 0).sum()),
        "per_sample_median_range": [round(float(expr.median().min()), 3), round(float(expr.median().max()), 3)],
        "n_missing": int(np.isnan(v).sum()),
        "n_duplicate_probe_ids": int(expr.index.duplicated().sum()),
        "n_control_probes_AFFX": int(expr.index.str.startswith("AFFX").sum()),
    }


def col_hashes(expr: pd.DataFrame, rows=None, nd=None) -> pd.Series:
    sub = expr if rows is None else expr.loc[rows]
    return pd.Series({c: hashlib.md5((np.round(sub[c].to_numpy(float), nd) if nd else sub[c].to_numpy(float)).tobytes()).hexdigest()
                      for c in sub.columns})


def main() -> None:
    out: dict = {"files": {}}
    for acc, p in list(FILES.items()) + [("GPL570", GPL)]:
        with gzip.open(p, "rb") as fh:                     # full decompress verifies CRC32/length
            n = sum(len(c) for c in iter(lambda: fh.read(1 << 20), b""))
        out["files"][acc] = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "decompressed_bytes": n}

    h1, e1 = read_series_matrix(FILES["GSE42568"]); m1 = characteristics(h1)
    h2, e2 = read_series_matrix(FILES["GSE65194"]); m2 = characteristics(h2)
    ann = read_gpl_annotation(GPL)

    # ---------------- GSE42568 (discovery) ----------------
    cls1 = m1["tissue"].map({"breast cancer": "cancer", "normal breast": "normal"})
    assert cls1.notna().all()
    consistent = (m1["source_name"].str.contains("cancer") == (cls1 == "cancer")).all() and \
                 (m1["title"].str.startswith("Breast cancer") == (cls1 == "cancer")).all() and \
                 (m1["description"].str.contains("cancer") == (cls1 == "cancer")).all()
    parsed = m1["title"].str.replace(r"^(Normal breast|Breast cancer), ", "", regex=True) \
        .str.extract(r"^([A-Za-z]+)(\w+?)_(\d+)_(\d+)_(\d+)$")
    m1["title_prefix"] = parsed[0]
    m1["date_ddmmyy"] = parsed[2] + "-" + parsed[3] + "-" + parsed[4]
    m1["date_yymm"] = parsed[4].str.zfill(2) + "-" + parsed[3].str.zfill(2)
    m1["label"] = cls1
    v1 = e1.to_numpy(float)
    frac_floor = (v1 == v1.min()).mean(axis=1)
    gsm_order = "".join("N" if c == "normal" else "T" for c in cls1)
    out["GSE42568"] = {
        "n_samples": int(e1.shape[1]), "n_probes": int(e1.shape[0]),
        "platform": header_val(h1, "!Series_platform_id"),
        "processing": h1["!Sample_data_processing"][0][0],
        "label_counts": cls1.value_counts().to_dict(),
        "label_fields_all_agree(tissue,source_name,title,description)": bool(consistent),
        "matrix_columns_match_header_order": list(e1.columns) == list(m1.index),
        "gsm_range": [m1.index.min(), m1.index.max()],
        "gsm_order_is_class_blocked": gsm_order,
        "scale": scale_summary(e1),
        "duplicate_identical_columns": int(col_hashes(e1).duplicated().sum()),
        "max_offdiag_sample_pearson": round(float(np.nanmax(np.where(np.eye(e1.shape[1], dtype=bool), np.nan, np.corrcoef(v1.T)))), 5),
        "floor_probes_all_samples": int((frac_floor == 1).sum()),
        "floor_probes_ge90pct": int((frac_floor >= 0.9).sum()),
        "floor_probes_ge50pct": int((frac_floor >= 0.5).sum()),
        "zero_variance_probes": int((v1.std(axis=1) == 0).sum()),
        "title_prefix_by_class": pd.crosstab(m1["title_prefix"], cls1).to_dict(),
        "date_yymm_by_class": pd.crosstab(m1["date_yymm"], cls1).to_dict(),
        "tumour_er_status": m1.loc[cls1 == "cancer", "er_status"].value_counts().to_dict(),
        "tumour_grade": m1.loc[cls1 == "cancer", "grade"].value_counts().to_dict(),
    }

    # ---------------- GSE65194 (external; structure only) ----------------
    g2 = m2["sample_group"]
    m2["tumour_id"] = m2["title"].str.extract(r"(TUM\d+)")[0]
    m2["is_replicate_array"] = m2["title"].str.contains(r"_rep[A-Z]$")
    n_arrays_per_tumour = m2[g2.isin(TUMOUR_GROUPS)].groupby("tumour_id").size()
    pairs = [g.index.tolist() for _, g in m2[g2.isin(TUMOUR_GROUPS)].groupby("tumour_id") if len(g) == 2]
    pr = [np.corrcoef(e2[a].to_numpy(float), e2[b].to_numpy(float))[0, 1] for a, b in pairs]
    # a-priori (metadata-only) inclusion rules for the future external evaluation
    m2["label"] = np.select([g2.isin(TUMOUR_GROUPS), g2.eq("Healthy")], ["cancer", "normal"], default="cell_line")
    m2["include_external"] = m2["label"].isin(["cancer", "normal"])
    m2["exclusion_reason"] = np.where(m2["label"].eq("cell_line"), "cell line (not tissue)", "")
    m2["collapse_unit"] = np.where(m2["label"].eq("cancer"), m2["tumour_id"], m2.index)
    sh = e1.index.intersection(e2.index)
    out["GSE65194"] = {
        "n_arrays": int(e2.shape[1]), "n_probes": int(e2.shape[0]),
        "platform": header_val(h2, "!Series_platform_id"),
        "processing": h2["!Sample_data_processing"][0][0],
        "sample_group_counts_arrays": g2.value_counts().to_dict(),
        "tumour_arrays": int(g2.isin(TUMOUR_GROUPS).sum()),
        "unique_tumours": int(n_arrays_per_tumour.size),
        "tumours_with_two_arrays": int((n_arrays_per_tumour == 2).sum()),
        "unique_tumours_by_subtype": m2[g2.isin(TUMOUR_GROUPS)].drop_duplicates("tumour_id")["sample_group"].value_counts().to_dict(),
        "replicate_pairs_bit_identical": int(sum(np.array_equal(e2[a].to_numpy(float), e2[b].to_numpy(float)) for a, b in pairs)),
        "replicate_pair_pearson_min_median_max": [round(float(min(pr)), 4), round(float(np.median(pr)), 4), round(float(max(pr)), 4)],
        "cell_lines": m2.loc[g2.eq("CellLine"), "title"].str.replace("CellLine, ", "").tolist(),
        "array_batch_by_group": pd.crosstab(g2, m2["array_batch"]).to_dict(),
        "matrix_columns_match_header_order": list(e2.columns) == list(m2.index),
        "scale": scale_summary(e2),
        "duplicate_identical_columns": int(col_hashes(e2).duplicated().sum()),
        "would_be_analysis_set": {"cancer_unique_tumours": int(n_arrays_per_tumour.size), "normal": int(g2.eq("Healthy").sum())},
    }

    # ---------------- cross-series ----------------
    p1, p2 = set(e1.index), set(e2.index)
    out["cross_series"] = {
        "shared_gsm": len(set(e1.columns) & set(e2.columns)),
        "shared_titles": len(set(m1.title) & set(m2.title)),
        "probe_ids_shared": len(p1 & p2), "probes_only_in_GSE42568": sorted(p1 - p2), "probes_only_in_GSE65194": sorted(p2 - p1),
        "shared_probe_row_order_identical": list(e1.index.intersection(e2.index)) == [i for i in e1.index if i in p2],
        "identical_columns_across_series_4dp": len(set(col_hashes(e1, sh, 4)) & set(col_hashes(e2, sh, 4))),
    }

    # ---------------- GPL570 ----------------
    sym, gid = ann["Gene symbol"].str.strip(), ann["Gene ID"].str.strip()
    empty = sym.isin(["", "---"]); multi = sym.str.contains("///", regex=False); ctrl = ann["ID"].str.startswith("AFFX")
    usable = ~empty & ~multi & ~ctrl
    vc = ann.loc[usable, "Gene symbol"].value_counts()
    out["GPL570"] = {
        "records": int(len(ann)), "unique_probe_ids": int(ann["ID"].nunique()),
        "annotation_ids_equal_GSE42568_matrix_ids": set(ann["ID"]) == p1,
        "annotation_date": "Aug 09 2016 (per file header)", "id_column": "ID", "symbol_column": "Gene symbol", "entrez_column": "Gene ID",
        "empty_symbol": int(empty.sum()), "multi_gene_probes": int(multi.sum()), "affx_control": int(ctrl.sum()),
        "usable_single_gene_probes": int(usable.sum()), "usable_fraction": round(float(usable.mean()), 4),
        "usable_if_multi_gene_exploded": int((~empty & ~ctrl).sum()),
        "unique_symbols_single_gene": int(vc.size), "genes_1_probe": int((vc == 1).sum()), "genes_ge2_probes": int((vc >= 2).sum()),
        "genes_ge5_probes": int((vc >= 5).sum()), "max_probes_per_gene": int(vc.max()),
        "fraction_usable_probes_in_multiprobe_genes": round(float(vc[vc >= 2].sum() / usable.sum()), 4),
        "symbols_with_gt1_entrez": int((ann[usable].groupby("Gene symbol")["Gene ID"].nunique() > 1).sum()),
        "usable_probes_in_both_series": int(ann.loc[usable, "ID"].isin(p1 & p2).sum()),
    }

    # ---------------- manifests + json ----------------
    (ROOT / "results/metrics").mkdir(parents=True, exist_ok=True)
    m1[["title", "label", "tissue", "er_status", "grade", "title_prefix", "date_ddmmyy"]].to_csv(ROOT / "results/metrics/sample_manifest_GSE42568.csv", index_label="gsm")
    m2[["title", "label", "sample_group", "tumour_id", "is_replicate_array", "array_batch", "hybridation batch",
        "include_external", "exclusion_reason", "collapse_unit"]].to_csv(ROOT / "results/metrics/sample_manifest_GSE65194.csv", index_label="gsm")
    (ROOT / "results/metrics/data_recon.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


def header_val(h, key):
    return sorted({x for row in h[key] for x in row})


if __name__ == "__main__":
    main()
