"""Loader for the discovery cohort (GSE42568). Never touches the external cohort."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bioxplain.data.geo import read_series_matrix, sample_table
from bioxplain.data.labels import derive_cancer_normal_labels
from bioxplain.utils.provenance import verify_file


@dataclass
class DiscoveryData:
    X: pd.DataFrame            # samples x probes (log2-like GC-RMA), restricted to the feature universe
    y: pd.Series               # 1 = cancer, 0 = normal (indexed like X)
    samples: pd.DataFrame      # metadata
    probe_to_gene: pd.Series   # probe_id -> gene symbol (feature universe, decision D5)
    entrez: pd.Series          # gene symbol -> Entrez ID


def load_feature_universe(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def load_discovery(root: str | pathlib.Path, verify: bool = True) -> DiscoveryData:
    """Load GSE42568 restricted to the pre-built feature universe (configs/feature_universe.tsv)."""
    root = pathlib.Path(root)
    manifest = json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE42568"]
    path = root / manifest["path"]
    if verify:
        verify_file(path, manifest["sha256"])
    header, expr = read_series_matrix(path)
    samples = sample_table(header)
    y = derive_cancer_normal_labels(samples)
    if list(expr.columns) != list(samples.index):
        raise ValueError("expression columns and metadata rows are in different order")
    universe = load_feature_universe(root / "configs/feature_universe.tsv")
    missing = set(universe.probe_id) - set(expr.index)
    if missing:
        raise ValueError(f"{len(missing)} universe probes absent from the discovery matrix")
    X = expr.loc[universe.probe_id].T.astype(float)
    X.index.name = "gsm"
    probe_to_gene = universe.set_index("probe_id")["gene_symbol"]
    entrez = universe.drop_duplicates("gene_symbol").set_index("gene_symbol")["entrez_id"]
    return DiscoveryData(X=X, y=y.loc[X.index], samples=samples, probe_to_gene=probe_to_gene, entrez=entrez)


def processing_month(titles: pd.Series) -> pd.Series:
    """'yy-mm' processing month parsed from GSE42568 sample titles like 'Breast cancer, T98_22_12_04' (day_month_year);
    NaN where the title carries no date (3 tumours)."""
    core = titles.str.replace(r"^(Normal breast|Breast cancer), ", "", regex=True)
    parts = core.str.extract(r"^[A-Za-z]+\w+?_(\d+)_(\d+)_(\d+)$")
    month = parts[2].str.zfill(2) + "-" + parts[1].str.zfill(2)
    return month


def restrict_subset(data: DiscoveryData, subset: str, seed: int = 0) -> DiscoveryData:
    """'full' | 'dec2004' (pre-specified processing-date sensitivity: samples processed 2004-12; undated tumours excluded) |
    'random_control' (a size-matched random subset with the same class counts as dec2004: a control for sample size alone)."""
    if subset == "full":
        return data
    month = processing_month(data.samples["title"]).reindex(data.X.index)
    dec = (month == "04-12").to_numpy()
    if subset == "dec2004":
        keep = dec
    elif subset == "random_control":
        rng = np.random.default_rng(seed)
        y = data.y.to_numpy()
        n0, n1 = int(((y == 0) & dec).sum()), int(((y == 1) & dec).sum())
        keep = np.zeros(len(y), bool)
        keep[rng.choice(np.flatnonzero(y == 0), n0, replace=False)] = True
        keep[rng.choice(np.flatnonzero(y == 1), n1, replace=False)] = True
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return DiscoveryData(X=data.X.loc[keep], y=data.y.loc[keep], samples=data.samples.loc[keep], probe_to_gene=data.probe_to_gene, entrez=data.entrez)
