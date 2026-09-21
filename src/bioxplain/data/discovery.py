"""Loader for the discovery cohort (GSE42568). Never touches the external cohort."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

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
