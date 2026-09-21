"""External cohort (GSE65194) access. The ONLY entry point is `load_external`, which first verifies the discovery freeze.

Analysis units (Addendum A4 / decisions D3, D4): 130 unique tumours (the two arrays of each duplicated tumour averaged per probe)
+ 11 healthy; the 14 cell lines are excluded. Rules use metadata only.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bioxplain.data.geo import read_series_matrix, sample_table
from bioxplain.freeze import FREEZE_JSON, FreezeError
from bioxplain.utils.provenance import sha256_file, verify_file

TUMOUR_GROUPS = ("TNBC", "Her2", "Luminal A", "Luminal B")


@dataclass
class ExternalData:
    X: pd.DataFrame          # units x probes (log-scale values as deposited; NOT comparable in scale to the discovery cohort)
    y: np.ndarray            # 1 = cancer, 0 = normal
    units: pd.DataFrame      # unit_id, subtype, n_arrays
    n_arrays_used: int


def verify_freeze(root) -> dict:
    root = pathlib.Path(root)
    f = root / FREEZE_JSON
    if not f.exists():
        raise FreezeError("discovery freeze not found: the external cohort may only be opened after the freeze")
    fz = json.loads(f.read_text())
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", fz["git"]["commit"], "HEAD"], cwd=root, capture_output=True).returncode == 0
    if not ok:
        raise FreezeError(f"freeze commit {fz['git']['commit']} is not an ancestor of HEAD")
    if sha256_file(root / fz["frozen_genes_file"]) != fz["frozen_genes_sha256"]:
        raise FreezeError("frozen gene table was modified after the freeze")
    return fz


def build_units(header, expr: pd.DataFrame, variant: str = "mean") -> ExternalData:
    """Apply the pre-specified unit rules to a parsed GSE65194-style series matrix (also used on toy data in tests)."""
    meta = sample_table(header)
    g = meta["sample_group"]
    keep = g.isin(TUMOUR_GROUPS) | g.eq("Healthy")                                   # cell lines (and anything else) excluded
    meta = meta[keep].copy()
    meta["tumour_id"] = meta["title"].str.extract(r"(TUM\d+)")[0]
    meta["unit"] = np.where(meta["sample_group"].isin(TUMOUR_GROUPS), meta["tumour_id"], meta.index)
    if meta.loc[meta.sample_group.isin(TUMOUR_GROUPS), "tumour_id"].isna().any():
        raise ValueError("a tumour array has no TUMnnn identifier")
    if variant == "repA":                                                            # sensitivity: keep only the first array of duplicated tumours
        is_b = meta["title"].str.contains(r"_repB$")
        meta = meta[~is_b]
    elif variant != "mean":
        raise ValueError(variant)
    cols, ys, rows = {}, {}, []
    for unit, grp in meta.groupby("unit", sort=True):
        cols[unit] = expr[grp.index].mean(axis=1).to_numpy()
        ys[unit] = int(grp["sample_group"].iloc[0] in TUMOUR_GROUPS)
        rows.append({"unit_id": unit, "subtype": grp["sample_group"].iloc[0], "n_arrays": len(grp)})
    X = pd.DataFrame(cols, index=expr.index).T
    units = pd.DataFrame(rows).set_index("unit_id").loc[X.index]
    return ExternalData(X=X, y=np.array([ys[u] for u in X.index]), units=units, n_arrays_used=int(len(meta)))


def load_external(root, variant: str = "mean") -> ExternalData:
    root = pathlib.Path(root)
    verify_freeze(root)                                                              # <- the gate
    man = json.loads((root / "configs/data_manifest.json").read_text())["files"]["GSE65194"]
    path = root / man["path"]
    verify_file(path, man["sha256"])
    header, expr = read_series_matrix(path)
    return build_units(header, expr, variant)
