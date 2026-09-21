"""Parsers for NCBI GEO series-matrix files and GPL annotation tables.

Only structure/metadata handling lives here; no statistics.
"""
from __future__ import annotations

import collections
import gzip
import io
import pathlib

import pandas as pd

Header = dict[str, list[list[str]]]


def read_series_matrix(path: str | pathlib.Path) -> tuple[Header, pd.DataFrame]:
    """Parse a GEO series matrix.

    Returns ``(header, expression)`` where ``header`` maps each ``!key`` to a list of value lists (a key may
    repeat, e.g. ``!Sample_characteristics_ch1``) and ``expression`` is probes x samples (GSM columns).
    """
    header: Header = collections.defaultdict(list)
    table_lines: list[str] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)
            elif line.startswith("!"):
                parts = line.split("\t")
                header[parts[0]].append([p.strip('"') for p in parts[1:]])
    if not table_lines:
        raise ValueError(f"{path}: no expression table found between series_matrix_table markers")
    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0, quotechar='"')
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [c.strip('"') for c in expr.columns]
    return dict(header), expr


def read_probe_ids(path: str | pathlib.Path) -> list[str]:
    """Probe IDs (first column of the expression table) only, without loading values."""
    ids: list[str] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                next(fh)  # column header row
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                ids.append(line.split("\t", 1)[0].strip('"'))
    return ids


def sample_table(header: Header) -> pd.DataFrame:
    """Per-sample metadata table indexed by GSM.

    Repeated ``!Sample_characteristics_ch1`` rows ("key: value") become one column each.
    """
    gsm = header["!Sample_geo_accession"][0]
    cols: dict[str, list[str]] = {}
    for row in header.get("!Sample_characteristics_ch1", []):
        keys = {r.split(":", 1)[0].strip() for r in row if ":" in r}
        key = keys.pop() if len(keys) == 1 else "|".join(sorted(keys))
        cols[key] = [r.split(":", 1)[1].strip() if ":" in r else r.strip() for r in row]
    df = pd.DataFrame(cols, index=gsm)
    df.insert(0, "title", header["!Sample_title"][0])
    df.insert(1, "source_name", header["!Sample_source_name_ch1"][0])
    df.insert(2, "description", header.get("!Sample_description", [[""] * len(gsm)])[0])
    return df


def read_gpl_annotation(path: str | pathlib.Path) -> pd.DataFrame:
    """Parse a ``GPLxxx.annot.gz`` file into a DataFrame of strings (one row per probe)."""
    rows: list[str] = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_table = True
            elif line.startswith("!platform_table_end"):
                in_table = False
            elif in_table:
                rows.append(line.rstrip("\n"))
    if not rows:
        raise ValueError(f"{path}: no platform table found")
    return pd.read_csv(io.StringIO("\n".join(rows)), sep="\t", dtype=str, keep_default_na=False, quoting=3)


def usable_probe_table(annotation: pd.DataFrame) -> pd.DataFrame:
    """Probes that map to exactly one gene: non-control (AFFX*), non-empty, not multi-gene ('///').

    Returns columns ``probe_id, gene_symbol, entrez_id``.
    """
    sym = annotation["Gene symbol"].str.strip()
    ids = annotation["Gene ID"].str.strip()
    ok = (
        ~sym.isin(["", "---"])
        & ~sym.str.contains("///", regex=False)
        & ~annotation["ID"].str.startswith("AFFX")
    )
    out = pd.DataFrame(
        {"probe_id": annotation.loc[ok, "ID"].values, "gene_symbol": sym[ok].values, "entrez_id": ids[ok].values}
    )
    return out
