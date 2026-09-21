"""Synthetic data for fast unit tests (no real GEO data needed)."""
from __future__ import annotations

import gzip
import pathlib

import numpy as np
import pandas as pd

FLOOR = 2.3


def make_toy_cohort(seed=0, n_normal=12, n_cancer=48, n_signal=6, n_noise=120, n_lowexp=30, signal_shift=3.0):
    """Toy GC-RMA-like cohort.

    Genes: ``S*`` signal genes (cancer-shifted, alternating up/down), ``N*`` noise genes, ``L*`` low-expression genes
    (mostly at the floor). Every gene has 1-3 probes. Returns (X samples x probes, y, probe_to_gene, signal_genes).
    """
    rng = np.random.default_rng(seed)
    n = n_normal + n_cancer
    y = np.r_[np.zeros(n_normal, int), np.ones(n_cancer, int)]     # class-blocked, like GEO order
    genes = [f"S{i}" for i in range(n_signal)] + [f"N{i}" for i in range(n_noise)] + [f"L{i}" for i in range(n_lowexp)]
    cols, mapping = {}, {}
    for gi, g in enumerate(genes):
        for pj in range(1 + (gi % 3)):
            probe = f"{g}_p{pj}_at"
            base = 7.0 + rng.normal(0, 0.5)
            x = base + rng.normal(0, 0.8, n)
            if g.startswith("S"):
                sign = 1 if int(g[1:]) % 2 == 0 else -1
                x = x + sign * signal_shift * y
            if g.startswith("L"):
                x = 0.8 + rng.normal(0, 0.6, n)                     # ~0.6% detected -> almost all clipped
            cols[probe] = np.maximum(x, FLOOR)                      # GC-RMA-like hard floor
            mapping[probe] = g
    X = pd.DataFrame(cols, index=[f"GSM{i:04d}" for i in range(n)])
    return X, y, pd.Series(mapping), [g for g in genes if g.startswith("S")]


def write_toy_series_matrix(path: pathlib.Path, *, mislabel_description=False, tissue_values=None) -> pathlib.Path:
    """Write a tiny GEO-style series matrix (3 normal + 3 cancer samples, 4 probes)."""
    gsm = [f"GSM{i}" for i in range(1, 7)]
    tissue = tissue_values or ["normal breast"] * 3 + ["breast cancer"] * 3
    titles = [f"Normal breast, N{i}" for i in range(3)] + [f"Breast cancer, T{i}" for i in range(3)]
    src = ["Breast tissue, normal"] * 3 + ["Breast tissue, cancer"] * 3
    desc = ["Normal breast tissue data."] * 3 + ["Breast cancer data."] * 3
    if mislabel_description:
        desc[3] = "Normal breast tissue data."

    def q(vals):
        return "\t".join(f'"{v}"' for v in vals)

    lines = [
        '!Series_title\t"Toy"',
        '!Series_geo_accession\t"GSETOY"',
        "!Sample_title\t" + q(titles),
        "!Sample_geo_accession\t" + q(gsm),
        "!Sample_source_name_ch1\t" + q(src),
        "!Sample_characteristics_ch1\t" + q([f"tissue: {t}" for t in tissue]),
        "!Sample_characteristics_ch1\t" + q(["age: NA", "age: NA", "age: NA", "age: 50", "age: 61", "age: 70"]),
        "!Sample_description\t" + q(desc),
        "!series_matrix_table_begin",
        '"ID_REF"\t' + q(gsm),
    ]
    rng = np.random.default_rng(0)
    for p in ["1000_at", "1001_at", "1002_s_at", "AFFX-x_at"]:
        lines.append(f'"{p}"\t' + "\t".join(f"{v:.4f}" for v in rng.normal(6, 1, 6)))
    lines.append("!series_matrix_table_end")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_toy_annotation(path: pathlib.Path) -> pathlib.Path:
    header = ["^Annotation", "!Annotation_platform = GPLTOY", "!platform_table_begin",
              "ID\tGene title\tGene symbol\tGene ID"]
    rows = [
        "1000_at\tt\tGENEA\t1",
        "1001_at\tt\tGENEB///GENEC\t2///3",     # multi-gene  -> unusable
        "1002_s_at\tt\t---\t---",                # unmapped    -> unusable
        "AFFX-x_at\tt\tCTRL\t9",                 # control     -> unusable
        "1003_at\tt\tGENEA\t1",                  # 2nd probe of GENEA -> usable
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(header + rows + ["!platform_table_end"]) + "\n")
    return path
