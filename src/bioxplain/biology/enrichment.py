"""Over-representation analysis with a transparent hypergeometric test (Addendum A5).

Libraries are downloaded once via gseapy.get_library (Enrichr), cached with a SHA-256 and date. Background = the eligible
gene universe intersected with the library's genes; the foreground is restricted the same way; terms with 5-500 background genes
are tested; Benjamini-Hochberg is applied within a library. Results are contextual evidence, never causal.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def fetch_library(name: str, cache_dir) -> tuple[dict[str, list[str]], dict]:
    """(term -> gene symbols, provenance). Cached under cache_dir/<name>.json so results do not drift between runs."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{name}.json"
    if not f.exists():
        import gseapy

        lib = gseapy.get_library(name)
        f.write_text(json.dumps({"downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "gseapy_version": gseapy.__version__,
                                 "library": name, "terms": lib}, sort_keys=True))
    blob = json.loads(f.read_text())
    meta = {"library": name, "downloaded_utc": blob["downloaded_utc"], "gseapy_version": blob["gseapy_version"], "n_terms": len(blob["terms"]),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return blob["terms"], meta


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def enrich(foreground, background, library: dict[str, list[str]], min_size: int = 5, max_size: int = 500) -> pd.DataFrame:
    lib_genes = set().union(*map(set, library.values()))
    bg = set(background) & lib_genes
    fg = set(foreground) & bg
    N, n = len(bg), len(fg)
    rows = []
    for term, genes in library.items():
        tg = set(genes) & bg
        K = len(tg)
        if K < min_size or K > max_size:
            continue
        k = len(fg & tg)
        p = float(hypergeom.sf(k - 1, N, K, n)) if n else 1.0
        odds = ((k + 0.5) * (N - K - n + k + 0.5)) / ((K - k + 0.5) * (n - k + 0.5))       # Haldane-Anscombe corrected
        rows.append({"term": term, "term_size_in_background": K, "overlap": k, "foreground_size": n, "background_size": N,
                     "odds_ratio": odds, "p_value": p, "overlap_genes": ";".join(sorted(fg & tg))})
    df = pd.DataFrame(rows)
    if len(df):
        df["adj_p_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
        df = df.sort_values(["adj_p_bh", "p_value", "term"]).reset_index(drop=True)
    return df
