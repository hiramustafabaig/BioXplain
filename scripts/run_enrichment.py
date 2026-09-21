"""python scripts/run_enrichment.py  ->  results/enrichment/

Addendum A5: runs once after the freeze, on the frozen lists only. Background = frozen eligible gene universe intersected with each
library's genes; hypergeometric test; Benjamini-Hochberg within library.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from bioxplain.biology.enrichment import enrich, fetch_library
from bioxplain.utils.provenance import git_state, software_versions, utc_now

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBRARIES = ("GO_Biological_Process_2023", "Reactome_2022", "MSigDB_Hallmark_2020")

if __name__ == "__main__":
    fz = json.loads((ROOT / "docs/freeze/discovery_freeze.json").read_text())
    frozen = pd.read_csv(ROOT / fz["frozen_genes_file"])
    background = frozen["gene"].tolist()
    fgs = {"S_top50": frozen.loc[frozen.in_S_top50, "gene"].tolist()}
    if frozen.in_S_cons.sum() >= 10:
        fgs["S_cons"] = frozen.loc[frozen.in_S_cons, "gene"].tolist()
    out = ROOT / "results/enrichment"
    out.mkdir(parents=True, exist_ok=True)
    meta, summary = [], []
    for lib_name in LIBRARIES:
        lib, m = fetch_library(lib_name, ROOT / "data/external/gene_sets")
        meta.append(m)
        for fg_name, fg in fgs.items():
            df = enrich(fg, background, lib)
            df.to_csv(out / f"{fg_name}__{lib_name}.csv", index=False)
            summary.append({
                "foreground": fg_name, "library": lib_name,
                "n_foreground": int(df.foreground_size.iloc[0]) if len(df) else 0,
                "n_background": int(df.background_size.iloc[0]) if len(df) else 0,
                "n_terms_tested": len(df),
                "n_adj_p_lt_0.05": int((df.adj_p_bh < 0.05).sum()) if len(df) else 0,
                "min_adj_p": float(df.adj_p_bh.min()) if len(df) else None,
            })
    pd.DataFrame(summary).to_csv(out / "enrichment_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "utc": utc_now().isoformat(), "libraries": meta, "background_definition": "frozen eligible gene universe intersected with library genes",
        "min_term_size": 5, "max_term_size": 500, "correction": "Benjamini-Hochberg within library", "git": git_state(ROOT),
        "software": software_versions(), "foreground_lists": {k: len(v) for k, v in fgs.items()},
    }, indent=2, default=str))
    print(pd.DataFrame(summary).to_string(index=False))
